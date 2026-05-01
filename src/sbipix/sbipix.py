"""
SBIPIX: Main class for simulation-based inference on pixel-level stellar population properties
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sbi
from sbi import utils as Ut
from sbi import inference as Inference
import pickle
import seaborn as sns
import sklearn.metrics as sm

from scipy import stats
truncnorm=stats.truncnorm
from tqdm import tqdm, trange
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

from .utils.sed_utils import mag_conversion
from .utils.cosmology import setup_cosmology
from .train.simulator import generate_atlas_parametric
from .plotting.diagnostics import plot_test_performance


class sbipix():
    """
    A class for simulation-based inference pipeline for studying stellar population 
    properties on integrated/resolved galaxies from JWST.

    This class provides a complete workflow for:
    1. Simulating galaxy SEDs with various star formation histories
    2. Training neural density estimators using simulation-based inference
    3. Inferring stellar population properties from observed photometry

    Attributes
    ----------
    n_filters : int
        Number of filters used from the filter_list (default: 19)
    filter_list : str
        Text file with paths for the filter files
    filter_path : str
        Path where filter_list is located
    atlas_path : str
        Path where atlas is located
    atlas_name : str
        Name of atlas object
    n_simulation : int
        Number of simulated galaxies for training
    parametric : bool
        If True, use parametric (τ-delayed) SFH; if False, use Dirichlet prior
    both_masses : bool
        If True, include both formed and surviving stellar masses
    infer_z : bool
        If True, infer redshift from photometry
    obs : np.ndarray
        Array of shape (n_simulation, n_filters) with simulated photometry
    theta : np.ndarray
        Array of shape (n_simulation, n_params) with physical properties
    labels : list
        Names of the physical properties in theta
    mag : np.ndarray
        Processed magnitudes ready for training (with noise, masks, limits)
    
    Observational Properties
    -----------------------
    include_sigma : bool
        Include photometric uncertainties in simulation
    include_mask : bool
        Include masking for unavailable filters
    include_limit : bool
        Include detection limits
    condition_sigma : bool
        Include uncertainties as network input
    mean_sigma_obs : np.ndarray
        Mean uncertainty distributions per magnitude bin and filter
    stds_sigma_obs : np.ndarray
        Standard deviation of uncertainty distributions
    percentiles : np.ndarray
        Percentiles for magnitude bins used to assign uncertainties
    limits : np.ndarray
        1σ depth limits for each filter
    
    Model Properties
    ---------------
    model_path : str
        Path for saving/loading trained models
    model_name : str
        Filename for the trained model
    means_test : np.ndarray
        Test set posterior means
    stds_test : np.ndarray
        Test set posterior standard deviations
    
    Observational Data
    -----------------
    catalog_path : str
        Path to observational catalogs
    catalog_name : str
        Name of the observational catalog
    mag_obs : np.ndarray
        Processed observational photometry
    posteriors_obs : np.ndarray
        Inferred posteriors for observed galaxies
    means_obs : np.ndarray
        Posterior means for observed galaxies
    stds_obs : np.ndarray
        Posterior uncertainties for observed galaxies

    Examples
    --------
    >>> # Basic usage for parametric SFH
    >>> model = SBIPIX()
    >>> model.parametric = True
    >>> model.simulate(n_simulation=50000)
    >>> model.load_obs_features()
    >>> model.add_noise_nan_limit_all()
    >>> model.train()
    >>> model.test_performance()
    
    >>> # For resolved galaxy analysis
    >>> posteriors = model.get_posteriors_resolved(phot_data, n_gal=10)
    """

    def __init__(self):
        """Initialize SBIPIX with default configuration for JADES analysis."""
        # Filter and data configuration
        self.n_filters = 19
        self.filter_list = 'filters_jades_no_wfc.dat'
        self.filter_path = '../obs/obs_properties/'
        self.atlas_name = 'atlas_obs_jades_no_wfc'
        self.atlas_path = './library/'
        
        # Simulation parameters
        self.n_simulation = 100000
        self.parametric = False  # Use Dirichlet by default
        self.both_masses = False
        self.remove_filters = None
        
        # Data arrays (initialized as None)
        self.obs = None
        self.theta = None
        self.mag = None
        
        # Parameter labels (updated based on SFH type)
        self.labels = [
            'log($\\rm{M}_{*}/\\rm{M}_{\\odot}$)',
            'log(SFR/($\\rm{M}_{\\odot}$/yr))',
            '$t_{25\\%}$', '$t_{50\\%}$', '$t_{75\\%}$',
            '[M/H]', 'Av', 'z'
        ]
        
        # Observational realism parameters
        self.include_sigma = False
        self.include_mask = False
        self.include_limit = False
        self.condition_sigma = False
        
        # Observational properties (loaded from files)
        self.mean_sigma_obs = None
        self.stds_sigma_obs = None
        self.percentiles = None
        self.limits = None
        
        # Model configuration
        self.model_path = "./library/"
        self.model_name = "posteriors.pkl"
        self.infer_z = True
        self.infer_z_integrated = False
        
        # Results storage
        self.means_test = None
        self.stds_test = None
        self.mode_test = None
        
        # Observational data
        self.catalog_path = './JADES/'
        self.catalog_name = 'ra_dec_mach_phot_spec_z.fits'
        self.mag_obs = None
        self.flags_obs = None
        self.id_specz = None
        self.id_photoz = None
        self.posteriors_obs = None
        self.means_obs = None
        self.stds_obs = None
        self.mode_obs = None
        self.ind_obs = None
        self.gal = None
        
        # Analysis type
        self.type = 'Resolved'  # 'Integrated' or 'Resolved'

    def simulate(self, mass_max=12, mass_min=4, sfr_prior_type='SFRflat', 
                 sfr_min=-9, sfr_max=2, ssfr_min=-12.0, ssfr_max=-7.5, 
                 z_prior='flat', z_min=0.0, z_max=10.0, Z_min=-2.27, Z_max=0.4, 
                 dust_model='Calzetti', dust_prior='flat', Av_min=0.0, Av_max=3.0, 
                 tx_alpha=1.0, Nparam=3):
        """
        Simulate a galaxy population using specified priors.

        Parameters
        ----------
        mass_max : float, optional
            Maximum log stellar mass (default: 12)
        mass_min : float, optional
            Minimum log stellar mass (default: 4)
        sfr_prior_type : str, optional
            Type of SFR prior: 'SFRflat', 'sSFRflat', 'sSFRlognormal' (default: 'SFRflat')
        sfr_min : float, optional
            Minimum log star formation rate (default: -9)
        sfr_max : float, optional
            Maximum log star formation rate (default: 2)
        ssfr_min : float, optional
            Minimum log specific star formation rate (default: -12.0)
        ssfr_max : float, optional
            Maximum log specific star formation rate (default: -7.5)
        z_prior : str, optional
            Type of redshift prior: 'flat', 'exp' (default: 'flat')
        z_min : float, optional
            Minimum redshift (default: 0.0)
        z_max : float, optional
            Maximum redshift (default: 10.0)
        Z_min : float, optional
            Minimum metallicity [M/H] (default: -2.27)
        Z_max : float, optional
            Maximum metallicity [M/H] (default: 0.4)
        dust_model : str, optional
            Dust attenuation model: 'Calzetti' (default: 'Calzetti')
        dust_prior : str, optional
            Type of dust prior: 'flat' (default: 'flat')
        Av_min : float, optional
            Minimum dust attenuation A_V (default: 0.0)
        Av_max : float, optional
            Maximum dust attenuation A_V (default: 3.0)
        tx_alpha : float, optional
            Alpha parameter for Dirichlet SFH prior (default: 1.0)
        Nparam : int, optional
            Number of SFH parameters for Dirichlet prior (default: 3)

        Notes
        -----
        This method sets up priors using dense_basis and generates a library of
        simulated galaxies. For parametric SFH (τ-delayed), it uses generate_atlas_parametric.
        For non-parametric SFH, it uses dense_basis.generate_atlas with Dirichlet priors.
        """
        import dense_basis as db
        
        # Set up priors
        priors = db.Priors()
        priors.mass_max = mass_max
        priors.mass_min = mass_min
        priors.sfr_prior_type = sfr_prior_type
        priors.sfr_min = sfr_min
        priors.sfr_max = sfr_max
        priors.ssfr_min = ssfr_min
        priors.ssfr_max = ssfr_max
        priors.z_prior = z_prior
        priors.z_min = z_min
        priors.z_max = z_max
        priors.Z_min = Z_min
        priors.Z_max = Z_max
        priors.dust_model = dust_model
        priors.dust_prior = dust_prior
        priors.Av_min = Av_min
        priors.Av_max = Av_max
        priors.tx_alpha = tx_alpha
        priors.Nparam = Nparam

        # Generate atlas based on SFH type
        if self.parametric:
            print("Generating parametric (τ-delayed) SFH atlas...")
            generate_atlas_parametric(
                priors, N_pregrid=self.n_simulation,
                fname=self.atlas_name, store=True, path=self.atlas_path,
                filter_list=self.filter_list, filt_dir=self.filter_path, 
                norm_method='none'
            )
        else:
            print("Generating non-parametric (Dirichlet) SFH atlas...")
            db.generate_atlas(
                N_pregrid=self.n_simulation, priors=priors,
                fname=self.atlas_name, store=True, path=self.atlas_path,
                filter_list=self.filter_list, filt_dir=self.filter_path, 
                norm_method='none'
            )

    def load_simulation(self):
        """
        Load the simulated galaxy population from saved atlas.

        Returns
        -------
        obs : np.ndarray
            Observed magnitudes (n_simulation, n_filters)
        theta : np.ndarray
            Physical parameters (n_simulation, n_params)

        Notes
        -----
        Updates self.obs, self.theta, and self.labels based on the SFH type.
        For parametric SFH, parameters are [M*, M*_formed, SFR, τ, t_i, [M/H], A_V, z].
        For Dirichlet SFH, parameters are [M*, SFR, t_25%, t_50%, t_75%, [M/H], A_V, z].
        """
        import dense_basis as db
        
        # Determine number of SFH parameters
        nparam = 2 if self.parametric else 3
        
        # Load atlas
        atlas = db.load_atlas(
            self.atlas_name, N_pregrid=self.n_simulation, 
            N_param=nparam, path=self.atlas_path
        )
        
        # Extract SEDs and convert to magnitudes
        zs = atlas['zval']
        atlas_seds = atlas['sed']
        
        # Apply filter removal if specified
        if self.remove_filters is not None:
            atlas_seds = atlas_seds[:, [i for i in range(len(atlas_seds[0,:]))
                                       if i not in self.remove_filters]]
        
        # Convert from microJy to AB magnitudes
        obs = -2.5 * np.log10(atlas_seds * 1e-6 / 3631)

        # Extract parameters based on SFH type
        if self.parametric:
            sfhs = atlas['sfh_tuple']
            theta = np.zeros((self.n_simulation, 8))
            theta[:, 0] = sfhs[:, 0]  # M* (surviving)
            theta[:, 1] = sfhs[:, 1]  # M* (formed)
            theta[:, 2] = sfhs[:, 2]  # SFR
            theta[:, 3] = sfhs[:, 3]  # τ
            theta[:, 4] = sfhs[:, 4]  # t_i
            theta[:, 5] = atlas['met'][:, 0]  # [M/H]
            theta[:, 6] = atlas['dust'][:, 0]  # A_V
            theta[:, 7] = atlas['zval'][:, 0]  # z
            
            self.labels = [
                'log($\\rm{M}_{*}/\\rm{M}_{\\odot}$)',
                'log($\\rm{M}_{*}^{\\rm{formed}}/\\rm{M}_{\\odot}$)',
                'log(SFR/($\\rm{M}_{\\odot}$/yr))',
                '$\\tau$ [Gyr]', '$t_i$ [Gyr]',
                '[M/H]', 'Av', 'z'
            ]
        else:
            # Dirichlet SFH parameters
            theta = np.zeros((self.n_simulation, 8))
            sfhs = atlas['sfh_tuple_rec']
            sfhs = np.reshape(sfhs, (self.n_simulation, 6))
            theta[:, 0] = sfhs[:, 0]  # M* (surviving)
            theta[:, 1] = sfhs[:, 1]  # SFR
            theta[:, 2] = sfhs[:, 3]  # t_25%
            theta[:, 3] = sfhs[:, 4]  # t_50%
            theta[:, 4] = sfhs[:, 5]  # t_75%
            theta[:, 5] = atlas['met'][:, 0]  # [M/H]
            theta[:, 6] = atlas['dust'][:, 0]  # A_V
            theta[:, 7] = atlas['zval'][:, 0]  # z

            # Add formed stellar mass if requested
            if self.both_masses:
                theta = np.concatenate((atlas['mstar'].reshape(-1,1), theta), axis=1)

        self.obs = obs
        self.theta = theta

        return obs, theta
    
    def load_obs_features(self):
        """
        Loads observational features from pre-saved numpy files.

        Parameters:
        None

        Returns:
        None
        """
        # Load observational features from the survey

        #mean of the distribution of noise in the galaxies for each filter and different bins of flux
        self.mean_sigma_obs = np.load(self.filter_path+'mean_sigma_jades_res_bins.npy') 
        #std of the distribution of noise in the galaxies for each filter and different bins of flux
        self.stds_sigma_obs = np.load(self.filter_path+'std_sigma_jades_res_bins.npy') 
        #different bins of flux for each filter
        self.percentiles = np.load(self.filter_path+'percentiles_jades_res_bins.npy')
        #1 sigma depth limits for each filter
        self.limits=np.load(self.filter_path+'background_noise_hainline.npy') 
        
        if self.remove_filters is not None:
                self.mean_sigma_obs = self.mean_sigma_obs[:, [i for i in range(len(self.mean_sigma_obs[0,:])) if i not in self.remove_filters]]
                self.stds_sigma_obs = self.stds_sigma_obs[:, [i for i in range(len(self.stds_sigma_obs[0,:])) if i not in self.remove_filters]]
                self.percentiles = self.percentiles[:, [i for i in range(len(self.percentiles[0,:])) if i not in self.remove_filters]]
                self.limits =   self.limits[[i for i in range(len(self.limits)) if i not in self.remove_filters]]

        print('Observational features loaded')        


    def add_noise_nan_limit_all_old(self):
        """
        Add realistic observational effects to all simulated galaxies.
        
        This includes:
        - Photometric uncertainties based on magnitude
        - Detection limits
        - Non-detections
        
        Updates self.mag with processed photometry ready for training.
        """
        self.mag = np.zeros((self.n_simulation, len(self.obs[0,:]), 2))

        for j in trange(self.n_simulation, desc="Adding observational realism"):
            for i in range(len(self.obs[0,:])):
                self.mag[j, i, :] = self._add_noise_nan_limit(self.obs[j][i], i)

    def _add_noise_nan_limit_old(self, mag, filter_idx):
        """
        Add noise and handle detection limits for a single magnitude measurement.

        Parameters
        ----------
        mag : float
            Input magnitude
        filter_idx : int
            Index of the filter

        Returns
        -------
        list
            [noisy_magnitude, uncertainty]

        ## Notes
        Error handling include_limits=False
        """
        
        # Magnitude-dependent uncertainty bins
        i_not_last_bin = [10,12,13,14,15,16,17,18]
        percentiles = self.percentiles
        flux = mag_conversion(mag, convert_to='flux')

        # Determine uncertainty based on magnitude and apply detection limit
        if self.include_limits and flux > self.limits[filter_idx]:
            # Assign uncertainty based on magnitude bin
            if mag < percentiles[0, filter_idx]:
                mag_err = np.random.normal(
                    self.mean_sigma_obs[filter_idx, 0], 
                    self.stds_sigma_obs[filter_idx, 0]
                )
            elif mag < percentiles[1, filter_idx]:
                mag_err = np.random.normal(
                    self.mean_sigma_obs[filter_idx, 1], 
                    self.stds_sigma_obs[filter_idx, 1]
                )
            elif mag < percentiles[2, filter_idx]:
                mag_err = np.random.normal(
                    self.mean_sigma_obs[filter_idx, 2], 
                    self.stds_sigma_obs[filter_idx, 2]
                )
            else:
                bin_idx = 2 if filter_idx in i_not_last_bin else 3
                mag_err = np.random.normal(
                    self.mean_sigma_obs[filter_idx, bin_idx], 
                    self.stds_sigma_obs[filter_idx, bin_idx]
                )

            # Add noise
            noise = np.random.normal(0.0, np.abs(mag_err))
            mag_n_noise = mag + noise

        else:
            # Non-detection
            mag_n_noise = 0.0
            mag_err = mag_conversion(self.limits[filter_idx], convert_to='mag')

        if self.include_sigma:
            return [mag_n_noise, mag_err]
        else:
            return [mag, mag_err]
        
    
    def add_noise_nan_limit_all(self):
        """
        Add realistic observational effects to all simulated galaxies (Vectorized).
        
        This includes:
        - Photometric uncertainties based on magnitude
        - Detection limits
        - Non-detections (set to 99.0)
        
        Updates self.mag with processed photometry ready for training.
        """
        n_sims, n_filts = self.obs.shape
        self.mag = np.zeros((n_sims, n_filts, 2))
        
        # We loop over FILTERS instead of GALAXIES. 
        # This allows us to process all simulations instantly using numpy arrays.
        for i in trange(n_filts, desc=f"Adding observational realism to {n_filts} filters"):
            
            # Extract the entire column of magnitudes for this filter
            mags_filter = self.obs[:, i]
            
            # Process all galaxies for this filter at once
            mag_n_noise, mag_err = self._add_noise_nan_limit(mags_filter, i)
            
            # Store results
            self.mag[:, i, 0] = mag_n_noise
            self.mag[:, i, 1] = mag_err

    def _add_noise_nan_limit(self, mag_array, filter_idx):
        """
        Vectorized noise addition and detection limits for an array of magnitudes.

        Parameters
        ----------
        mag_array : np.ndarray
            Array of input magnitudes for a single filter (all galaxies).
        filter_idx : int
            Index of the filter.

        Returns
        -------
        tuple
            (noisy_magnitudes_array, uncertainties_array)
        """
        
        
        flux_array = mag_conversion(mag_array, convert_to='flux')

        # 1. Determine limits
        limit_flux = self.limits[filter_idx] #in microjy
        limit_mag_err = mag_conversion(limit_flux, convert_to='mag')

        # 2. Setup defaults for non-detections (Using 99.0)
        final_mags = np.full_like(mag_array, 99.0)
        final_errs = np.full_like(mag_array, limit_mag_err)

        # 3. Create mask for detected pixels
        if self.include_limit:
            detected_mask = flux_array > limit_flux
        else:
            detected_mask = np.ones_like(flux_array, dtype=bool)

        mags_det = mag_array[detected_mask]

        if mags_det.size > 0:
            # 4. Find magnitude bins dynamically (replaces the slow if/elif chain)
            percentiles_f = self.percentiles[:, filter_idx]
            bin_indices = np.digitize(mags_det, percentiles_f)
            
            # Clip to max bin index to avoid IndexError for very faint/bright sources
            max_bin = self.mean_sigma_obs.shape[1] - 1
            bin_indices = np.clip(bin_indices, 0, max_bin)

            # 5. Extract statistics for the corresponding bins
            pixel_means = self.mean_sigma_obs[filter_idx, bin_indices]
            pixel_stds = self.stds_sigma_obs[filter_idx, bin_indices]

            # 6. Correct Truncnorm Math (Truncate at 0)
            a_std = (0.0 - pixel_means) / pixel_stds
            
            mag_errs_det = truncnorm.rvs(
                a=a_std, 
                b=np.inf, 
                loc=pixel_means, 
                scale=pixel_stds
            )

            # 7. Add noise
            noise = np.random.normal(0.0, mag_errs_det)
            
            if self.include_sigma:
                final_mags[detected_mask] = mags_det + noise
            else:
                final_mags[detected_mask] = mags_det
                
            final_errs[detected_mask] = mag_errs_det

        return final_mags, final_errs    

    

    def train(self, min_thetas=[6, -10, 0, 0, 0, -2.3, 0, 0], 
              max_thetas=[12, 3, 1, 1, 1, 0.4, 3, 10], 
              n_max=1000000, epochs_max=None, nblocks=15, nhidden=500, 
              val_fraction=0.1, device='cpu'):
        """
        Train the neural density estimator using simulation-based inference.

        Parameters
        ----------
        min_thetas : list, optional
            Lower bounds for posterior parameters
        max_thetas : list, optional  
            Upper bounds for posterior parameters
        n_max : int, optional
            Maximum number of training samples (default: 1000000)
        epochs_max : int, optional
            Maximum training epochs (default: None, uses early stopping)
        nblocks : int, optional
            Number of coupling blocks in normalizing flow (default: 15)
        nhidden : int, optional
            Number of hidden features per block (default: 500)
        val_fraction : float, optional
            Fraction of data for validation (default: 0.1)
        device : str, optional
            Device for training: 'cpu' or 'cuda' (default: 'cpu')

        Notes
        -----
        Trains a Masked Autoregressive Flow (MAF) using Neural Posterior Estimation.
        Saves the trained model to self.model_path + self.model_name.
        """
        # Prepare observations based on configuration
        if self.condition_sigma:
            obs = np.reshape(self.mag, (self.n_simulation, 2 * len(self.obs[0])))
        elif self.include_mask or self.include_limit or self.include_sigma:
            obs = self.mag[:, :, 0]
        else:
            print('No noise, mask or limit included')
            obs = self.obs if self.mag is None else self.mag[:, :, 0]

        # Initialize neural network
        maf_model = sbi.neural_nets.posterior_nn(
            'maf', hidden_features=nhidden, num_transforms=nblocks, num_layers=2
        )

        if self.infer_z:
            # Define parameter bounds
            lower_bounds = torch.tensor(min_thetas, dtype=torch.float32)
            upper_bounds = torch.tensor(max_thetas, dtype=torch.float32)
            bounds = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)
            
            print('Lower bounds:', lower_bounds)
            print('Upper bounds:', upper_bounds)

            # Initialize NPE
            anpe = Inference.SNPE(prior=bounds, density_estimator=maf_model, device=device)

            # Add training data
            anpe.append_simulations(
                torch.as_tensor(self.theta[:n_max, :].astype(np.float32)).to(device),
                torch.as_tensor(obs[:n_max, :].astype(np.float32)).to(device)
            )
        else:
            # Training without redshift inference
            lower_bounds = torch.tensor(min_thetas[:-1], dtype=torch.float32)
            upper_bounds = torch.tensor(max_thetas[:-1], dtype=torch.float32)
            bounds = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)
            
            anpe = Inference.SNPE(prior=bounds, density_estimator=maf_model, device=device)
            
            # Concatenate redshift as input
            obs_with_z = np.concatenate([
                obs[:n_max, :], 
                np.reshape(self.theta[:n_max, -1], (n_max, 1))
            ], axis=1)
            
            anpe.append_simulations(
                torch.as_tensor(self.theta[:n_max, :-1].astype(np.float32)).to(device),
                torch.as_tensor(obs_with_z.astype(np.float32)).to(device)
            )

        # Train
        train_kwargs = {
            'show_train_summary': True, 
            'retrain_from_scratch': True,
            'validation_fraction': val_fraction
        }
        if epochs_max is not None:
            train_kwargs['max_num_epochs'] = epochs_max

        p_theta_x_est = anpe.train(**train_kwargs)

        # Build posterior and save
        qphi = anpe.build_posterior(p_theta_x_est)
        
        model_file = self.model_path + self.model_name
        anpe_file = self.model_path + 'anpe_' + self.model_name

        with open(model_file, "wb") as f:
            pickle.dump(qphi, f)
        with open(anpe_file, "wb") as f:
            pickle.dump(anpe, f)

        print(f"Model saved to {model_file}")

    def test_performance(self, n_test=1000, n_samples=100, return_posterior=False, device='cpu'):
        """
        Test model performance on held-out simulations.

        Parameters
        ----------
        n_test : int, optional
            Number of test samples (default: 1000)
        n_samples : int, optional
            Number of posterior samples per test case (default: 100)
        return_posterior : bool, optional
            Whether to return full posterior samples (default: False)
        device : str, optional
            Device for inference (default: 'cpu')

        Returns
        -------
        posteriors : np.ndarray, optional
            Full posterior samples if return_posterior=True

        Notes
        -----
        Updates self.means_test, self.stds_test with test results.
        """
        # Load trained model
        with open(self.model_path + self.model_name, 'rb') as f:
            qphi = pickle.load(f)

        means, stds, posteriors, modes = [], [], [], []

        # Prepare test observations
        if self.condition_sigma:
            obs = np.reshape(self.mag, (self.n_simulation, 2 * len(self.obs[0])))
        elif self.include_mask or self.include_limit or self.include_sigma:
            obs = self.mag[:, :, 0]
        else:
            obs = self.obs

        if not self.infer_z:
            obs = np.concatenate([obs, np.reshape(self.theta[:, -1], (len(obs), 1))], axis=1)

        print(obs)

        # Run inference on test set
        for j in trange(n_test, desc="Testing performance"):
            posterior_samples = np.array(
                qphi.sample(
                    (n_samples,), 
                    x=torch.as_tensor(obs[j].astype(np.float32)).to(device),
                    show_progress_bars=False
                ).detach().to('cpu')
            )
            
            posteriors.append(posterior_samples)
            stds.append(np.std(posterior_samples, axis=0))
            means.append(np.median(posterior_samples, axis=0))
            modes.append(stats.mode(np.round(posterior_samples, 1), axis=0))

        self.means_test = np.array(means)
        self.stds_test = np.array(stds)
        self.mode_test = np.array(modes)

        if return_posterior:
            return np.array(posteriors)
        

    def _get_posterior_obs(self, obs, qphi, n_samples=1000, bar=True, input_z=None,device='cpu'):
        """
        Generate posterior samples for observed data.

        Parameters:
        - obs: numpy array
            Observed data.
        - qphi: object
            Trained model for sampling.
        - n_samples: int, optional (default=1000)
            Number of samples to generate.
        - bar: bool, optional (default=True)
            Whether to show a progress bar.
        - input_z: numpy array, optional
            Input redshift values.

        Returns:
        - numpy array
            Posterior samples.
        """
        if not self.infer_z and input_z is not None:
            input_z = np.repeat(input_z, len(obs), axis=0)
            obs = np.concatenate([obs, np.reshape(input_z, newshape=(len(obs), 1))], axis=1)
        
        posteriors = []
        if bar:
            for i in trange(len(obs)):
                p = np.array(qphi.sample((n_samples,), x=torch.as_tensor(np.array([obs[i, :]]).astype(np.float32)).to(device), show_progress_bars=False).detach().to('cpu'))
                posteriors.append(p)
        else:
            for i in range(len(obs)):
                p = np.array(qphi.sample((n_samples,), x=torch.as_tensor(np.array([obs[i, :]]).astype(np.float32)).to(device), show_progress_bars=True).detach().to('cpu'))
                posteriors.append(p)

        return np.array(posteriors)        
        

    def get_posteriors_resolved(self, phot_arr, n_gal, n_samples=50, save=True, return_stats=True,sigma_arr=None, bar=True, input_z=None, device='cpu'):
        """
        Generate posterior samples for resolved galaxy photometry.

        Parameters
        ----------
        phot_arr : np.ndarray
            Photometric data array (n_pixels, n_filters)
        n_gal : int
            Galaxy identifier for saving
        n_samples : int, optional
            Number of posterior samples (default: 50)
        save : bool, optional
            Whether to save results (default: True)
        return_stats : bool, optional
            Whether to return summary statistics (default: True)
        sigma_arr : np.ndarray, optional
            Photometric uncertainties
        bar : bool, optional
            Whether to show progress bar (default: True)
        input_z : float, optional
            Input redshift if not inferring
        device : str, optional
            Device for inference (default: 'cpu')

        Returns
        -------
        Various arrays depending on options selected
            Posterior samples, summary statistics, and coordinate information
        """
        # Apply detection limits and convert to magnitudes
        if self.include_limit:
            for i in range(len(phot_arr[0, :])):
                # 1. Create mask BEFORE overwriting the array
                is_non_detect = phot_arr[:, i] < self.limits[i]
                
                # 2. Update photometry using the mask
                phot_arr[:, i] = np.where(
                    is_non_detect, 
                    99.0, # previously 0, but 99 is more standard for non-detections 
                    mag_conversion(phot_arr[:, i])
                )
                
                # 3. Update errors using the SAME mask
                if self.condition_sigma and sigma_arr is not None:
                    sigma_arr[:, i] = np.where(
                        is_non_detect, 
                        self.limits[i], 
                        np.abs(sigma_arr[:, i])
                    )
            mag_arr = phot_arr
        else:
            mag_arr = mag_conversion(phot_arr)
            coords_ok = np.where(~np.isnan(np.sum(mag_arr, axis=1)))[0]
            mag_arr = mag_arr[coords_ok, :]

        # Prepare full array with uncertainties if needed
        if self.condition_sigma:
            full_arr = np.zeros((len(phot_arr[:, 0]), len(phot_arr[0,:]), 2))
            full_arr[:, :, 0] = mag_arr
            
            for i in range(len(full_arr[:, 0, 0])):
                for j in range(len(full_arr[0, :, 0])):
                    if sigma_arr[i, j] == self.limits[j]:
                        full_arr[i, j, 1] = mag_conversion(self.limits[j], convert_to='mag')
                    else:
                        full_arr[i, j, 1] = (sigma_arr[i, j] * np.abs(-2.5 / (np.log(10) * phot_arr[i, j])))
            
            mag_arr = np.reshape(full_arr, (len(phot_arr[:, 0]), len(phot_arr[0, :]) * 2))
        
        self.gal = mag_arr

        # Load appropriate model
        if not self.infer_z_integrated:
            model_file = self.model_path + self.model_name
        else:
            model_file = self.model_path + 'integrated_z' + self.model_name
        
        with open(model_file, 'rb') as f:
            qphi = pickle.load(f)

        # Generate posteriors
        posteriors_full = self._get_posterior_obs(
            self.gal, qphi, n_samples=n_samples, bar=bar, 
            input_z=input_z, device=device
        )

        # Compute summary statistics if requested
        if return_stats:
            means = np.median(posteriors_full, axis=1)
            stds = np.std(posteriors_full, axis=1)
            modes = []
            for p in posteriors_full:
                mode = []
                for k in range(len(p[0, :])):
                    mode.append(stats.mode(np.round(p[:, k], 1)))
                modes.append(mode)

        # Save results if requested
        if save:
            np.save(f'post_gal_{n_gal}.npy', posteriors_full)
            if not self.include_limit:
                np.save(f'coords_gal_{n_gal}.npy', coords_ok)
            if return_stats:
                np.save(f'means_gal_{n_gal}.npy', means)
                np.save(f'stds_gal_{n_gal}.npy', stds)
                np.save(f'modes_gal_{n_gal}.npy', modes)

        # Return appropriate results
        if return_stats and not self.include_limit:
            return posteriors_full, means, stds, modes, coords_ok
        elif return_stats and self.include_limit:
            return posteriors_full, means, stds, modes
        elif self.include_limit and not return_stats:
            return posteriors_full
        else:
            return posteriors_full, coords_ok