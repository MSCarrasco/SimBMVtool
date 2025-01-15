import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from gammapy.data import Observations, PointingMode, ObservationMetaData
import math
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord,EarthLocation,AltAz, angular_separation, position_angle,Angle
from astropy.visualization import quantity_support
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm,SymLogNorm

import yaml, shutil, glob, logging
from copy import deepcopy
import pandas as pd
from pathlib import Path
import os, pathlib
import pickle as pk
import seaborn as sns
from itertools import product

# %matplotlib inline

from IPython.display import display
from gammapy.data import DataStore, FixedPointingInfo, Observation, observatory_locations, PointingMode
from gammapy.datasets import MapDataset,MapDatasetEventSampler,Datasets,MapDatasetOnOff
from gammapy.irf import load_irf_dict_from_file, Background2D, Background3D, FoVAlignment
from gammapy.makers import FoVBackgroundMaker,MapDatasetMaker, SafeMaskMaker, RingBackgroundMaker
from gammapy.maps import MapAxis, WcsGeom, WcsNDMap, Map
from regions import CircleAnnulusSkyRegion, CircleSkyRegion, EllipseSkyRegion
from gammapy.maps.region.geom import RegionGeom
from gammapy.estimators import ExcessMapEstimator
from gammapy.datasets import Datasets, MapDataset, MapDatasetOnOff
from astropy.visualization.wcsaxes import SphericalCircle

from gammapy.modeling import Fit,Parameter
from gammapy.modeling.models import (
    FoVBackgroundModel,
    GaussianSpatialModel,
    Models,
    SpatialModel,
    PowerLawSpectralModel,
    PowerLawNormSpectralModel,
    LogParabolaSpectralModel,
    SkyModel,
)
from gammapy.catalog import SourceCatalogGammaCat, SourceCatalogObject

from scipy.stats import norm as norm_stats
from gammapy.stats import CashCountsStatistic
from gammapy.modeling import Parameter, Parameters
from gammapy.utils.compat import COPY_IF_NEEDED

from itertools import product
from baccmod import RadialAcceptanceMapCreator, Grid3DAcceptanceMapCreator, BackgroundCollectionZenith
from baccmod.toolbox import (get_unique_wobble_pointings)

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union
from gammapy.datasets import MapDatasetEventSampler
from gammapy.irf import Background3D, BackgroundIRF
from gammapy.makers.utils import make_map_background_irf

from toolbox import (get_value_at_threshold,
                     compute_residuals,
                     scale_value,
                     get_geom,
                     get_exclusion_mask_from_dataset_geom,
                     get_skymaps_dict)

from base_simbmvtool_creator import BaseSimBMVtoolCreator

import logging
logger = logging.getLogger(__name__)

class BMVCreator(BaseSimBMVtoolCreator):
    def init_save_paths(self) -> None:
        region_shape = self.region_shape if self.region_shape != "n_circles" else f"{self.n_circles}_circle{'s'*(self.n_circles > 1)}"
        self.end_name=f'{self.bkg_dim}D_{region_shape}_Ebins_{self.nbin_E_acc}_offsetbins_{self.nbin_offset_acc}_offset_max_{self.size_fov_acc.value:.1f}'+f'_{self.cos_zenith_binning_parameter_value}sperW'*self.zenith_binning+f'_exclurad_{self.exclu_rad}'*((self.exclu_rad != 0) & (self.region_shape != 'noexclusion'))
        self.index_suffix = f"_with_bkg_{self.bkg_dim}d_{self.method}_{self.end_name[3:]}"
        self.acceptance_files_dir = f"{self.output_dir}/{self.subdir}/{self.method}/acceptances"
        self.plots_dir=f"{self.output_dir}/{self.subdir}/{self.method}/plots"
        
        Path(self.acceptance_files_dir).mkdir(parents=True, exist_ok=True)
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
        print("acceptance files: ",self.acceptance_files_dir)

    def init_exclusion_region(self, cfg_exclusion_region='stored_config', init_save_paths=False) -> None:
        '''Initialize the exclusion regions applied for the modeling and/or the skymaps
        The parameters need to be stored directly, or in self.config["source"]["exclusion_region"], or to be given in a dictionary'''
        
        if isinstance(cfg_exclusion_region, dict):
            self.cfg_exclusion_region = cfg_exclusion_region
        elif cfg_exclusion_region == 'stored_config':
            self.cfg_source = self.config["source"]
            self.cfg_exclusion_region = self.cfg_source["exclusion_region"]
        
        if not cfg_exclusion_region == 'stored_parameters': self.region_shape = self.cfg_exclusion_region["shape"]
        self.source_region = []
        
        if (self.region_shape=='noexclusion'):
            # Parts of the code don't work without an exlcusion region, so a dummy one is declared at the origin of the ICRS frame (and CircleSkyRegion object needs a non-zero value for the radius) 
            self.source_region.append(CircleSkyRegion(center=SkyCoord(ra=0. * u.deg, dec=0. * u.deg, frame='icrs'),radius=1*u.deg))
        
        elif (self.region_shape=='n_circles'):
            if not cfg_exclusion_region == 'stored_parameters':
                self.n_circles = len(self.cfg_exclusion_region["n_circles"])
                self.n_circles_radec = []
                self.n_circles_radius = []
            else:
                self.n_circles = len(self.n_circles_radius)
            
            for i in range(self.n_circles):
                if not cfg_exclusion_region == 'stored_parameters':
                    cfg_circle = self.cfg_exclusion_region["n_circles"][f"circle_{i+1}"]
                    self.n_circles_radec.append([cfg_circle["ra"],cfg_circle["dec"]])
                    self.n_circles_radius.append(cfg_circle["radius"])

                circle_pos = SkyCoord(ra=self.n_circles_radec[i][0] * u.deg, dec=self.n_circles_radec[i][1] * u.deg, frame='icrs')
                circle_rad = self.n_circles_radius[i] * u.deg
                self.source_region.append(CircleSkyRegion(center=circle_pos, radius=circle_rad))
                if i == 0:
                    self.exclusion_radius=circle_rad
                    self.exclu_rad = circle_rad.to_value(u.deg)
        
        elif (self.region_shape=='ellipse'):
            if not cfg_exclusion_region == 'stored_parameters': 
                self.width = self.cfg_exclusion_region["ellipse"]["width"] * u.deg
                self.height = self.cfg_exclusion_region["ellipse"]["height"] * u.deg
                self.angle = self.cfg_exclusion_region["ellipse"]["angle"] * u.deg
            self.source_region = [EllipseSkyRegion(center=self.source_pos, width=self.width, height=self.height, angle=self.angle)]
        
        self.exclude_regions=[]
        for region in self.source_region: self.exclude_regions.append(region)
        
        self.single_region=True     # Set it to False to mask additional regions in the FoV. The example here adds a region on Zeta Tauri position. So far you need to declare it manually in the init function
        if not self.single_region:
            # TO-DO: catalog of classic regions to mask. Here it is an example for masking zeta tauri on crab observations
            zeta_region = CircleSkyRegion(center=SkyCoord(ra=84.4125*u.deg, dec=21.1425*u.deg), radius=self.exclusion_radius)
            self.exclude_regions.append(zeta_region)
        
        self.source_info = [self.source_name, self.source_pos, self.source_region]
        if init_save_paths: self.init_save_paths()
    
    def init_background_maker(self, cfg_background_maker='stored_config', init_save_paths=False) -> None:
        '''Initialize the background maker parameters applied for the skymaps
        The parameters need to be stored in self.config["background"]["maker"] or to be given in a dictionary'''
        
        if isinstance(cfg_background_maker, dict):
            self.cfg_background_maker = cfg_background_maker
        else:
            self.cfg_background = self.config["background"]
            self.cfg_background_maker = self.cfg_background["maker"]
        
        self.correlation_radius = self.cfg_background_maker["correlation_radius"]
        self.correlate_off = self.cfg_background_maker["correlate_off"]
        self.ring_bkg_param = [self.cfg_background_maker["ring"]["internal_ring_radius"],self.cfg_background["maker"]["ring"]["width"]]
        self.fov_bkg_param = self.cfg_background_maker["fov"]["method"]
        
        if init_save_paths: self.init_save_paths()    

    def init_background_modeling(self, cfg_acceptance='stored_config', init_save_paths=False) -> None:
        '''Initialize the background maker parameters applied for the skymaps
        The parameters need to be stored in self.config["acceptance"] or to be given in a dictionary'''
        
        if isinstance(cfg_acceptance, dict):
            self.cfg_acceptance = cfg_acceptance
        elif cfg_acceptance == 'stored_config':
            self.cfg_acceptance = self.config["acceptance"]
        
        self.method = self.cfg_acceptance["method"]

        self.out_collection = self.cfg_acceptance["out_collection"]
        if 'single_file_path' in self.cfg_acceptance.keys(): self.single_file_path = self.cfg_acceptance["single_file_path"]

        # BAccMod parameters also used for zenith binning SimBMVtool methods
        self.cos_zenith_binning = self.cfg_acceptance["cos_zenith_binning"]
        self.zenith_binning=self.cos_zenith_binning["zenith_binning"]
        self.initial_cos_zenith_binning=self.cos_zenith_binning['initial_cos_zenith_binning']
        self.cos_zenith_binning_method=self.cos_zenith_binning['cos_zenith_binning_method']
        self.cos_zenith_binning_parameter_value=self.cos_zenith_binning['cos_zenith_binning_parameter_value']

        if self.cfg_acceptance["tool"] == 'baccmod':
            self.runwise_normalisation=self.cos_zenith_binning["runwise_normalisation"]
            self.fit_fnc=self.cfg_acceptance["fit"]["fnc"]
            self.fit_bounds=self.cfg_acceptance["fit"]["bounds"]
        
        if init_save_paths: self.init_save_paths()

    def init_config_bmv(self, config_path) -> None:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        # Print the loaded configuration
        for key in config.keys(): print(f"{key}: {config[key]}")
        self.config = config
        self.config_path = config_path
        self.cfg_paths = config["paths"]
        self.cfg_data = self.config["data"]

        # Paths
        self.path_data=self.cfg_paths["path_data"] # path to files to get run information used for simulation
        self.output_dir = self.cfg_paths["output_dir"]
        self.subdir = self.cfg_paths["subdir"]
        if self.simulator or ((not self.real_data) & (not self.external_data)): self.simulated_obs_dir = f"{self.output_dir}/{self.subdir}/simulated_data"

        # Data
        self.run_list = np.array(self.cfg_data["run_list"])
        self.all_obs_ids = np.array([])

        # Source        
        self.init_exclusion_region()

        # Background maker
        self.init_background_maker()

        # BAccMod parameters
        self.init_background_modeling()

        # Output files
        self.init_save_paths()

    def __init__(self, config_path:str, external_data=False, real_data=False, external_model=False):
        super().__init__(external_data=external_data, real_data=real_data, external_model=external_model)
        if config_path is not None: 
            self.init_config(config_path)
            self.init_config_bmv(config_path)
        else: ValueError("Config file needed")
    
    def load_output_background_irfs(self, config_path=None) -> None:
        if config_path is not None: self.init_config(config_path)

        if self.out_collection:
            self.bkg_output_irf_collection = {}

            if (self.all_obs_ids.size == 0) and not self.external_data:
                paths = list(Path(self.acceptance_files_dir).rglob("acceptance_*.fits"))
                self.all_obs_ids = np.arange(len(paths))+1
            self.obs_ids = self.all_obs_ids if self.run_list.shape[0] == 0 else self.run_list
            if self.real_data:
                for iobs,obs_id in enumerate(self.obs_ids): self.bkg_output_irf_collection[iobs] = Background3D.read(f'{self.acceptance_files_dir}/acceptance_obs-{obs_id}.fits')
            else:

                for obs_id in self.obs_ids: self.bkg_output_irf_collection[int(obs_id)] = Background3D.read(f'{self.acceptance_files_dir}/acceptance_obs-{obs_id}.fits')
        else:
            self.bkg_output_irf = Background3D.read(self.single_file_path)
    
    def get_baccmod_zenith_binning(self,
                                       observations: Observations
                                       ) -> None:
        """
        Calculate the cos zenith binning with the baccmod method

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        background : BackgroundCollectionZenith
            The collection of background model with the zenith associated to each model

        """

        # Determine binning method. Convention : per_wobble methods have negative values
        methods = {'min_livetime': 1, 'min_livetime_per_wobble': -1, 'min_n_observation': 2,
                   'min_n_observation_per_wobble': -2}
        try:
            i_method = methods[self.cos_zenith_binning_method]
        except KeyError:
            logger.error(f" KeyError : {self.cos_zenith_binning_method} not a valid zenith binning method.\nValid "
                         f"methods are {[*methods]}")
            raise
        per_wobble = i_method < 0

        # Initial bins edges
        cos_zenith_bin = np.sort(np.arange(1.0, 0. - self.initial_cos_zenith_binning, -self.initial_cos_zenith_binning))
        zenith_bin = np.rad2deg(np.arccos(cos_zenith_bin)) * u.deg

        # Cut observations if requested
        compute_observations = observations

        # Determine initial bins values
        cos_zenith_observations = np.array(
            [np.cos(obs.get_pointing_altaz(obs.tmid).zen) for obs in compute_observations])
        livetime_observations = np.array(
            [obs.observation_live_time_duration.to_value(u.s) for obs in compute_observations])

        # Select the quantity used to count observations
        if i_method in [-1, 1]:
            cut_variable_weights = livetime_observations
        elif i_method in [-2, 2]:
            cut_variable_weights = np.ones(len(cos_zenith_observations), dtype=int)

        # Gather runs per separation angle or all together. Define the minimum multiplicity (-1) to create a zenith bin.
        if per_wobble:
            wobble_observations = np.array(
                get_unique_wobble_pointings(compute_observations))
            multiplicity_wob = 1
        else:
            wobble_observations = np.full(len(cos_zenith_observations), 'any', dtype=np.object_)
            multiplicity_wob = 0

        cumsum_variable = {}
        for wobble in np.unique(wobble_observations):
            # Create an array of cumulative weight of the selected variable vs cos(zenith)
            cumsum_variable[wobble] = np.cumsum(np.histogram(cos_zenith_observations[wobble_observations == wobble],
                                                             bins=cos_zenith_bin,
                                                             weights=cut_variable_weights[
                                                                 wobble_observations == wobble])[0])
        # Initiate the list of index of selected zenith bin edges
        zenith_selected = [0]

        i = 0
        n = len(cos_zenith_bin) - 2

        while i < n:
            # For each wobble, find the index of the first zenith which fulfills the zd binning criteria if any
            # Then concatenate and sort the results for all wobbles
            candidate_i_per_wobble = [np.nonzero(cum_cut_variable >= self.cos_zenith_binning_parameter_value)[0][:1]
                                      for cum_cut_variable in cumsum_variable.values()]  # nonzero is assumed sorted
            candidate_i = np.sort(np.concatenate(candidate_i_per_wobble))

            if len(candidate_i) > multiplicity_wob:
                # If the criteria is fulfilled save the correct index.
                # The first and only candidate_i in the non-per_wobble case and the second in the per_wobble case.
                i = candidate_i[multiplicity_wob]
                zenith_selected.append(i + 1)
                for wobble in np.unique(wobble_observations):
                    # Reduce the cumulative sum by the value at the selected index for the next iteration
                    cumsum_variable[wobble] -= cumsum_variable[wobble][i]
            else:
                # The zenith bin creation criteria is not fulfilled, the last bin edge is set to the end of the
                # cos(zenith) array
                if i == 0:
                    zenith_selected.append(n + 1)
                else:
                    zenith_selected[-1] = n + 1
                i = n
        cos_zenith_bin = cos_zenith_bin[zenith_selected]

        # Associate each observation to the correct bin
        binned_observations = []
        for i in range((len(cos_zenith_bin) - 1)):
            binned_observations.append(Observations())
        for obs in compute_observations:
            binned_observations[np.digitize(np.cos(obs.get_pointing_altaz(obs.tmid).zen), cos_zenith_bin) - 1].append(
                obs)

        # Determine the center of the bin (weighted as function of the livetime of each observation)
        bin_center = []
        for i in range(len(binned_observations)):
            weighted_cos_zenith_bin_per_obs = []
            livetime_per_obs = []
            for obs in binned_observations[i]:
                weighted_cos_zenith_bin_per_obs.append(
                    obs.observation_live_time_duration * np.cos(obs.get_pointing_altaz(obs.tmid).zen))
                livetime_per_obs.append(obs.observation_live_time_duration)
            bin_center.append(np.sum([wcos.value for wcos in weighted_cos_zenith_bin_per_obs]) / np.sum(
                [livet.value for livet in livetime_per_obs]))

        self.cos_zenith_bin_edges = np.array(cos_zenith_bin)
        self.cos_zenith_bin_centers = np.array(bin_center)
    
    def create_zenith_binned_collections(self, collections=["output", "observation"], zenith_bins=4) -> None:
        '''Initialise zenith binned collections: observations and models
        collections = ["true", "output", "observation"]: List of the collections you want binned with
        zenith_bins = int, np.array, 'stored', 'baccmod'
        Observations need to be loaded previously'''
        
        self.cos_zenith_observations = np.array(
                    [np.cos(obs.get_pointing_altaz(obs.tmid).zen) for obs in self.obs_collection])
        
        if isinstance(zenith_bins, int):
            cos_min, cos_max = self.cos_zenith_observations.min(), self.cos_zenith_observations.max()
            self.cos_zenith_bin_edges = np.flip(np.linspace(cos_min, cos_max, zenith_bins+1))
            self.cos_zenith_bin_centers = self.cos_zenith_bin_edges[:-1] + 0.5*(self.cos_zenith_bin_edges[1:]-self.cos_zenith_bin_edges[:-1])
        elif not isinstance(zenith_bins, str) and hasattr(zenith_bins, "__len__"):
            self.cos_zenith_bin_edges = zenith_bins
            self.cos_zenith_bin_centers = self.cos_zenith_bin_edges[:-1] + 0.5*(self.cos_zenith_bin_edges[1:]-self.cos_zenith_bin_edges[:-1])
        elif (zenith_bins == 'baccmod'):
            self.get_baccmod_zenith_binning(self.obs_collection)
        elif (zenith_bins != 'stored'):
            ValueError("Zenith binning not in options")

        if self.cos_zenith_bin_edges[0] < self.cos_zenith_bin_edges[1]: self.cos_zenith_bin_edges = np.flip(self.cos_zenith_bin_edges)
        if (self.cos_zenith_bin_centers.shape[0] > 1) and (self.cos_zenith_bin_centers[0] < self.cos_zenith_bin_centers[1]): self.cos_zenith_bin_centers = np.flip(self.cos_zenith_bin_centers)

        i_collection_array = np.arange(0,len(self.obs_ids))
        
        self.obs_in_coszd_bin = []
        values_true = []
        values_output = []
        values_obs = []

        for cos_max,cos_min,(icos,cos_center) in zip(self.cos_zenith_bin_edges[:-1],self.cos_zenith_bin_edges[1:],enumerate(self.cos_zenith_bin_centers)):
            is_in_coszd_bin = np.where((self.cos_zenith_observations >= cos_min) & (self.cos_zenith_observations < cos_max))
            self.obs_in_coszd_bin.append(self.obs_ids[is_in_coszd_bin])
            iobs_in_coszd_bin = i_collection_array[np.where((self.cos_zenith_observations >= cos_min) & (self.cos_zenith_observations < cos_max))]
            collection_true = []
            collection_output = []
            collection_obs = Observations()
            for iobs in iobs_in_coszd_bin:
                if "true" in collections: 
                    collection_true.append(self.bkg_true_down_irf_collection[iobs])
                    if iobs == iobs_in_coszd_bin[-1]: 
                        values_true.append(collection_true)
                        if cos_center == self.cos_zenith_bin_centers[-1]: self.zenith_binned_bkg_true_down_irf_collection = dict(zip(self.cos_zenith_bin_centers,values_true))
                if "output" in collections: 
                    collection_output.append(self.bkg_output_irf_collection[iobs])
                    if iobs == iobs_in_coszd_bin[-1]:
                        values_output.append(collection_output) 
                        if cos_center == self.cos_zenith_bin_centers[-1]: self.zenith_binned_bkg_output_irf_collection = dict(zip(self.cos_zenith_bin_centers,values_output))
                if "observation" in collections:
                    collection_obs.append(self.obs_collection[int(iobs)])
                    if iobs == iobs_in_coszd_bin[-1]: 
                        values_obs.append(collection_obs)  
                        if cos_center == self.cos_zenith_bin_centers[-1]: self.zenith_binned_obs_collection = dict(zip(self.cos_zenith_bin_centers,values_obs))

    def write_datastore_with_new_model(self, model_path=''):
        # For each observation get the acceptance map, save it and add the saved file path to the data store as a background map
        data_store = DataStore.from_dir(f"{self.save_path}",hdu_table_filename=f'hdu-index.fits.gz',obs_table_filename=f'obs-index.fits.gz')
        all_obs_ids = np.array(data_store.obs_table["OBS_ID"].data)
        
        # Bkg row cannot be overwritten, if it exists it needs to be removed before adding the new one
        if 'bkg' in data_store.hdu_table['HDU_TYPE']:
            data_store.hdu_table.remove_rows(data_store.hdu_table['HDU_TYPE']=='bkg')

        if model_path != '': path = Path(model_path)
        else: model_path = self.acceptance_files_dir

        if path.is_file():
            file_dir = path.parent
            file_name = path.name
        elif path.is_dir():
            paths = sorted(list(path.rglob("*.fits")))
            file_dir = model_path
            file_name = [file_path.name for file_path in paths]
        for i,obs_id in enumerate(all_obs_ids):
            if obs_id not in self.obs_ids: 
                data_store.hdu_table.remove_rows(data_store.hdu_table['OBS_ID']==obs_id)
                data_store.obs_table.remove_rows(data_store.obs_table['OBS_ID']==obs_id)
            else:
                data_store.hdu_table.add_row({'OBS_ID': f"{obs_id}", 
                                                'HDU_TYPE': 'bkg',
                                                "HDU_CLASS": f"bkg_{self.bkg_dim}d",
                                                "FILE_DIR": str(file_dir),
                                                "FILE_NAME": str(file_name[obs_id]) if path.is_dir() else str(file_name),
                                                "HDU_NAME": "BACKGROUND"})

        # Save the new data store for future use
        data_store.hdu_table.write(f"{self.save_path}/hdu-index{self.index_suffix}.fits.gz",format="fits",overwrite=True) 
        data_store.obs_table.write(f"{self.save_path}/obs-index{self.index_suffix}.fits.gz",format="fits",overwrite=True)
    
    def do_baccmod(self, config_path=None):
        if config_path is not None: self.init_config(config_path)

        self.load_observation_collection()

        if self.bkg_dim==2:
            acceptance_model_creator = RadialAcceptanceMapCreator(self.energy_axis_acceptance,
                                                                self.offset_axis_acceptance,
                                                                exclude_regions=self.exclude_regions,
                                                                initial_cos_zenith_binning=self.initial_cos_zenith_binning,
                                                                cos_zenith_binning_method=self.cos_zenith_binning_method,
                                                                cos_zenith_binning_parameter_value=self.cos_zenith_binning_parameter_value)
        elif self.bkg_dim==3:
            acceptance_model_creator = Grid3DAcceptanceMapCreator(self.energy_axis_acceptance,
                                                            self.offset_axis_acceptance,
                                                            exclude_regions=self.exclude_regions,
                                                            initial_cos_zenith_binning=self.initial_cos_zenith_binning,
                                                            cos_zenith_binning_method=self.cos_zenith_binning_method,
                                                            cos_zenith_binning_parameter_value=self.cos_zenith_binning_parameter_value,
                                                            method=self.method,
                                                            fit_fnc=self.fit_fnc,
                                                            fit_bounds=self.fit_bounds)

        acceptance_model = acceptance_model_creator.create_acceptance_map_per_observation(self.obs_collection,zenith_binning=self.zenith_binning,runwise_normalisation=self.runwise_normalisation) 

        if not self.multiple_simulation_subdir:
            # For each observation get the acceptance map, save it and add the saved file path to the data store as a background map
            self.data_store_out = DataStore.from_dir(f"{self.save_path}",hdu_table_filename=f'hdu-index.fits.gz',obs_table_filename=f'obs-index.fits.gz')

            # Bkg row cannot be overwritten, if it exists it needs to be removed before adding the new one
            if 'bkg' in self.data_store_out.hdu_table['HDU_TYPE']:
                self.data_store_out.hdu_table.remove_rows(self.data_store_out.hdu_table['HDU_TYPE']=='bkg')

            for i in range(len(self.obs_ids)):
                obs_id=self.obs_ids[i]
                hdu_acceptance = acceptance_model[obs_id].to_table_hdu()
                hdu_acceptance.writeto(f'{self.acceptance_files_dir}/acceptance_obs-{obs_id}.fits', overwrite=True)
                self.data_store_out.hdu_table.add_row({'OBS_ID': f"{obs_id}", 
                                                'HDU_TYPE': 'bkg',
                                                "HDU_CLASS": f"bkg_{self.bkg_dim}d",
                                                "FILE_DIR": str(self.acceptance_files_dir),
                                                "FILE_NAME": f'acceptance_obs-{obs_id}.fits',
                                                "HDU_NAME": "BACKGROUND"})

            # Save the new data store for future use
            self.data_store_out.hdu_table.write(f"{self.save_path}/hdu-index{self.index_suffix}.fits.gz",format="fits",overwrite=True) 
            self.data_store_out.obs_table.write(f"{self.save_path}/obs-index{self.index_suffix}.fits.gz",format="fits",overwrite=True)
            self.load_observation_collection(from_index=True, verbose=False)
        else:
            # I still don't know how to create a new datastore from an observation collection, without having to save them again and store them twice
            for i in range(len(self.obs_ids)):
                obs_id=self.obs_ids[i]
                hdu_acceptance = acceptance_model[obs_id].to_table_hdu()
                hdu_acceptance.writeto(f'{self.acceptance_files_dir}/acceptance_obs-{obs_id}.fits', overwrite=True)
        self.load_output_background_irfs()
    
    def get_unstacked_dataset(self, axis_info='irf'):
        source,source_pos,source_region = self.source_info
        if axis_info=='irf': emin,emax,offset_max,nbin_offset = self.axis_info_irf
        elif axis_info=='map': emin,emax,offset_max,nbin_offset = self.axis_info_map
        else: emin,emax,offset_max,nbin_offset = axis_info

        exclusion = self.region_shape != 'noexclusion'

        # Declare the non-spatial axes 
        unit="TeV"
        nbin_energy = 10

        energy_axis = MapAxis.from_energy_bounds(
            emin, emax, nbin=nbin_energy, per_decade=True, unit=unit, name="energy"
        )

        # Reduced IRFs are defined in true energy (i.e. not measured energy). 
        # The bounds need to take into account the energy dispersion. 
        edisp_frac = 0.3
        emin_true,emax_true = ((1-edisp_frac)*emin , (1+edisp_frac)*emax)
        nbin_energy_true = 20

        energy_axis_true = MapAxis.from_energy_bounds(
            emin_true, emax_true, nbin=nbin_energy_true, per_decade=True, unit=unit, name="energy_true"
        )
        
        nbins_map = 2*nbin_offset
        binsize = offset_max / (nbins_map / 2)

        # Create the geometry with the additional axes
        geom = WcsGeom.create(
            skydir=(source_pos.ra.degree,source_pos.dec.degree),
            binsz=binsize, 
            npix=(nbins_map, nbins_map),
            frame="icrs",
            proj="CAR",
            axes=[energy_axis],
        )

        # Get the energy-integrated image
        geom_image = geom.to_image().to_cube([energy_axis.squash()])

        base_map_dataset = MapDataset.create(geom=geom_image)
        unstacked_datasets = Datasets()

        maker = MapDatasetMaker(selection=["counts", "background"])
        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=offset_max)
        
        for obs in self.obs_collection:
            dataset_map = maker.run(base_map_dataset.copy(), obs)
            dataset_map = maker_safe_mask.run(dataset_map, obs)
            unstacked_datasets.append(dataset_map)
        return unstacked_datasets
    
    def get_stacked_dataset(self, bkg_method=None, axis_info='irf'):
        source,source_pos,source_region = self.source_info
        if axis_info=='irf': emin,emax,offset_max,nbin_offset = self.axis_info_irf
        elif axis_info=='map': emin,emax,offset_max,nbin_offset = self.axis_info_map
        else: emin,emax,offset_max,nbin_offset = axis_info

        exclusion = self.region_shape != 'noexclusion'

        # Declare the non-spatial axes 
        unit="TeV"
        nbin_energy = 10

        energy_axis = MapAxis.from_energy_bounds(
            emin, emax, nbin=nbin_energy, per_decade=True, unit=unit, name="energy"
        )

        # Reduced IRFs are defined in true energy (i.e. not measured energy). 
        # The bounds need to take into account the energy dispersion. 
        edisp_frac = 0.3
        emin_true,emax_true = ((1-edisp_frac)*emin , (1+edisp_frac)*emax)
        nbin_energy_true = 20

        energy_axis_true = MapAxis.from_energy_bounds(
            emin_true, emax_true, nbin=nbin_energy_true, per_decade=True, unit=unit, name="energy_true"
        )
        
        nbins_map = 2*nbin_offset
        binsize = offset_max / (nbins_map / 2)

        # Create the geometry with the additional axes
        geom = WcsGeom.create(
            skydir=(source_pos.ra.degree,source_pos.dec.degree),
            binsz=binsize, 
            npix=(nbins_map, nbins_map),
            frame="icrs",
            proj="CAR",
            axes=[energy_axis],
        )

        # Get the energy-integrated image
        geom_image = geom.to_image().to_cube([energy_axis])

        base_map_dataset = MapDataset.create(geom=geom_image)
        unstacked_datasets = Datasets()

        maker = MapDatasetMaker(selection=["counts", "background"])
        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=offset_max)
        
        for obs in self.obs_collection:
            dataset_map = maker.run(base_map_dataset.copy(), obs)
            dataset_map = maker_safe_mask.run(dataset_map, obs)
            unstacked_datasets.append(dataset_map)

        # Stack the datasets
        unstacked_datasets_local = unstacked_datasets.copy()

        ## Make the exclusion mask
        if exclusion: exclusion_mask = geom_image.region_mask(self.exclude_regions, inside=False)
        else: exclusion_mask=None
        
        ## Make the MapDatasetOnOff

        if bkg_method == 'ring': 
            internal_ring_radius, width_ring = self.ring_bkg_param
            ring_bkg_maker = RingBackgroundMaker(r_in=internal_ring_radius * u.deg,
                                                            width=width_ring * u.deg,
                                                            exclusion_mask=exclusion_mask)
            stacked_on_off = MapDatasetOnOff.create(geom=geom_image, energy_axis_true=energy_axis_true, name="stacked")
            print("Ring background method is applied")
        else: 
            stacked_on_off = Datasets()
            if bkg_method == 'FoV':
                FoV_background_maker = FoVBackgroundMaker(method=self.fov_bkg_param, exclusion_mask=exclusion_mask)
                print("FoV background method is applied")
            else: print("No background maker is applied")

        unstacked_datasets_local = unstacked_datasets.copy()

        for dataset_loc in unstacked_datasets_local:
            # Ring extracting makes sense only for 2D analysis
            if bkg_method == 'ring': 
                dataset_on_off = ring_bkg_maker.run(dataset_loc)
                stacked_on_off.stack(dataset_on_off)
            else:
                if bkg_method == 'FoV':  
                    dataset_loc.counts.data[~dataset_loc.mask_safe] = 0
                    dataset_loc = FoV_background_maker.run(dataset_loc)
                stacked_on_off.append(dataset_loc)
        
        if bkg_method == 'ring': return stacked_on_off
        else: return stacked_on_off.stack_reduce()
    
    def get_dfprofile(self, bkg_irf) -> pd.DataFrame:
        E_centers = bkg_irf.axes["energy"].center.value.round(1)
        fov_edges = bkg_irf.axes['fov_lon'].edges.value
        fov_centers = bkg_irf.axes['fov_lon'].center.value.round(1)
        offset_edges = fov_edges[fov_edges >= 0]
        offset_centers = fov_centers[fov_centers >= 0]

        radius_edges = self.radius_edges
        radius_centers = self.radius_centers

        dfcoord = pd.DataFrame(index=pd.Index(fov_centers,name='fov_lon'),columns=pd.Index(fov_centers,name='fov_lat'))
        for (lon,lat) in product(dfcoord.index,dfcoord.columns): dfcoord.loc[lon,lat] = np.sqrt(lon**2 + lat**2)
        dfcoord = dfcoord.apply(pd.to_numeric, errors='coerce')
        
        for iEbin,Ebin in enumerate(E_centers):
            dfdata = pd.DataFrame(data=np.array(bkg_irf.data[iEbin,:,:],dtype='float64'),index=pd.Index(fov_centers,name='fov_lon'),columns=pd.Index(fov_centers,name='fov_lat'))
            dfdata.replace(0, np.nan, inplace=True)
            dfprofile_lat = dfdata.describe()
            dfprofile_lat.loc['sum'] = dfdata.sum()
            dfprofile_lon= dfdata.T.describe()
            dfprofile_lon.loc['sum'] = dfdata.T.sum()

            for r_mid, r_low, r_high in zip(radius_centers,radius_edges[:-1],radius_edges[1:]):
                values = dfdata[(dfcoord >= r_low) & (dfcoord < r_high)].values
                dfvalues = pd.Series(values[~np.isnan(values)]).describe().to_frame().rename(columns={0:r_mid.round(1)})
                dfvalues.loc['sum'] = np.nansum(values)
                if r_mid == radius_centers[0]: dfprofile_rad = dfvalues.copy()
                else: dfprofile_rad = dfprofile_rad.join(dfvalues.copy())
            dfprofile_rad = pd.concat([dfprofile_rad], axis=0, keys=['fov_offset'])
            dfprofile_Ebin = pd.concat([dfprofile_lat, dfprofile_lon], keys=['fov_lat','fov_lon'],join='outer')
            dfprofile_Ebin = pd.concat([dfprofile_Ebin, dfprofile_rad], keys=['lon_lat','offset'],join='outer',axis=1)
            dfprofile_Ebin = pd.concat([dfprofile_Ebin], axis=1, keys=[Ebin])
            if iEbin == 0: 
                dfprofile = dfprofile_Ebin.copy()
                dfdata_all = dfdata.replace(np.nan, 0).copy()
            else: 
                dfprofile=pd.concat([dfprofile,dfprofile_Ebin],axis=1)
                dfdata_all = dfdata_all + dfdata.replace(np.nan, 0).copy()
        
        dfdata_all.replace(0, np.nan, inplace=True)
        dfprofile_lat = dfdata_all.describe()
        dfprofile_lat.loc['sum'] = dfdata_all.sum()
        dfprofile_lon= dfdata_all.T.describe()
        dfprofile_lon.loc['sum'] = dfdata_all.T.sum()

        for r_mid, r_low, r_high in zip(radius_centers,radius_edges[:-1],radius_edges[1:]):
            values = dfdata_all[(dfcoord >= r_low) & (dfcoord < r_high)].values
            dfvalues = pd.Series(values[~np.isnan(values)]).describe().to_frame().rename(columns={0:r_mid.round(1)})
            dfvalues.loc['sum'] = np.nansum(values)
            if r_mid == radius_centers[0]: dfprofile_rad = dfvalues.copy()
            else: dfprofile_rad = dfprofile_rad.join(dfvalues.copy())
        
        dfprofile_rad = pd.concat([dfprofile_rad], axis=0, keys=['fov_offset'])
        dfprofile_Ebin = pd.concat([dfprofile_lat, dfprofile_lon], keys=['fov_lat','fov_lon'],join='outer')
        dfprofile_Ebin = pd.concat([dfprofile_Ebin, dfprofile_rad], keys=['lon_lat','offset'],join='outer',axis=1)
        dfprofile_Ebin = pd.concat([dfprofile_Ebin], axis=1, keys=[-1.0])
        dfprofile=pd.concat([dfprofile,dfprofile_Ebin],axis=1)
        return dfprofile.sort_index()
    
    def plot_profile(self, irf='both', i_irf=0, profile='both', stat='sum', bias=False, ratio_lim = [0.95,1.05], all_Ebins=False, fig_save_path=''):
        if not self.external_data:
            if self.true_collection: bkg_true_irf = deepcopy(self.bkg_true_down_irf_collection[i_irf])
            else: bkg_true_irf = deepcopy(self.bkg_true_down_irf)
            
            profile_true = self.get_dfprofile(bkg_true_irf)
            fov_edges = bkg_true_irf.axes['fov_lon'].edges.value
            fov_centers = bkg_true_irf.axes['fov_lon'].center.value
            weights_Ebinall = np.sum(bkg_true_irf.data, axis=0)[np.newaxis, :, :].flatten()
        
        if (irf=='output') or (irf=='both'): 
            if self.out_collection: bkg_out_irf = deepcopy(self.bkg_output_irf_collection[i_irf])
            else: bkg_out_irf = deepcopy(self.bkg_output_irf)
            
            profile_out = self.get_dfprofile(bkg_out_irf)
            if self.external_data:
                fov_edges = bkg_out_irf.axes['fov_lon'].edges.value
                fov_centers = bkg_out_irf.axes['fov_lon'].center.value
                weights_Ebinall = np.sum(bkg_out_irf.data, axis=0)[np.newaxis, :, :].flatten()

        if bias:
            dummy_bkg_true = bkg_true_irf.data
            dummy_bkg_output = bkg_out_irf.data
            dummy_bkg_irf = deepcopy(bkg_out_irf)

            for iEbin,Ebin in enumerate(self.Ebin_mid):
                diff = dummy_bkg_output[iEbin,:,:] - dummy_bkg_true[iEbin,:,:]
                dummy_bkg_true[iEbin,:,:][np.where(dummy_bkg_true[iEbin,:,:] == 0.0)] = np.nan
                res = diff / dummy_bkg_true[iEbin,:,:]
                # diff[np.abs(res) >= 0.9] = np.nan
                dummy_bkg_irf.data[iEbin,:,:] = diff / dummy_bkg_true[iEbin,:,:]
            
            profile_bias = self.get_dfprofile(dummy_bkg_irf)
        
        E_centers = self.Ebin_mid
        offset_edges = fov_edges[fov_edges >= 0]
        offset_centers = fov_centers[fov_centers >= 0]
        radius_centers = self.radius_centers
        radius_edges = self.radius_edges
        
        # Flatten and filter NaNs
        x_centers = fov_centers
        y_centers = fov_centers
        X_, Y_ = np.meshgrid(x_centers, y_centers)
        r_center = np.sqrt(X_**2 + Y_**2).flatten()
        r_threshold = 99.8
        r_3sig = get_value_at_threshold(r_center, weights_Ebinall, r_threshold, plot=False)

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        lw=1
        ls_true,ls_out,ls_ratio = ("-","-","-")
        m_lat,m_lon = ("o","o")

        if (profile=='lon_lat') or (profile=='both'):
            suptitle='FoV coordinate profile'
            titles=['Longitude','Latitude']
            xlabels=["FoV Lat [°]", "FoV Lon [°]"]
            if bias: suptitle += f': bias {stat} (diff/true)' 
            if (irf=='output') or (irf=='both'):
                if self.external_data or bias: fig, (ax_lon,ax_lat) = plt.subplots(1,2,figsize=(12,(2.5/4)*6))
                else: fig, ((ax_lon,ax_lat),(ax_lon_ratio,ax_lat_ratio)) = plt.subplots(2,2,figsize=(12,6), gridspec_kw={'height_ratios': [2.5, 1.5]})
            if ((irf=='true') or (irf=='both')) and not bias: fig_true, (ax_lon_true,ax_lat_true) = plt.subplots(1,2,figsize=(12,(2.5/4)*6))
            for iEbin,Ebin in enumerate(np.concatenate((E_centers,[-1]))):
                if not all_Ebins and Ebin != -1: continue
                else:
                    if (Ebin != -1): label=self.Ebin_labels[iEbin]
                    else: label=f"{self.e_min.value:.1f}-{self.e_max.value:.1f} {self.energy_axis_acceptance.unit}"

                    if not self.external_data: fov_is_in_3sig = np.abs(profile_true.xs((Ebin), axis=1).xs(('lon_lat'),axis=1).loc[('fov_lon','sum')].index) < r_3sig
                    else: fov_is_in_3sig = np.abs(profile_out.xs((Ebin), axis=1).xs(('lon_lat'),axis=1).loc[('fov_lon','sum')].index) < r_3sig
                    
                    if (irf=='output') or (irf=='both'):
                        if bias:
                            y_lon=100*profile_bias.xs((Ebin), axis=1).xs(('lon_lat'),axis=1).loc[('fov_lon', stat)]
                            y_lat=100*profile_bias.xs((Ebin), axis=1).xs(('lon_lat'),axis=1).loc[('fov_lat', stat)]
                            ylabel=f'Bias {stat.capitalize()} [%]'
                            plot_condition = fov_is_in_3sig
                            if np.any(offset_edges >= r_3sig): xlim = [-offset_edges[offset_edges >= r_3sig][0],offset_edges[offset_edges >= r_3sig][0]]
                            else: xlim = [fov_edges[0],fov_edges[-1]]
                        else:
                            y_lon=profile_out.xs((Ebin), axis=1).xs(('lon_lat'),axis=1).loc[('fov_lon', stat)]
                            y_lat=profile_out.xs((Ebin), axis=1).xs(('lon_lat'),axis=1).loc[('fov_lat', stat)]
                            ylabel=f'{stat} [events]'
                            plot_condition = y_lon.index == y_lon.index
                            xlim = [fov_edges[0],fov_edges[-1]]
                        
                        if all_Ebins or (Ebin == -1):
                            sns.lineplot(x=fov_centers[plot_condition], y=y_lon[plot_condition], ax=ax_lon, label=label, lw=lw, ls=ls_true, marker=m_lon)
                            sns.lineplot(x=fov_centers[plot_condition], y=y_lat[plot_condition], ax=ax_lat, label=label, lw=lw, ls=ls_true, marker=m_lat)
                            
                        for iax, ax in enumerate([ax_lon,ax_lat]):
                            ax.set(xlim=xlim,title=titles[iax],xlabel=xlabels[iax],ylabel=ylabel)
                            ax.grid(True, alpha=0.2)
                            if not bias: ax.set(yscale='log')

                    if not self.external_data and not bias:
                        y_true_lon=profile_true.xs((Ebin), axis=1).xs(('lon_lat'),axis=1).loc[('fov_lon',stat)]
                        y_true_lat=profile_true.xs((Ebin), axis=1).xs(('lon_lat'),axis=1).loc[('fov_lat',stat)]
                        
                        if  ((irf=='true') or (irf=='both')) and (all_Ebins or (Ebin == -1)):
                            sns.lineplot(x=fov_centers, y=y_true_lon, ax=ax_lon_true, label=label, lw=lw, ls=ls_out, marker=m_lon)
                            sns.lineplot(x=fov_centers, y=y_true_lat, ax=ax_lat_true, label=label, lw=lw, ls=ls_out, marker=m_lat)

                            for iax, ax in enumerate([ax_lon_true,ax_lat_true]):
                                ax.set(xlim=xlim,title=titles[iax],xlabel=xlabels[iax],ylabel=ylabel)
                                ax.grid(True, alpha=0.2)
                                if not bias: ax.set(yscale='log')
                            fig_true.suptitle(suptitle+": true")
                            fig_true.tight_layout()
                    
                    if ((irf=='output') or  (irf=='both')) and not self.external_data and not bias:
                        y_ratio_lon = y_lon/y_true_lon
                        y_ratio_lat = y_lat/y_true_lat

                        if all_Ebins or (Ebin == -1):
                            sns.lineplot(x=fov_centers[fov_is_in_3sig], y=y_ratio_lon[fov_is_in_3sig], ax=ax_lon_ratio, lw=lw, ls=ls_ratio, marker=m_lon)
                            sns.lineplot(x=fov_centers[fov_is_in_3sig], y=y_ratio_lat[fov_is_in_3sig], ax=ax_lat_ratio, lw=lw, ls=ls_ratio, marker=m_lat)
                            for iax, ax in enumerate([ax_lon_ratio,ax_lat_ratio]):
                                ax.set(xlim=xlim,ylim=ratio_lim,xlabel=xlabels[iax], ylabel='Ratio (out / true)')
                                ax.grid(True, alpha=0.2)
            if not bias: suptitle += ': output'
            fig.suptitle(suptitle)
            fig.tight_layout()

            plt.show()
            if fig_save_path != '': fig.savefig(fig_save_path[:-4]+f"_coordinate_Ebin_{'all' if (Ebin==-1) else iEbin}.png", dpi=300, transparent=False, bbox_inches='tight')

        if (profile=='offset') or (profile=='both'):
            suptitle='FoV offset profile'
            title='Offset'
            xlabel="FoV offset [°]"
            if bias: suptitle += f': bias {stat} (diff/true)' 

            if (irf=='output') or (irf=='both'):
                if self.external_data or bias: fig, ax = plt.subplots(figsize=(12,(2.5/4)*6))
                else: fig, (ax,ax_ratio) = plt.subplots(2,1,figsize=(12,6), gridspec_kw={'height_ratios': [2.5, 1.5]})
            if ((irf=='true') or (irf=='both')) and not bias: fig_true, ax_true = plt.subplots(figsize=(12,(2.5/4)*6))
            for iEbin,Ebin in enumerate(np.concatenate((E_centers,[-1]))):
                if not all_Ebins and Ebin != -1: continue
                else:
                    if (Ebin != -1): label=self.Ebin_labels[iEbin]
                    else: label=f"{self.e_min.value:.1f}-{self.e_max.value:.1f} {self.energy_axis_acceptance.unit}"

                    offset_is_in_3sig = radius_edges[:-1] < r_3sig
                    
                    if (irf=='output') or (irf=='both'):
                        if bias:
                            y=100*profile_bias.xs((Ebin), axis=1).xs(('offset'),axis=1).loc[('fov_offset', stat)]
                            ylabel=f'Bias {stat.capitalize()} [%]'
                            plot_condition = offset_is_in_3sig
                            if np.any(offset_edges >= r_3sig): xlim = [0,offset_edges[offset_edges >= r_3sig][0]]
                            else: xlim = [0,fov_edges[-1]]
                        else:
                            y=profile_out.xs((Ebin), axis=1).xs(('offset'),axis=1).loc[('fov_offset', stat)]
                            ylabel=f'{stat} [events]'
                            plot_condition = y.index == y.index
                            xlim = [0,fov_edges[-1]]
                        
                        if all_Ebins or (Ebin == -1):
                            sns.lineplot(x=radius_centers[plot_condition], y=y[plot_condition], ax=ax, label=label, lw=lw, ls=ls_true, marker=m_lon)
                            
                            ax.set(xlim=xlim,title=title,xlabel=xlabel,ylabel=ylabel)
                            ax.grid(True, alpha=0.2)
                            if not bias: ax.set(yscale='log')

                
                    if not self.external_data and not bias:
                        y_true=profile_true.xs((Ebin), axis=1).xs(('offset'),axis=1).loc[('fov_offset',stat)]
                        
                        if  ((irf=='true') or  (irf=='both')) and  (all_Ebins or (Ebin == -1)):
                            sns.lineplot(x=radius_centers, y=y_true, ax=ax_true, label=label, lw=lw, ls=ls_out, marker=m_lon)

                            ax_true.set(xlim=xlim,title=title,xlabel=xlabel,ylabel=ylabel)
                            ax_true.grid(True, alpha=0.2)
                            if not bias: ax_true.set(yscale='log')
                            fig_true.suptitle(suptitle+": true")
                            fig_true.tight_layout()
                
                if ((irf=='output') or (irf=='both')) and not self.external_data and not bias:
                        y_ratio = y/y_true

                        if all_Ebins or (Ebin == -1):
                            sns.lineplot(x=radius_centers[offset_is_in_3sig], y=y_ratio[offset_is_in_3sig], ax=ax_ratio, lw=lw, ls=ls_ratio, marker=m_lon)
                            ax_ratio.set(xlim=xlim,ylim=ratio_lim,xlabel=xlabel, ylabel='Ratio (out / true)')
                            ax_ratio.grid(True, alpha=0.2)
            if not bias: suptitle += ': output'
            fig.suptitle(suptitle)
            fig.tight_layout()

            plt.show()
            if fig_save_path != '': fig.savefig(fig_save_path[:-4]+f"_offset_Ebin_{'all' if (Ebin==-1) else iEbin}.png", dpi=300, transparent=False, bbox_inches='tight')    

    def plot_model(self, data='acceptance', irf='true', residuals='none', profile='none', downsampled=True, i_irf=0, zenith_binned=False, title='', fig_save_path='', plot_hist=False) -> None:
        '''
        data types = ['acceptance', 'bkg_map']
        irf types = ['true', 'output', 'both']
        residuals types = ['none',' diff/true', 'diff/sqrt(true)']
        profile types = ['none','radial','lon_lat','all']
        '''
        res_lim_for_nan = 0.9

        fov_max = self.size_fov_acc.to_value(u.deg)
        fov_lim = [-fov_max,fov_max]
        fov_bin_edges = np.linspace(-fov_max,fov_max,7)

        # TO-DO: chose which irf you want to to compare instead of first one by default
        
        plot_true_data = (irf == 'true') or (irf == 'both')
        if plot_true_data:
            if zenith_binned:
                plot_true_data = i_irf in self.zenith_binned_bkg_true_down_irf_collection.keys()
                if plot_true_data:
                    zenith_binned_models = self.zenith_binned_bkg_true_down_irf_collection[i_irf]
                    for i, model in enumerate(zenith_binned_models):
                        if i==0: true = zenith_binned_models[0]
                        else: true.data += model.data
                    true.data /= (i)
                else: print(f"No model in zenith bin with mean zd = {np.rad2deg(np.arccos(i_irf)):.1f}°")
            else:
                if self.true_collection: true = self.bkg_true_down_irf_collection[i_irf+1]
                else: true = self.bkg_true_down_irf
        
        plot_out_data = (irf == 'output') or (irf == 'both')
        if plot_out_data: 
            if zenith_binned:
                plot_out_data = i_irf in self.zenith_binned_bkg_output_irf_collection.keys()
                if plot_out_data:
                    zenith_binned_models = self.zenith_binned_bkg_output_irf_collection[i_irf]
                    for i, model in enumerate(zenith_binned_models):
                        if i==0: out = zenith_binned_models[0]
                        else: out.data += model.data
                    out.data /= (i)
                else: print(f"No model in zenith bin with mean zd = {np.rad2deg(np.arccos(i_irf)):.1f}°")
            else:
                if self.out_collection: out = self.bkg_output_irf_collection[i_irf+1]
                else: out = self.bkg_output_irf
        
        plot_residuals_data = (residuals != "none") & (plot_true_data+plot_out_data == 2)
        
        if plot_true_data or plot_out_data:
            radec_map = (data == 'bkg_map')
            if radec_map:            
                # By default the map is for W1 pointing
                # TO-DO: option to chose the pointing
                pointing, run_info = self.wobble_pointings[0], self.wobble_run_info[0]
                pointing_info = FixedPointingInfo(mode=PointingMode.POINTING, fixed_icrs=pointing, location=self.loc)
                obstime=Time(self.t_ref)
                ontime= self.livetime_simu * u.s

                if downsampled:
                    geom_irf = self.bkg_true_down_irf
                    oversampling=None
                    offset_max = self.size_fov_acc
                else: 
                    geom_irf = self.bkg_true_irf
                    oversampling=self.down_factor
                    offset_max = self.size_fov_irf
                
                geom=get_geom(geom_irf,None,run_info)

                if plot_true_data:
                    map_true = make_map_background_irf(pointing_info, ontime, true, geom, oversampling=oversampling, use_region_center=True, obstime=obstime, fov_rotation_error_limit=self.fov_rotation_error_limit)
                    map_true_cut = map_true.cutout(position=pointing,width=2*offset_max)
                    map_true_cut.sum_over_axes(["energy"]).plot()
                    true = map_true.data

                if plot_out_data:
                    map_out = make_map_background_irf(pointing_info, ontime, out, geom, oversampling=oversampling, use_region_center=True, obstime=obstime, fov_rotation_error_limit=self.fov_rotation_error_limit)
                    map_out_cut = map_true.cutout(position=pointing,width=2*offset_max)
                    map_out_cut.sum_over_axes(["energy"]).plot()
                    out = map_out.data
                
                xlabel,ylabel=("Ra offset [°]", "Dec offset [°]")
                cbar_label = 'Counts'
            else: 
                xlabel,ylabel=("FoV Lat [°]", "FoV Lon [°]")
                cbar_label = 'Background [MeV$^{-1}$s$^{-1}$sr$^{-1}$]'
                if plot_true_data: true = true.data
                if plot_out_data: out = out.data
            
            res_type_label = " diff / true" if residuals == "diff/true" else "diff / $\sqrt{true}$"

            rot = 65
            nncols = 3
            n = self.nbin_E_acc
            cols = min(nncols, n)
            rows = 1 + (n - 1) // cols
            width = 16
            cfraction = 0.15

            if plot_residuals_data: self.res_arr = compute_residuals(out, true, residuals=residuals,res_lim_for_nan=0.9)

            fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(width, rows * width // (cols * (1 + cfraction))))
            for iax, ax in enumerate(axs.flat[:n]):
                if plot_residuals_data:
                    res_label = f'{res_type_label} [%]'
                    res = self.res_arr[iax,:,:]
                    res *= 100
                    vlim = np.nanmax(np.abs(res))
                    colorbarticks = np.concatenate((np.flip(-np.logspace(-1,3,5)),[0],np.logspace(-1,3,5)))
                    heatmap = ax.imshow(res, origin='lower', norm=SymLogNorm(linthresh=0.03, linscale=0.03, vmin=-vlim, vmax=vlim), cmap='coolwarm')
                    plt.colorbar(heatmap,ax=ax, shrink=1, ticks=colorbarticks, label=res_label)
                else:
                    if (irf == 'output'): data = out
                    elif (irf == 'true'): data = true
                    elif (irf == 'both'): ValueError("IRF type cannot be both, please chose between true and output")

                    data[iax,:,:][np.where(data[iax,:,:] == 0.0)] = np.nan
                    heatmap = ax.imshow(data[iax,:,:], origin='lower', cmap='viridis')

                    plt.colorbar(heatmap,ax=ax, shrink=1, label=cbar_label)

                x_lim = ax.get_xlim()
                xticks_new = scale_value(fov_bin_edges,fov_lim,x_lim).round(1)
                if radec_map: ax.set_xticks(rotation=rot, ticks=xticks_new, labels=np.flip(fov_bin_edges.round(1)))
                else: ax.set_xticks(rotation=rot, ticks=xticks_new, labels=fov_bin_edges.round(1))
                ax.set_yticks(ticks=xticks_new, labels=fov_bin_edges.round(1))
                ax.set(title=self.Ebin_labels[iax],xlabel=xlabel,ylabel=ylabel)

            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
            if fig_save_path != '': fig.savefig(fig_save_path, dpi=300, transparent=False, bbox_inches='tight')
            if plot_residuals_data: 
                if plot_hist:
                    fig_hist, ax_hist = plt.subplots(figsize=(4,3))
                    res_arr_flat = np.array(self.res_arr).flatten()
                    res_arr_flat = res_arr_flat[~np.isnan(res_arr_flat)]
                    res_arr_flat_absmax = np.max(np.abs(res_arr_flat))
                    sns.histplot(res_arr_flat, bins=np.linspace(-res_arr_flat_absmax,res_arr_flat_absmax,20), ax=ax_hist, element='step', fill=False, stat='density', color='k',line_kws={'color':'k'})
                    ax_hist.set(title=f'{title} distribution', xlabel=res_label, yscale='log')
                    # Define the text content and position it on the right
                    textstr = '\n'.join((
                        r'$\mu=%.2f$' % (np.nanmean(res_arr_flat), ),
                        r'$\mathrm{median}=%.2f$' % (np.nanmedian(res_arr_flat), ),
                        r'$\sigma=%.2f$' % (np.nanstd(res_arr_flat), ),
                        f'sum= {np.sum(res_arr_flat):.2f}'
                    ))

                    # Add the text box on the right of the plot
                    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                    ax_hist.text(1.05, 0.8, textstr, transform=ax_hist.transAxes,
                                fontsize=10, verticalalignment='center', bbox=props)
                    plt.show()
                    if fig_save_path != '': fig.savefig(fig_save_path[:-4]+'_distrib.png', dpi=300, transparent=False, bbox_inches='tight')
        else: print("No data to plot")
    
    def plot_zenith_binned_model(self, data='acceptance', irf='output', i_bin=-1, zenith_bins='baccmod', residuals='none', profile='none', fig_save_path='') -> None:
        '''Create a zenith binned collection and plot model data
        collections = ["true", "output", "observation"]: List of the collections you want binned with
        zenith_bins = int, np.array, 'stored', 'baccmod'
        By default all zenith bins are plotted with i_bin == -1
        Set it to bin index value to plot a single bin'''
        plot_all_bins = (i_bin ==-1)
        collections = ["true", "output"] if irf == "both" else [irf]
        self.create_zenith_binned_collections(collections=collections, zenith_bins=zenith_bins)
        for icos,cos_center in enumerate(self.cos_zenith_bin_centers):
            if plot_all_bins or (icos == i_bin):
                zd_bin_center = np.rad2deg(np.arccos(cos_center))
                title = f"Zenith binned averaged model data\nzd = {zd_bin_center:.1f}°, {self.obs_in_coszd_bin[icos].shape[0]} runs"
                if fig_save_path == '': fig_save_path_zd_bin=f"{self.plots_dir}/averaged_binned_acceptance_zd_{zd_bin_center:.0f}.png"
                else:  fig_save_path_zd_bin=f"{fig_save_path[:-4]}_{zd_bin_center:.0f}.png"
                print(fig_save_path_zd_bin)
                self.plot_model(data=data, irf=irf, residuals=residuals, profile=profile, downsampled=True, i_irf=cos_center, zenith_binned=True, title=title, fig_save_path=fig_save_path_zd_bin, plot_hist=False)
    
    def plot_zenith_binned_data(self, data='livetime', per_wobble=True):
        if self.cos_zenith_binning_method == "livetime": print(f"observation per bin: ", 
                                                                list(np.histogram(self.cos_zenith_observations, bins=np.flip(self.cos_zenith_bin_edges))[0]))
        wobble_observations = self.dfobs_table.WOBBLE.to_numpy()
        wobble_observations_bool_arr = np.array([(wobble_observations == wobble) for wobble in np.unique(np.array(wobble_observations))]).astype(int)
        livetime_observations = np.array([obs.observation_live_time_duration.to_value(u.s) for obs in self.obs_collection])

        if data == "livetime":
            cut_variable_weights = livetime_observations
        elif data == "observation":
            cut_variable_weights = np.ones(len(self.cos_zenith_observations),dtype=int)

        livetime_observations_and_wobble = [np.array(cut_variable_weights)*wobble_bool for wobble_bool in wobble_observations_bool_arr]

        for i,wobble in enumerate(np.unique(np.array(wobble_observations))):
            print(f"{wobble} observation per bin: {np.histogram(self.cos_zenith_observations, bins=np.flip(self.cos_zenith_bin_edges), weights=1*wobble_observations_bool_arr[i])[0]}")
            print(f"{wobble} livetime per bin: {np.histogram(self.cos_zenith_observations, bins=np.flip(self.cos_zenith_bin_edges), weights=livetime_observations_and_wobble[i])[0].round(0)}")

        fig,ax=plt.subplots(figsize=(5,5))

        title = f"Livetime{' per wobble and' * per_wobble} per cos(zd) bin\n"
        if per_wobble:
            for i,wobble in enumerate(np.unique(np.array(self.dfobs_table.WOBBLE))): 
                ax.hist(self.cos_zenith_observations, bins=np.flip(self.cos_zenith_bin_edges),weights=livetime_observations_and_wobble[i],alpha=0.6,label=f"{wobble}")
        else: ax.hist(self.cos_zenith_observations, bins=np.flip(self.cos_zenith_bin_edges),weights=livetime_observations,alpha=0.6)

        new_ticks_coszd=np.concatenate(([self.cos_zenith_bin_edges[-1]],self.cos_zenith_bin_centers,[self.cos_zenith_bin_edges[0]]))

        ax.set_xticks(new_ticks_coszd)
        ax.set_xticklabels(np.round(new_ticks_coszd,2), rotation=45)

        # Create a second x-axis
        ax2 = ax.twiny()
        ax2.set_xticks(new_ticks_coszd)
        ax2.set_xticklabels(np.degrees(np.arccos(new_ticks_coszd)).astype(int), rotation=45)

        # Set labels
        ax.set_xlabel('cos(zd) bin center')
        ax.set_ylabel('livetime [s]')
        ax2.set_xlabel('cos(zd) bin center in zenith [°]')

        xlim=(self.cos_zenith_bin_edges[-1],self.cos_zenith_bin_edges[0])
        ax.set_xlim(xlim)
        ax2.set_xlim(xlim)
        # if 'livetime' in self.cos_zenith_binning_method: ax.hlines([min_cut_per_cos_zenith_bin],xlim[0],xlim[1],ls='-',color='red',label='min livetime',alpha=0.5)

        ylim=ax.get_ylim()    
        ax.vlines(self.cos_zenith_bin_edges,ylim[0],ylim[1],ls=':',color='grey',label='bin edges',alpha=0.5)
        ax.legend(loc='best')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_exclusion_mask(self, i_wobble=0, figsize=(4,4)):
        geom=get_geom(None,self.axis_info_dataset,self.wobble_mean_pointings[i_wobble])
        geom_image = geom.to_image().to_cube([self.energy_axis_acceptance.squash()])

        # Make the exclusion mask
        exclusion_mask = geom_image.region_mask(self.exclude_regions, inside=False)

        #Plot
        fig,plot=plt.subplots(figsize=figsize)
        exclusion_mask.cutout(self.source_pos, self.size_fov_acc).plot(fig=fig)
        plt.show()
    
    def plot_residuals_histogram(self, skymaps_dict=None) -> plt.figure:
        if isinstance(skymaps_dict, dict): self.skymaps_dict = skymaps_dict
        elif skymaps_dict is None:
            if not hasattr(self, 'skymaps_dict'): ValueError("No stored skymaps_dict")
        
        if ('significance_all' in self.skymaps_dict.keys()) & ('significance_off' in self.skymaps_dict.keys()):
            geom_map = self.skymaps_dict["significance_all"]._geom
            energy_axis_map = geom_map.axes[0]
            geom_image = geom_map.to_image().to_cube([energy_axis_map.squash()])
            exclusion_mask = geom_image.region_mask(self.exclude_regions, inside=False)

            significance_all = self.skymaps_dict["significance_all"].data[np.isfinite(self.skymaps_dict["significance_all"].data)]
            significance_off = self.skymaps_dict["significance_off"].data[np.logical_and(np.isfinite(self.skymaps_dict["significance_all"].data), 
                                                                    exclusion_mask.data)]
        else:
            ValueError("Significance maps not in skymaps_dict")
        

        emin_map = energy_axis_map.edges.min()
        emax_map = energy_axis_map.edges.max()

        fig, ax1 = plt.subplots(figsize=(4,4))
        ax1.hist(
            significance_all,
            range=(-8,8),
            density=True,
            alpha=0.5,
            color="red",
            label="all bins",
            bins=30,
        )

        ax1.hist(
            significance_off,
            range=(-8,8),
            density=True,
            alpha=0.5,
            color="blue",
            label="off bins",
            bins=30,
        )

        # Now, fit the off distribution with a Gaussian
        mu, std = norm_stats.fit(significance_off)
        x = np.linspace(-8, 8, 30)
        p = norm_stats.pdf(x, mu, std)
        ax1.plot(x, p, lw=2, color="black")
        p2 = norm_stats.pdf(x, 0, 1)
        #ax.plot(x, p2, lw=2, color="green")
        ax1.set_title(f"Background residuals map E= {emin_map:.1f} - {emax_map:.1f}")
        ax1.text(-2.,0.001, f'mu = {mu:3.2f}\nstd={std:3.2f}',fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_xlabel("Significance")
        ax1.set_yscale("log")
        ax1.set_ylim(1e-5, 1)
        ax1.set_xlim(-5,6.)
        #xmin, xmax = np.min(significance_off), np.max(significance_off)
        #ax.set_xlim(xmin, xmax)
        print("mu = ", mu," std= ",std )
        plt.tight_layout()
        return fig

    def plot_lima_maps(self, dataset, axis_info, cutout=False, method='FoV', fig_save_path=''):
        '''Compute and store skymaps'''
        emin_map, emax_map, offset_max, nbin_offset = axis_info
        internal_ring_radius,width_ring = self.ring_bkg_param
        if cutout: dataset = dataset.cutout(position = self.source_pos, width=offset_max)

        self.exclusion_mask = get_exclusion_mask_from_dataset_geom(dataset, self.exclude_regions)
        self.skymaps_dict = get_skymaps_dict(dataset, self.exclude_regions, self.correlation_radius, self.correlate_off, 'all')
        
        significance_all = self.skymaps_dict["significance_all"]

        # Significance and excess
        fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(
            figsize=(18, 11),subplot_kw={"projection": significance_all.geom.wcs}, ncols=3, nrows=2
        )
        # fig.delaxes(ax1)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        ax1.set_title("Spatial residuals map: diff/sqrt(model)")
        dataset.plot_residuals_spatial(method='diff/sqrt(model)',ax=ax1, add_cbar=True, stretch="linear",norm=CenteredNorm())
        # plt.colorbar(g,ax=ax1, shrink=1, label='diff/sqrt(model)')
        
        ax4.set_title("Significance map")
        #significance_map.plot(ax=ax1, add_cbar=True, stretch="linear")
        self.skymaps_dict["significance_all"].plot(ax=ax4, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')
        
        ax5.set_title("Off significance map")
        #significance_map.plot(ax=ax1, add_cbar=True, stretch="linear")
        self.skymaps_dict["significance_off"].plot(ax=ax5, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')

        ax6.set_title("Excess map")
        self.skymaps_dict["excess"].plot(ax=ax6, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')
        
        # Background and counts

        ax2.set_title("Background map")
        self.skymaps_dict["background"].plot(ax=ax2, add_cbar=True, stretch="linear")

        ax3.set_title("Counts map")
        self.skymaps_dict["counts"].plot(ax=ax3, add_cbar=True, stretch="linear")
        
        if method=='ring':
            ring_center_pos = self.source_pos
            r1 = SphericalCircle(ring_center_pos, self.exclusion_radius,
                            edgecolor='yellow', facecolor='none',
                            transform=ax5.get_transform('icrs'))
            r2 = SphericalCircle(ring_center_pos, internal_ring_radius * u.deg,
                                edgecolor='white', facecolor='none',
                                transform=ax5.get_transform('icrs'))
            r3 = SphericalCircle(ring_center_pos, internal_ring_radius * u.deg + width_ring * u.deg,
                                edgecolor='white', facecolor='none',
                                transform=ax5.get_transform('icrs'))
            ax5.add_patch(r2)
            ax5.add_patch(r1)
            ax5.add_patch(r3)
        plt.tight_layout()

        if fig_save_path == '': fig_save_path=f"{self.plots_dir}/skymaps.png"
        fig.savefig(f"{fig_save_path[:-4]}_data.png", dpi=300, transparent=False, bbox_inches='tight')
        
        # Residuals
        fig = self.plot_residuals_histogram()
        fig.savefig(f"{fig_save_path[:-4]}_sigma_residuals.png", dpi=300, transparent=False, bbox_inches='tight')

    def plot_skymaps(self, bkg_method='ring', stacked_dataset=None, axis_info_dataset=None, axis_info_map=None):
        if axis_info_dataset is not None: self.axis_info_dataset = axis_info_dataset
        # if axis_info_map is not None: self.axis_info_map = axis_info_map
        
        if stacked_dataset is None: 
            self.stacked_dataset = self.get_stacked_dataset(bkg_method=bkg_method, axis_info=self.axis_info_dataset)
        elif isinstance(stacked_dataset, MapDatasetOnOff): 
            self.stacked_dataset = stacked_dataset
            e_min = self.stacked_dataset._geom.axes["energy"].edges.min().to(u.TeV)
            e_max = self.stacked_dataset._geom.axes["energy"].edges.max().to(u.TeV)
            offset_max = self.stacked_dataset._geom.width.to(u.deg)[0][0]
            nbin_offset = int(self.stacked_dataset._geom._shape[0]/2)
            self.axis_info_dataset = [e_min, e_max, offset_max, nbin_offset]
        else: ValueError("No valid value for stacked_datasets")
        
        # TO-DO: change this to be able to give different parameters for dataset and map
        self.axis_info_map = self.axis_info_dataset
        cutout = (self.axis_info_dataset[2] != self.axis_info_map[2]) & (self.axis_info_dataset[2] > self.axis_info_map[2])
        
        self.plot_lima_maps(self.stacked_dataset, self.axis_info_dataset, cutout, bkg_method)
