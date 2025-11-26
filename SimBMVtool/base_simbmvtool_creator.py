import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from toolbox import (get_obs_collection,
                    get_run_info,
                    get_dfobs_table)

from bkg_irf_models import (GaussianSpatialModel_LinearGradient,
                            GaussianSpatialModel_LinearGradient_half)

from bkg_irf import (evaluate_bkg,
                    get_bkg_irf,
                    get_irf_map,
                    get_cut_downsampled_irf_from_map,
                    get_bkg_true_irf_from_config)

import gammapy
gammapy_v = gammapy.__version__
gammapy_custom = '0.8.dev17165+g9e45af09c.d20241210'
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

from itertools import product
from baccmod import RadialAcceptanceMapCreator, Grid3DAcceptanceMapCreator, BackgroundCollectionZenith
from baccmod.toolbox import (get_unique_wobble_pointings)

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union
from gammapy.datasets import MapDatasetEventSampler
if gammapy_v == gammapy_custom: from gammapy.irf import Background3D, BackgroundIRF
else: from gammapy.irf.background import Background3D, BackgroundIRF
from gammapy.makers.utils import make_map_background_irf


# Site locations for main gamma-ray experiments

HESS = EarthLocation(lat=-23.271667*u.deg, lon=16.5*u.deg, height=1800*u.m)
VERITAS = EarthLocation(lat=31.6750*u.deg, lon=-110.9519*u.deg, height=1268*u.m)
CTAN = EarthLocation(lat=28.761944*u.deg, lon=-17.890556*u.deg, height=2190*u.m)
CTAS = EarthLocation(lat=-24.683333*u.deg, lon=-70.3*u.deg, height=2150*u.m)
HAWC = EarthLocation(lat=18.994722*u.deg, lon=-97.308611*u.deg, height=4100*u.m)
LHAASO = EarthLocation(lat=29.358611*u.deg, lon=100.154167*u.deg, height=4410*u.m)
AUGER = EarthLocation(lat=-35.103333*u.deg, lon=-69.5*u.deg, height=1400*u.m)

sites = {
    "HESS": HESS,
    "VERITAS": VERITAS,
    "CTAN": CTAN,
    "CTAS": CTAS,
    "HAWC": HAWC,
    "LHAASO": LHAASO,
    "AUGER": AUGER,
}


class BaseSimBMVtoolCreator(ABC):

    def __init__(self,
                simulator=False,
                external_data=False,
                real_data=False,
                external_model=False) -> None:
        """
        Create the class to perform model validation.

        TO-DO: Parameters description
        ----------
        """
        self.simulator=simulator
        self.external_data=external_data
        self.real_data=real_data
        self.external_model=external_model

    def init_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        # Print the loaded configuration
        for key in config.keys(): print(f"{key}: {config[key]}")
        self.config = config
        self.config_path = config_path
        self.cfg_paths = config["paths"]
        self.cfg_data = self.config["data"]
        self.cfg_source = config["source"]
        self.cfg_background = config["background"]
        self.cfg_acceptance = config["acceptance"]
        if self.simulator or (not self.external_data):
            self.cfg_simulation = config["simulation"]
            self.cfg_wobbles = config["wobbles"]
            self.cfg_irf = config["background"]["irf"]

        # Paths
        self.path_data=self.cfg_paths["path_data"] # path to files to get run information used for simulation
        self.output_dir = self.cfg_paths["output_dir"]
        self.subdir = self.cfg_paths["subdir"]
        if self.simulator or (not self.real_data and not self.external_data): self.simulated_obs_dir = f"{self.output_dir}/{self.subdir}/simulated_data"
        
        # Data
        self.obs_pattern = self.cfg_data["obs_pattern"] # If not external data, pattern of the input IRFs' observations, else pattern to load the observations
        if self.simulator: self.run_list = np.array([])

        # Source
        try:
            catalog = SourceCatalogGammaCat(os.environ.get('GAMMAPY_DATA')+'/catalogs/gammacat/gammacat.fits.gz')
        except:
            ValueError('No shell variable named GAMMAPY_DATA, please export path to gammapy-datasets')
        
        self.source_name=self.cfg_source["catalog_name"]
        if self.source_name in catalog.table['common_name']:
            self.source = catalog[self.source_name]
            self.source_pos=self.source.position
        else:
            self.source = SourceCatalogObject(data={'Source_Name':self.source_name,'RA': self.cfg_source["coordinates"]["ra"]*u.deg, 'DEC':self.cfg_source["coordinates"]["dec"]})
            self.source_pos=self.source.position
        
        if self.simulator:
            self.source_info = [self.source_name, self.source_pos, None]
            self.flux_to_0 = self.cfg_source["flux_to_0"]
            self.custom_source = self.cfg_source["is_custom"]
            if self.custom_source:
                custom_source_dict = self.cfg_source["custom_source"]
                custom_source_dict["name"] = self.source_name
                self.source_model = SkyModel.from_dict(custom_source_dict)

        # Output background model (Acceptance parameters are used to define oversampled true IRF)
        self.bkg_dim = self.cfg_acceptance["dimension"]
        self.fov_alignment=self.cfg_acceptance["FoV_alignment"]
        if "fov_rotation_error_limit" in list(self.cfg_acceptance.keys()):
            self.fov_rotation_error_limit=self.cfg_acceptance["fov_rotation_error_limit"] * u.deg

        self.e_min, self.e_max = float(self.cfg_acceptance["energy"]["e_min"])* u.TeV, float(self.cfg_acceptance["energy"]["e_max"])*u.TeV
        self.size_fov_acc = float(self.cfg_acceptance["offset"]["offset_max"]) * u.deg
        self.nbin_E_acc, self.nbin_offset_acc = self.cfg_acceptance["energy"]["nbin"],self.cfg_acceptance["offset"]["nbin"]

        self.offset_axis_acceptance = MapAxis.from_bounds(0.*u.deg, self.size_fov_acc, nbin=self.nbin_offset_acc, name='offset')
        self.energy_axis_acceptance = MapAxis.from_energy_bounds(self.e_min, self.e_max, nbin=self.nbin_E_acc, name='energy')
        
        self.Ebin_edges=self.energy_axis_acceptance.edges
        self.Ebin_tuples = np.array([(edge_inf.value,edge_sup.value) for edge_inf,edge_sup in zip(self.Ebin_edges[:-1],self.Ebin_edges[1:])])
        self.Ebin_labels = [f"{ebin_min:.1f}-{ebin_max:.1f} {self.energy_axis_acceptance.unit}" for (ebin_min, ebin_max) in self.Ebin_tuples]

        self.Ebin_mid = self.energy_axis_acceptance.center.value.round(1)
        self.ncols = int((len(self.Ebin_mid)+1)/2)
        self.nrows = int(len(self.Ebin_mid)/self.ncols)

        self.radius_edges = np.linspace(0,np.sqrt(2 * self.size_fov_acc.value**2),10)
        self.radius_centers = self.radius_edges[:-1]+0.5*(self.radius_edges[1:]-self.radius_edges[:-1])

        if self.simulator or not self.external_data:
            # True background model
            self.true_collection = self.cfg_acceptance["true_collection"]
            self.down_factor=self.cfg_irf["down_factor"]
            self.nbin_E_irf = self.nbin_E_acc * self.down_factor

            self.energy_axis_irf = MapAxis.from_energy_bounds(self.e_min, self.e_max, nbin=self.nbin_E_irf, name='energy')

            self.offset_factor=self.cfg_irf["offset_factor"]
            self.size_fov_irf = self.size_fov_acc * self.offset_factor
            self.nbin_offset_irf = self.nbin_offset_acc * self.down_factor * self.cfg_irf["offset_factor"]
            self.offset_axis_irf = MapAxis.from_bounds(0.*u.deg, self.size_fov_irf, nbin=self.nbin_offset_irf, name='offset')

            ## Spectral model
            bkg_model_dict = self.cfg_background["custom_source"]
            bkg_spectral_model = PowerLawNormSpectralModel.from_dict(bkg_model_dict["spectral"]) 

            ## Spatial model
            spatial_model = self.cfg_background['custom_source']["spatial"]["type"]
            if spatial_model ==  "GaussianSpatialModel_LinearGradient":
                bkg_spatial_model = GaussianSpatialModel_LinearGradient.from_dict(bkg_model_dict["spatial"])
            else: bkg_spatial_model = SpatialModel.from_dict(bkg_model_dict["spatial"])

            self.bkg_true_model = FoVBackgroundModel(dataset_name=bkg_model_dict["name"], spatial_model=bkg_spatial_model, spectral_model=bkg_spectral_model)
                # Simulated background
        if self.simulator or (not self.real_data and not self.external_data):
            self.lon_grad = 0 if spatial_model !=  "GaussianSpatialModel_LinearGradient" else bkg_spatial_model.lon_grad.value
            self.lat_grad = 0 if spatial_model !=  "GaussianSpatialModel_LinearGradient" else bkg_spatial_model.lat_grad.value
        
        # Some variables used for the custom plotting methods
        self.axis_info_acceptance = [self.e_min,self.e_max,self.size_fov_acc,self.nbin_offset_acc]
        self.axis_info_dataset = [self.e_min, self.e_max, self.nbin_E_acc, self.size_fov_acc, (2*self.size_fov_acc.to_value(u.deg),2*self.size_fov_acc.to_value(u.deg))]
        self.axis_info_map = [self.e_min, self.e_max, self.size_fov_acc, 10 * self.nbin_offset_acc]

        # Simulation

        if self.simulator or not self.external_data:
            self.loc = EarthLocation.of_site(self.cfg_simulation['earthlocation_obs_site'])
            self.t_ref = self.cfg_simulation["t_ref"]
            self.delay = self.cfg_simulation["delay"]
            self.time_oversampling = self.cfg_simulation["time_oversampling"] * u.s
        
            self.single_pointing = self.cfg_simulation["single_pointing"] # Code will just use wobble_1 informations
            self.obs_collection_type = self.cfg_simulation["obs_collection_type"]
            self.one_obs_per_wobble = self.obs_collection_type == 'one_obs_per_wobble'
            self.n_run = self.cfg_simulation["n_run"]
            self.livetime_simu = self.cfg_simulation["livetime"]

            #  Wobbles
            self.n_wobbles = len(self.cfg_wobbles)
            self.wobble_seeds = []
            self.wobble_runs = []
            self.wobble_pointings = []
            self.wobble_run_info = []
            
            for i in range(self.n_wobbles):
                cfg_wobble = self.cfg_wobbles[f"wobble_{i+1}"]
                self.wobble_runs.append(cfg_wobble["run"])
                self.wobble_seeds.append(cfg_wobble["seed"])
                wobble_livetime,wobble_pointing,wobble_file_name = get_run_info(self.path_data,self.obs_pattern,self.wobble_runs[-1]) 
                self.wobble_pointings.append(wobble_pointing)
                # The true livetime is retrieved by get_run_info in case you want to implement a very realistic simulation pipeline with a simulation for each true observation you have
                # Here we use the same info for every run, which is the simulation livetime
                # The get_run_info method is mostly use to have realistic pointings, but you can decide the values yourself by changing the next lines
                self.wobble_run_info.append([self.loc, self.source_pos, self.wobble_runs[-1], self.livetime_simu*(self.n_run if self.one_obs_per_wobble else 1), wobble_pointing, wobble_file_name])
            
            self.livetime_per_wobble = self.n_run * self.livetime_simu*u.s
            self.tot_livetime_simu = self.n_wobbles * self.livetime_per_wobble

            print(f"Total simulated livetime: {self.tot_livetime_simu.to(u.h):.1f}")
    
        self.multiple_simulation_subdir = False # TO-DO adapt for multiple subdirectories
        self.save_path_simu_joined = ''
        if self.multiple_simulation_subdir: self.save_path = self.save_path_simu_joined
        elif not self.real_data: self.save_path = f"{self.simulated_obs_dir}"
        else: self.save_path = self.cfg_paths['path_data']
    
    def get_bkg_true_irf(self, downsample=True, downsample_only=True, plot=False, verbose=False):        
        bkg_true_rates = evaluate_bkg(self.bkg_dim, self.bkg_true_model, self.energy_axis_irf, self.offset_axis_irf)
        bkg_true_irf = get_bkg_irf(bkg_true_rates, 2*self.nbin_offset_irf, self.energy_axis_irf, self.offset_axis_irf, self.bkg_dim, FoV_alignment=self.fov_alignment)
        
        if not downsample: 
            return bkg_true_irf
        else:
            bkg_true_map = get_irf_map(bkg_true_rates,[self.energy_axis_irf,self.offset_axis_irf],self.n_run*self.livetime_simu)
            _, bkg_true_down_irf = get_cut_downsampled_irf_from_map(bkg_true_map,[self.energy_axis_acceptance,self.offset_axis_acceptance], [self.offset_factor, self.down_factor], self.bkg_dim, self.n_run * self.livetime_simu, plot=plot, verbose=verbose, FoV_alignment=self.fov_alignment)
            if downsample_only: 
                return bkg_true_down_irf
            else: 
                return bkg_true_irf, bkg_true_down_irf
    
    def load_true_background_irfs(self, config_path=None) -> None:
        if config_path is not None: self.init_config(config_path)
        
        if self.true_collection:
            self.bkg_true_irf_collection = {}
            self.bkg_true_down_irf_collection = {}
            self.bkg_true_map_collection = {}
            self.bkg_true_down_map_collection = {}
            lat_grad_values = []
            lon_grad_values = []
            tmp_config = self.config
            n_run_per_wobble = self.cfg_simulation["n_run"]
            n_third = n_run_per_wobble // 3
            for i_step in range(n_run_per_wobble):
                if  self.cfg_simulation["linear_grad_evolution"]: n_step = i_step / n_run_per_wobble
                else:
                    if i_step < n_third: n_step = 0
                    else: n_step = 0.5 if i_step < 2*n_third else 1
                if self.lon_grad !=0:
                    self.lon_grad_step = np.diff(np.linspace(0,abs(self.lon_grad), round(n_run_per_wobble)))[0]
                    # lon_grad_new = np.round(n_step * n_run_per_wobble * self.lon_grad_step,1)
                    lon_grad_new = n_step * n_run_per_wobble * self.lon_grad_step
                    lon_grad_values.append(lon_grad_new)
                    for param in tmp_config['background']['custom_source']['spatial']['parameters']:
                        if param['name'] == 'lon_grad':
                            param['value'] = lon_grad_new
                if self.lat_grad !=0:
                    self.lat_grad_step = np.diff(np.linspace(0,abs(self.lat_grad), round(n_run_per_wobble)))[0]
                    # lat_grad_new = np.round(n_step * n_run_per_wobble * self.lat_grad_step,1)
                    lat_grad_new = n_step * n_run_per_wobble * self.lat_grad_step
                    lat_grad_values.append(lat_grad_new)
                    for param in tmp_config['background']['custom_source']['spatial']['parameters']:
                        if param['name'] == 'lat_grad':
                            param['value'] = lat_grad_new

                if (i_step == 0) or  (i_step == n_run_per_wobble-1): plot,verbose=(False,True)
                else: plot,verbose=(False,False)

                bkg_true_irf, bkg_true_down_irf, bkg_true_map, bkg_true_down_map = get_bkg_true_irf_from_config(tmp_config,downsample=True,downsample_only=False,plot=plot,verbose=verbose, return_map=True)
                for i_wobble in range(len(self.wobble_pointings)):
                    self.bkg_true_irf_collection[i_step + i_wobble*n_run_per_wobble] = bkg_true_irf
                    self.bkg_true_down_irf_collection[i_step + i_wobble*n_run_per_wobble] = bkg_true_down_irf
                    self.bkg_true_map_collection[i_step + i_wobble*n_run_per_wobble] = bkg_true_map
                    self.bkg_true_down_map_collection[i_step + i_wobble*n_run_per_wobble] = bkg_true_down_map
            self.lon_lat_grad_values = [np.array(lon_grad_values), np.array(lat_grad_values)]
        else:
            self.bkg_true_irf, self.bkg_true_down_irf, self.bkg_true_map, self.bkg_true_down_map = get_bkg_true_irf_from_config(self.config,downsample=True,downsample_only=False,plot=False,verbose=True, return_map=True)
            bkg_spatial_model_params = self.config['background']['custom_source']['spatial']['parameters']
            lon_lat_grad_values = []
            for param in bkg_spatial_model_params:
                if (param['name'] == 'lon_grad') or (param['name'] == 'lat_grad'): lon_lat_grad_values.append(param['value'])
            self.lon_lat_grad_values = np.array(lon_lat_grad_values)
            
    def get_background_irf(self, type='true', downsampled=True, i_irf=0) -> BackgroundIRF:
        if type=='true':
            if self.true_collection:
                if downsampled: return self.bkg_true_down_irf_collection[i_irf]
                else: return self.bkg_true_irf_collection[i_irf]
            else:
                if downsampled: return self.bkg_true_down_irf
                else: return self.bkg_true_irf
        elif type=='output':
            if self.out_collection: return self.bkg_output_irf_collection[i_irf]
            else: return self.bkg_output_irf
    
    def get_2d_binned_stat_from_skymap(self, skymap, energy_offset_axes = [None,None]):
        center_map = skymap._geom.center_skydir
        energy_axis, offset_axis = energy_offset_axes
        if energy_axis is None: energy_axis = self.energy_axis_acceptance
        if offset_axis is None: offset_axis = self.offset_axis_acceptance
        data = np.zeros((energy_axis.nbin, offset_axis.nbin))
        for i in range(offset_axis.nbin):
            if np.isclose(0. * u.deg, offset_axis.edges[i]):
                selection_region = CircleSkyRegion(center=center_map, radius=offset_axis.edges[i + 1])
            else:
                selection_region = CircleAnnulusSkyRegion(center=center_map,
                                                            inner_radius=offset_axis.edges[i],
                                                            outer_radius=offset_axis.edges[i + 1])
            selection_map = skymap.geom.to_image().region_mask([selection_region])
            for j in range(energy_axis.nbin):
                value = u.dimensionless_unscaled * np.sum(skymap.data[j, :, :] * selection_map)
                data[j, i] = value
        return data

    def get_dfbkg_stat(self, i_irf_arr=None, bkg_map_arr=None, return_df=True, by_grad=False, n_run=1):
        """
        Compute background statistics per energy bin, optionally by gradient.
        
        Parameters
        ----------
        bkg_map_arr : np.ndarray, optional
            Precomputed array of background maps (IRF x Ebin x pixels). If None, it is built from self.bkg_true_down_map_collection.
        i_irf_arr : list or np.ndarray, optional
            Indices of IRFs to include. If None, use all.
        return_df : bool
            Whether to return the DataFrame.
        by_grad : bool
            If True, compute statistics per spatial gradient.
        """

        # Prepare bkg_map_arr
        if bkg_map_arr is None:
            try: bkg_map_collection = deepcopy(self.bkg_true_down_map_collection)
            except:
                bkg_map_collection = [deepcopy(self.bkg_true_down_map)]
                bkg_map_collection[0].data *= n_run
            i_map_arr = np.arange(len(bkg_map_collection)) if i_irf_arr is None else i_irf_arr
            if self.bkg_dim == 3:
                bkg_map_arr = [bkg_map_collection[i].data for i in i_map_arr]
            else:
                bkg_map_arr = []
                for i in i_map_arr: bkg_map_arr.append(self.get_2d_binned_stat_from_skymap(bkg_map_collection[i]))
        else:
            i_map_arr = np.arange(bkg_map_arr.shape[0]) if i_irf_arr is None else i_irf_arr
            bkg_map_arr = bkg_map_arr[i_map_arr]
        
        i_map_arr = np.array(i_map_arr, dtype=int)
        bkg_map_arr = np.array(bkg_map_arr)
        
        try:
            n_irf, n_Ebin, n_pixel_i, n_pixel_j = bkg_map_arr.shape
            n_pixel = n_pixel_i * n_pixel_j
        except:
            n_irf, n_Ebin, n_pixel = bkg_map_arr.shape

        interval_index = pd.IntervalIndex.from_breaks(
            self.energy_axis_acceptance.edges.to_value(u.TeV).round(4)
        )

        # Prepare gradient values if needed
        if by_grad:
            lon_grad_values_all = pd.Series(np.concatenate((np.array(self.lon_lat_grad_values[0]),np.array(self.lon_lat_grad_values[0]))).round(1))
            lat_grad_values_all = pd.Series(np.concatenate((np.array(self.lon_lat_grad_values[1]),np.array(self.lon_lat_grad_values[1]))).round(1))
            # Select only the IRFs requested
            lon_grad_values = lon_grad_values_all[lon_grad_values_all.index.isin(i_map_arr)].to_numpy()
            lat_grad_values = lat_grad_values_all[lat_grad_values_all.index.isin(i_map_arr)].to_numpy()

            cfg_grad = self.cfg_background['spatial_model_gradient']
            values = [[], lat_grad_values, lon_grad_values, np.concatenate((lon_grad_values, lat_grad_values))]
            idx_grad = int(cfg_grad["lon_grad_variation"]) * 2 + int(cfg_grad["lat_grad_variation"])
            grad_values_unique = np.unique(values[idx_grad])

            # Build MultiIndex
            index = pd.MultiIndex.from_product(
                [interval_index, grad_values_unique, np.arange(n_pixel)],
                names=["Ebin", "grad", "i_pixel"]
            )
        else:
            index = pd.MultiIndex.from_product(
                [interval_index, np.arange(n_pixel)],
                names=["Ebin", "i_pixel"]
            )

        # Initialize DataFrame
        dfbkg_stat = pd.DataFrame(index=index, columns=["count"])
        dfbkg_stat["count"] = 0.0

        # Build per-IRF DataFrames and concatenate
        df_list = []
        for idx_map, i_irf in enumerate(i_map_arr):
            data_map = bkg_map_arr[idx_map]

            if by_grad:
                # Per IRF, select corresponding grad value
                grad_val = values[idx_grad][idx_map]
                idxs = pd.MultiIndex.from_product(
                    [interval_index, [grad_val], np.arange(n_pixel)],
                    names=["Ebin", "grad", "i_pixel"]
                )
            else:
                idxs = pd.MultiIndex.from_product(
                    [interval_index, np.arange(n_pixel)],
                    names=["Ebin", "i_pixel"]
                )

            df_tmp = pd.DataFrame(
                data = data_map.reshape(n_Ebin * n_pixel),
                index = idxs,
                columns=[f"count_{i_irf}"]
            )

            df_list.append(df_tmp)

        # Concatenate all IRF columns at once
        dfbkg_stat = pd.concat([dfbkg_stat] + df_list, axis=1)

        # Compute total count across IRFs
        dfbkg_stat["count"] = dfbkg_stat[[col for col in dfbkg_stat.columns if col.startswith("count_")]].sum(axis=1)

        self.dfbkg_stat = deepcopy(dfbkg_stat)

        if by_grad:
            display(dfbkg_stat.groupby(by=["Ebin","grad"]).describe()["count"].T)
        else:
            display(dfbkg_stat.groupby(by=["Ebin"]).describe()["count"].T)

        if return_df:
            return dfbkg_stat
    
    def load_observation_collection(self, config_path=None, from_index=False, verbose=True) -> None:
        if config_path is not None: self.init_config(config_path)
        
        if not self.multiple_simulation_subdir:
            if from_index: self.pattern = self.index_suffix
            elif not self.real_data: self.pattern = f"obs_*.fits"
            else: self.pattern = self.obs_pattern
            print(f"Obs collection loading pattern: {self.pattern}")
            self.data_store, self.obs_collection = get_obs_collection(self.save_path,self.pattern,self.multiple_simulation_subdir,from_index=from_index,with_datastore=True)
            self.obs_table = self.data_store.obs_table
            self.dfobs_table = get_dfobs_table(self.obs_table)
            
            # Save the new data store for future use
            if not from_index:
                if not pathlib.Path(f"{self.save_path}/hdu-index.fits.gz").exists(): 
                    self.data_store.hdu_table.write(f"{self.save_path}/hdu-index.fits.gz",format="fits")
                if not pathlib.Path(f"{self.save_path}/obs-index.fits.gz").exists(): 
                    self.data_store.obs_table.write(f"{self.save_path}/obs-index.fits.gz",format="fits")

            self.all_sources = np.unique(self.dfobs_table["OBJECT"].to_numpy())
            self.all_obs_ids = self.dfobs_table.index.to_numpy()
            if verbose:
                print("Available sources: ", self.all_sources)
                print(f"{len(self.all_obs_ids)} available runs: ",self.all_obs_ids)
            
            all_obs_ids = self.dfobs_table.index.to_numpy()
            sources = self.dfobs_table.OBJECT.str.lower().str.replace(" ", "").to_numpy()
            is_source = [self.source_name.lower().replace(" ", "") in source for source in sources]

            if (len(self.all_sources) > 1) and (not from_index):
                self.all_obs_ids = all_obs_ids[is_source]
                if (self.run_list.shape[0] == 0): self.run_list = self.all_obs_ids
                if verbose: print(f"{len(self.all_obs_ids)} available runs for source {self.source_name}: ",self.all_obs_ids)
            else:
                if verbose: print(f"All runs pointing on source {self.source_name}")
            
            self.obs_ids = self.all_obs_ids if (self.run_list.shape[0] == 0) else self.run_list
            
            if (len(self.all_sources) > 1) | (self.run_list.shape[0] != 0): 
                self.data_store, self.obs_collection = get_obs_collection(self.save_path,self.pattern,self.multiple_simulation_subdir,from_index=from_index,with_datastore=True,obs_ids=self.obs_ids)
                self.obs_table = self.data_store.obs_table[is_source]
                if verbose: print(f"{len(self.obs_ids)} selected runs: ",self.obs_ids)
                self.dfobs_table = get_dfobs_table(self.obs_table)
                self.dfobs_table = self.dfobs_table.loc[self.obs_ids]
            else: 
                if verbose: print("All runs selected")
            
            if self.real_data:
                # Add telescope position to observations
                for iobs in range(len(self.obs_collection)):
                    if self.obs_collection[iobs]._meta is None:
                        meta_dict = self.obs_collection[iobs].events.table.meta
                        if ('CTA' in meta_dict['TELESCOP']) or ('LST-1' in meta_dict['TELESCOP']) or ('MAGIC' in meta_dict['TELESCOP']):
                            self.loc = EarthLocation.of_site('Roque de los Muchachos')
                        elif ('HESS' in meta_dict['TELESCOP']) or ('H.E.S.S' in meta_dict['TELESCOP']):
                            self.loc = HESS
                        elif ('VERITAS' in meta_dict['TELESCOP']):
                            self.loc = VERITAS
                        elif ('HAWC' in meta_dict['TELESCOP']):
                            self.loc = HAWC
                        elif ('LHAASO' in meta_dict['TELESCOP']):
                            self.loc = LHAASO
                        elif ('AUGER' in meta_dict['TELESCOP']):
                            self.loc = AUGER

                        meta_dict.__setitem__('GEOLON',str(self.loc.lon.value))
                        meta_dict.__setitem__('GEOLAT',str(self.loc.lat.value))
                        meta_dict.__setitem__('GEOALT',str(self.loc.height.to_value(u.m)))
                        meta_dict.__setitem__('deadtime_fraction',str(1-meta_dict['DEADC']))
                        
                        self.obs_collection[iobs]._meta = ObservationMetaData.from_header(meta_dict)
                        # Sometimes the assignation does not work
                        if self.obs_collection[iobs]._meta is None:
                            self.obs_collection[iobs].events.meta = meta_dict
                        self.obs_collection[iobs]._location = self.loc
                        self.obs_collection[iobs].pointing._location = self.loc
                        # self.obs_collection[iobs].obs_info['observatory_earth_location'] = self.loc # <- modifié pour être accessible à l'intérieur de la méthode qui récupère le pointé
            self.dfobs_table["WOBBLE"] = get_unique_wobble_pointings(self.obs_collection)
            self.wobble_mean_pointings = []
            for wobble in self.dfobs_table.WOBBLE.unique():
                mean_radec = self.dfobs_table[self.dfobs_table.WOBBLE == wobble][["RA_PNT","DEC_PNT"]].mean()
                self.wobble_mean_pointings.append(np.array([float(mean_radec["RA_PNT"]), float(mean_radec["DEC_PNT"])]))

            if 'FILE_NAME' not in self.dfobs_table:
                dfhdu_table = self.data_store.hdu_table.to_pandas().set_index('OBS_ID')[['HDU_TYPE', 'FILE_NAME']]
                for col in ['HDU_TYPE', 'FILE_NAME']: 
                        if col in dfhdu_table.columns:
                            if (dfhdu_table[col].apply(lambda x: isinstance(x, bytes)).all()): dfhdu_table[col] = dfhdu_table[col].str.decode('utf-8')
                dfhdu_table = dfhdu_table[~dfhdu_table.FILE_NAME.duplicated() & (dfhdu_table.HDU_TYPE == 'events')]
                self.dfobs_table['FILE_NAME'] = self.path_data + '/' + dfhdu_table.loc[dfhdu_table.index.isin(self.dfobs_table.index), 'FILE_NAME']
        else:
            self.pattern = f"{self.obs_collection_type}_{self.save_name_suffix[:-8]}*/obs_*{self.save_name_obs}.fits" # Change pattern according to your sub directories
            self.obs_collection = get_obs_collection(self.simulated_obs_dir,self.pattern,self.multiple_simulation_subdir,with_datastore=False)
            self.all_obs_ids = np.arange(1,len(self.obs_collection)+1,1)
        self.n_run = len(self.obs_collection)
        self.total_livetime = sum([obs.observation_live_time_duration for obs in self.obs_collection])
        if not isinstance(self.total_livetime, u.Quantity): self.total_livetime *= u.s
        if verbose==True: print(f"Total livetime: {self.total_livetime.to(u.h):.1f}")