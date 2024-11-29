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
from gammapy.data import FixedPointingInfo, Observation, observatory_locations, PointingMode
from gammapy.datasets import MapDataset,MapDatasetEventSampler,Datasets,MapDatasetOnOff
from gammapy.irf import load_irf_dict_from_file, Background2D, Background3D, FoVAlignment
from gammapy.makers import FoVBackgroundMaker,MapDatasetMaker, SafeMaskMaker, RingBackgroundMaker
from gammapy.maps import MapAxis, WcsGeom, WcsNDMap, Map
from regions import CircleAnnulusSkyRegion, CircleSkyRegion, EllipseSkyRegion
from gammapy.maps.region.geom import RegionGeom
from gammapy.estimators import ExcessMapEstimator
from gammapy.datasets import Datasets, MapDataset, MapDatasetOnOff
from astropy.visualization.wcsaxes import SphericalCircle

from gammapy.data import DataStore

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
from acceptance_modelisation import RadialAcceptanceMapCreator, Grid3DAcceptanceMapCreator, BackgroundCollectionZenith
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union
from gammapy.datasets import MapDatasetEventSampler
from gammapy.irf import Background3D, BackgroundIRF
from gammapy.makers.utils import make_map_background_irf

fov_rotation_time_step = 100000 * u.s


#----
# Save/retrieve pickle data
def any2pickle(save_path:str,data):
    '''Save data in pickle format: no info loss'''
    with open(save_path, 'wb') as f1:
        pk.dump(data, f1)
    print('Pickle file saved at '+ save_path)


def pickle2any(save_path:str,datatype='dataframe'):
    '''Read pickle data'''
    path = pathlib.Path(save_path)
    if path.exists():
        if datatype=='dataframe': return pd.read_pickle(path)
        else:
            with open(os.fspath(path), 'rb') as f1:
                dataframe = pk.load(f1)
                return dataframe
    else: 
        print('Error : No pickle file found at '+ os.fspath(path))
        return path.exists()

#----

def compute_sigma_eff(lon_0, lat_0, lon, lat, phi, major_axis, e):
    """Effective radius, used for the evaluation of elongated models."""
    phi_0 = position_angle(lon_0, lat_0, lon, lat)
    d_phi = phi - phi_0
    minor_axis = Angle(major_axis * np.sqrt(1 - e**2))

    a2 = (major_axis * np.sin(d_phi)) ** 2
    b2 = (minor_axis * np.cos(d_phi)) ** 2
    denominator = np.sqrt(a2 + b2)
    sigma_eff = major_axis * minor_axis / denominator
    return minor_axis, sigma_eff

class GaussianSpatialModel_LinearGradient(SpatialModel):
    r"""Two-dimensional Gaussian model.

    For more information see :ref:`gaussian-spatial-model`.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position.
        Default is "0 deg", "0 deg".
    lon_grad, lat_grad : `~astropy.coordinates.Angle`
        Gradient applied to each set of coordinates.
        Default is "0 deg", "0 deg".
    sigma : `~astropy.coordinates.Angle`
        Length of the major semiaxis of the Gaussian, in angular units.
        Default is 1 deg.
    e : `float`
        Eccentricity of the Gaussian (:math:`0< e< 1`).
        Default is 0.
    phi : `~astropy.coordinates.Angle`
        Rotation angle :math:`\phi`: of the major semiaxis.
        Increases counter-clockwise from the North direction.
        Default is 0 deg.
    frame : {"icrs", "galactic"}
        Center position coordinate frame.
    """

    tag = ["GaussianSpatialModel_LinearGradient", "gauss_grad"]

    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)

    lon_grad = Parameter("lon_grad", "0 deg-1")
    lat_grad = Parameter("lat_grad", "0 deg-1", min=-90, max=90)

    sigma = Parameter("sigma", "1 deg", min=0)
    e = Parameter("e", 0, min=0, max=1, frozen=True)
    phi = Parameter("phi", "0 deg", frozen=True)

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, sigma, e, phi, lon_grad, lat_grad):
        """Evaluate model."""
        sep = angular_separation(lon, lat, lon_0, lat_0)

        if e == 0:
            a = 1.0 - np.cos(sigma)
            norm = (1 / (4 * np.pi * a * (1.0 - np.exp(-1.0 / a)))).value
        else:
            minor_axis, sigma_eff = compute_sigma_eff(
                lon_0, lat_0, lon, lat, phi, sigma, e
            )
            a = 1.0 - np.cos(sigma_eff)
            norm =  (1 / (2 * np.pi * sigma * minor_axis)).to_value("sr-1")
        
        lon_grad_factor = (lon * lon_grad).to_value("")
        # if (lon.to_value(u.deg) > 0).any(): lon_grad_factor[np.where(lon.to_value(u.deg) > 0)] = -1
        if (lon_grad_factor < -1).any(): lon_grad_factor[np.where(lon_grad_factor < -1)] = -1

        lat_grad_factor = (lat * lat_grad).to_value("")
        if (lat_grad_factor < -1).any(): lat_grad_factor[np.where(lat_grad_factor < -1)] = -1

        norm *=  (1 + lon_grad_factor) * (1 +lat_grad_factor ) 
        exponent = -0.5 * ((1 - np.cos(sep)) / a)
        return u.Quantity(norm * np.exp(exponent).value, "sr-1", copy=COPY_IF_NEEDED)
    
class GaussianSpatialModel_LinearGradient_half(SpatialModel):
    r"""Two-dimensional Gaussian model.

    For more information see :ref:`gaussian-spatial-model`.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position.
        Default is "0 deg", "0 deg".
    lon_grad, lat_grad : `~astropy.coordinates.Angle`
        Gradient applied to each set of coordinates.
        Default is "0 deg", "0 deg".
    sigma : `~astropy.coordinates.Angle`
        Length of the major semiaxis of the Gaussian, in angular units.
        Default is 1 deg.
    e : `float`
        Eccentricity of the Gaussian (:math:`0< e< 1`).
        Default is 0.
    phi : `~astropy.coordinates.Angle`
        Rotation angle :math:`\phi`: of the major semiaxis.
        Increases counter-clockwise from the North direction.
        Default is 0 deg.
    frame : {"icrs", "galactic"}
        Center position coordinate frame.
    """

    tag = ["GaussianSpatialModel_LinearGradient_half", "gauss_half"]

    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)

    lon_grad = Parameter("lon_grad", "0 deg-1")
    lat_grad = Parameter("lat_grad", "0 deg-1", min=-90, max=90)

    sigma = Parameter("sigma", "1 deg", min=0)
    e = Parameter("e", 0, min=0, max=1, frozen=True)
    phi = Parameter("phi", "0 deg", frozen=True)

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, sigma, e, phi, lon_grad, lat_grad):
        """Evaluate model."""
        sep = angular_separation(lon, lat, lon_0, lat_0)

        if e == 0:
            a = 1.0 - np.cos(sigma)
            norm = (1 / (4 * np.pi * a * (1.0 - np.exp(-1.0 / a)))).value
        else:
            minor_axis, sigma_eff = compute_sigma_eff(
                lon_0, lat_0, lon, lat, phi, sigma, e
            )
            a = 1.0 - np.cos(sigma_eff)
            norm =  (1 / (2 * np.pi * sigma * minor_axis)).to_value("sr-1")
        
        lon_grad_factor = (lon * lon_grad).to_value("")
        if (lon.to_value(u.deg) > 0).any(): lon_grad_factor[np.where(lon.to_value(u.deg) > 0)] = -1
        if (lon_grad_factor < -1).any(): lon_grad_factor[np.where(lon_grad_factor < -1)] = -1

        lat_grad_factor = (lat * lat_grad).to_value("")
        if (lat_grad_factor < -1).any(): lat_grad_factor[np.where(lat_grad_factor < -1)] = -1

        norm *=  (1 + lon_grad_factor) * (1 +lat_grad_factor ) 
        exponent = -0.5 * ((1 - np.cos(sep)) / a)
        return u.Quantity(norm * np.exp(exponent).value, "sr-1", copy=COPY_IF_NEEDED)

def get_data_store(dir_path, pattern, from_index=False):
    if from_index:
        data_store = DataStore.from_dir(f"{dir_path}",hdu_table_filename=f'hdu-index{pattern}.fits.gz',obs_table_filename=f'obs-index{pattern}.fits.gz')
    else:
        path = Path(dir_path)
        paths = sorted(list(path.rglob(pattern)))
        data_store = DataStore.from_events_files(paths)
    
    return data_store

def get_obs_collection(dir_path, pattern, multiple_simulation_subdir=False,from_index=False,with_datastore=True):
    if not multiple_simulation_subdir:
        data_store = get_data_store(dir_path, pattern, from_index)
        obs_collection = data_store.get_observations(required_irf='all-optional')
    else:
        path = Path(dir_path)
        paths = sorted(list(path.rglob(pattern)))
        obs_collection = Observations()
        for path in paths:
            obs_collection.append(Observation.read(path))
        for iobs in range(len(obs_collection)):
            obs_id=iobs+1
            obs_collection[iobs].events.table.meta['OBS_ID'] = obs_id
            obs_collection[iobs].obs_id = obs_id
    
    if with_datastore:
        return data_store, obs_collection
    else: 
        return obs_collection
    
def get_run_info(path_data:str, pattern:str, obs_id:int):
    """returns array with [livetime,pointing_radec,file_name]"""
    loc = EarthLocation.of_site('Roque de los Muchachos')
    data_store = get_data_store(path_data, pattern)
    obs_table = data_store.obs_table
    livetime = obs_table[obs_table["OBS_ID"] == obs_id]["LIVETIME"].data[0]
    ra = obs_table[obs_table["OBS_ID"] == obs_id]["RA_PNT"].data[0]
    dec = obs_table[obs_table["OBS_ID"] == obs_id]["DEC_PNT"].data[0]
    pointing= SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
    print(f"--Run {obs_id}--\nlivetime: {livetime}\npointing radec: {pointing}")
    hdu_table = data_store.hdu_table
    file_name = hdu_table[hdu_table["OBS_ID"]==obs_id]["FILE_NAME"][0]
    return livetime, pointing, file_name

# +
# Les functions modifiées (pas proprement)
def get_empty_obs_simu(Bkg_irf, axis_info, run_info, src_models, path_data:str,flux_to_0=True, t_ref_str="2000-01-01 00:00:00", t_delay=0,verbose=False):
    '''Loads irf from file and return a simulated observation with its associated dataset'''

    loc, source_pos, run, livetime, pointing, file_name = run_info
    # print(livetime)
    
    if axis_info is not None: 
        e_min, e_max, offset_max, nbin_offset= axis_info
        energy_axis = MapAxis.from_energy_bounds(e_min, e_max, nbin=10, name='energy')
        nbins_map = 2*nbin_offset
    else:
        energy_axis = Bkg_irf.axes[0]
        e_min = energy_axis.edges.min()
        e_max = energy_axis.edges.max()
        offset_axis = Bkg_irf.axes[1]
        offset_max = offset_axis.edges.max()
        nbins_map = offset_axis.nbin
    
    binsize = offset_max / (nbins_map / 2)

    # Loading IRFs
    irfs = load_irf_dict_from_file(
        f"{path_data}/{file_name}"
    )
    if Bkg_irf is not None: irfs['bkg'] = Bkg_irf
    if verbose: [print(irfs[irf]) for irf in irfs]

    if not isinstance(livetime, u.Quantity): livetime*=u.s

    pointing_info = FixedPointingInfo(mode=PointingMode.POINTING, fixed_icrs=pointing)#,legacy_altaz=pointing_altaz)
    t_ref=Time(t_ref_str)
    delay=t_delay*u.s
    # print(delay)
    obs = Observation.create(
        pointing=pointing_info, livetime=livetime,irfs=irfs, location=loc, obs_id='0',reference_time=t_ref,
		tstart=delay,
		tstop=delay + livetime,
    )

    obs._location = loc
    obs._pointing = pointing_info
    obs.pointing._location = loc
    obs.aeff.meta["TELESCOP"] = "CTA_NORTH"
    obs.aeff.meta['INSTRUME'] = 'Northern Array'

    return obs

def get_empty_dataset_and_obs_simu(Bkg_irf, axis_info, run_info, src_models, path_data:str,flux_to_0=True, t_ref_str="2000-01-01 00:00:00", t_delay=0,verbose=False):
    '''Loads irf from file and return a simulated observation with its associated dataset'''

    loc, source_pos, run, livetime, pointing, file_name = run_info
    # print(livetime)
    
    if axis_info is not None: 
        e_min, e_max, offset_max, nbin_offset= axis_info
        energy_axis = MapAxis.from_energy_bounds(e_min, e_max, nbin=10, name='energy')
        nbins_map = 2*nbin_offset
    else:
        energy_axis = Bkg_irf.axes[0]
        e_min = energy_axis.edges.min()
        e_max = energy_axis.edges.max()
        offset_axis = Bkg_irf.axes[1]
        offset_max = offset_axis.edges.max()
        nbins_map = offset_axis.nbin
    
    binsize = offset_max / (nbins_map / 2)

    # Loading IRFs
    irfs = load_irf_dict_from_file(
        f"{path_data}/{file_name}"
    )

    if Bkg_irf is not None: irfs['bkg'] = Bkg_irf
    if verbose: [print(irfs[irf]) for irf in irfs]

    if not isinstance(livetime, u.Quantity): livetime*=u.s

    pointing_info = FixedPointingInfo(mode=PointingMode.POINTING, fixed_icrs=pointing)#,legacy_altaz=pointing_altaz)
    t_ref=Time(t_ref_str)
    delay=t_delay*u.s
    # print(delay)
    obs = Observation.create(
        pointing=pointing_info, livetime=livetime, irfs=irfs, location=loc, obs_id='0',reference_time=t_ref,
		tstart=delay,
		tstop=delay + livetime,
    )

    obs._location = loc
    obs._pointing = pointing_info
    obs.pointing._location = loc
    obs.aeff.meta["TELESCOP"] = "CTA_NORTH"
    obs.aeff.meta['INSTRUME'] = 'Northern Array'

    if verbose: print(obs.obs_info)

    edisp_frac = 0.3
    e_min_true,e_max_true = ((1-edisp_frac)*e_min , (1+edisp_frac)*e_max)
    nbin_energy_map_true = 20

    energy_axis_true = MapAxis.from_energy_bounds(
        e_min_true, e_max_true, nbin=nbin_energy_map_true, per_decade=True, unit="TeV", name="energy_true"
    )

    migra_axis = MapAxis.from_bounds(-1, 2, nbin=150, node_type="edges", name="migra")

    geom = WcsGeom.create(
            skydir=(pointing.ra.degree,pointing.dec.degree),
            binsz=binsize,
            npix=(nbins_map, nbins_map),
            frame="icrs",
            proj="CAR",
            axes=[energy_axis],
        )
    if verbose: print(geom)

    empty = MapDataset.create(
        geom,
        energy_axis_true=energy_axis_true,
        migra_axis=migra_axis,
        name="my-dataset",
    )

    maker = MapDatasetMaker(selection=["exposure", "background", "edisp"], fov_rotation_error_limit=1 * u.deg)
    dataset = maker.run(empty, obs)
    if verbose: print(obs.obs_info)
    if verbose: print(dataset)

    # Models
    spatial_model = src_models.spatial_model()

    # Uncomment if you want to use the true source spectral model. Here we declare a model with an amplitude of 0 cm-2 s-1 TeV-1 to simulate only background
    if not flux_to_0: spectral_model = src_models.spectral_model()
    else: spectral_model = LogParabolaSpectralModel(
            alpha=2,beta=0.2 / np.log(10), amplitude=0. * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
    )

    model_simu = SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        name="model-simu",
    )

    # Here I have a doubt  about why I have to declare another model,
    # but if I try to use our model, or no model at all, it doesn't work anymore
    # In any case, I confirmed it is simulating according to the IRF model
    bkg_model = FoVBackgroundModel(dataset_name="my-dataset")
    #bkg_model.spectral_model.norm.value = 0.01
    models = Models([model_simu,bkg_model])
    dataset.models = models
    if verbose: print(dataset.models)

    return dataset,obs

def stack_with_meta(eventlist1, eventlist2):
    meta1 = deepcopy(eventlist1.table.meta)
    meta2 = deepcopy(eventlist2.table.meta)
    eventlist1.stack(eventlist2)
    eventlist1.table.meta['TSTART'] = np.min([meta1['TSTART'], meta2['TSTART']])
    eventlist1.table.meta['TSTOP'] = np.max([meta1['TSTOP'], meta2['TSTOP']])
    eventlist1.table.meta['TELAPSED'] = eventlist1.table.meta['TSTOP']- eventlist1.table.meta['TSTART']
    eventlist1.table.meta['ONTIME'] = meta1['ONTIME'] + meta2['ONTIME']
    eventlist1.table.meta['LIVETIME'] = meta1['LIVETIME'] + meta2['LIVETIME']
    return eventlist1

# -

def evaluate_bkg(bkg_dim,bkg_model,energy_axis,offset_axis):
    ecenters = energy_axis.center.to_value(u.TeV)
    centers = offset_axis.center.to_value(u.deg)

    if bkg_dim==2:
        E,offset = np.meshgrid(ecenters, centers, indexing='ij')
        return bkg_model.evaluate(E*u.TeV,offset*u.deg,offset*u.deg)
    elif bkg_dim == 3:
        centers = np.concatenate((-np.flip(centers), centers), axis=None)
        E, y, x = np.meshgrid(ecenters, centers, centers, indexing='ij')
        return bkg_model.evaluate(E*u.TeV,x*u.deg,y*u.deg)

def get_bkg_irf(bkg_map, n_bins_map, energy_axis, offset_axis, bkg_dim, livetime=1*u.s, stacked=False, FoV_alignment='ALTAZ'):
    '''Transform a background map into a background IRF'''
    if not isinstance(livetime, u.Quantity): livetime*=u.s
    if FoV_alignment=='ALTAZ': fov_alignment = FoVAlignment.ALTAZ
    elif FoV_alignment=='RADEC': fov_alignment = FoVAlignment.RADEC
    
    is_WcsNDMap = isinstance(bkg_map, WcsNDMap)
    oversample_map=1
    spatial_resolution = np.min(
            np.abs(offset_axis.edges[1:] - offset_axis.edges[:-1])) / oversample_map
    max_offset = np.max(offset_axis.edges)
    if n_bins_map is None: n_bins_map = 2 * int(np.rint((max_offset / spatial_resolution).to(u.dimensionless_unscaled)))
    spatial_bin_size = max_offset / (n_bins_map / 2)
    center_map = SkyCoord(ra=0. * u.deg, dec=0. * u.deg, frame='icrs')
    geom_irf = WcsGeom.create(skydir=center_map, npix=(n_bins_map, n_bins_map),
                                binsz=spatial_bin_size, frame="icrs", axes=[energy_axis])      

    if bkg_dim==2:
        if is_WcsNDMap:
            data_background = np.zeros((energy_axis.nbin, offset_axis.nbin)) * u.Unit('s-1 MeV-1 sr-1')
            down_factor = bkg_map.data.shape[1]/data_background.shape[1]
            bkg_map = bkg_map.downsample(factor=down_factor, preserve_counts=True).data
            for i in range(offset_axis.nbin):
                if np.isclose(0. * u.deg, offset_axis.edges[i]):
                    selection_region = CircleSkyRegion(center=center_map, radius=offset_axis.edges[i + 1])
                else:
                    selection_region = CircleAnnulusSkyRegion(center=center_map,
                                                                inner_radius=offset_axis.edges[i],
                                                                outer_radius=offset_axis.edges[i + 1])
                selection_map = geom_irf.to_image().to_cube([energy_axis.squash()]).region_mask([selection_region])
                for j in range(energy_axis.nbin):
                    value = u.dimensionless_unscaled * np.sum(bkg_map[j, :, :] * selection_map)

                    value /= (energy_axis.edges[j + 1] - energy_axis.edges[j])
                    value /= 2. * np.pi * (np.cos(offset_axis.edges[i]) - np.cos(offset_axis.edges[i+1])) * u.steradian
                    value /= livetime
                    data_background[j, i] = value
        else: data_background = bkg_map
        bkg_irf = Background2D(axes=[energy_axis, offset_axis],
                                        data=data_background.to(u.Unit('s-1 MeV-1 sr-1')),fov_alignment=fov_alignment)

    elif bkg_dim==3:
        edges = offset_axis.edges
        extended_edges = np.concatenate((-np.flip(edges), edges[1:]), axis=None)
        extended_offset_axis_x = MapAxis.from_edges(extended_edges, name='fov_lon')
        bin_width_x = np.repeat(extended_offset_axis_x.bin_width[:, np.newaxis], n_bins_map, axis=1)
        extended_offset_axis_y = MapAxis.from_edges(extended_edges,  name='fov_lat')
        bin_width_y = np.repeat(extended_offset_axis_y.bin_width[np.newaxis, :], n_bins_map, axis=0)

        if is_WcsNDMap:
            solid_angle = 4. * (np.sin(bin_width_x / 2.) * np.sin(bin_width_y / 2.)) * u.steradian
            down_factor = bkg_map.data.shape[1]/solid_angle.shape[1]
            bkg_map = bkg_map.downsample(factor=down_factor, preserve_counts=True).data

            if not stacked:
                data_background = bkg_map / solid_angle[np.newaxis, :, :] / energy_axis.bin_width[:, np.newaxis,
                                                                                    np.newaxis] / livetime
            else:
                data_background = bkg_map / u.steradian / energy_axis.unit / u.s
        else: data_background = bkg_map
        bkg_irf = Background3D(axes=[energy_axis, extended_offset_axis_x, extended_offset_axis_y],
                                    data=data_background.to(u.Unit('s-1 MeV-1 sr-1')),
                                    fov_alignment=fov_alignment)
    return bkg_irf

def get_irf_map(irf_rates, irf_axes, livetime):
    irf_energy_axis, irf_offset_axis = irf_axes
    max_offset = np.max(irf_offset_axis.edges)
    n_bins_map = irf_rates.shape[1]
    spatial_bin_size = max_offset / (n_bins_map / 2)
    center_map = SkyCoord(ra=0 * u.deg, dec=0 * u.deg, frame='icrs')
    geom = WcsGeom.create(skydir=center_map, npix=(n_bins_map, n_bins_map),
                                binsz=spatial_bin_size, frame="icrs", axes=[irf_energy_axis])
    irf_map = WcsNDMap(data=irf_rates,geom=geom,unit=u.dimensionless_unscaled)

    edges = irf_offset_axis.edges
    extended_edges = np.concatenate((-np.flip(edges), edges[1:]), axis=None)
    extended_offset_axis_x = MapAxis.from_edges(extended_edges, name='fov_lon')
    bin_width_x = np.repeat(extended_offset_axis_x.bin_width[:, np.newaxis], n_bins_map, axis=1)
    extended_offset_axis_y = MapAxis.from_edges(extended_edges,  name='fov_lat')
    bin_width_y = np.repeat(extended_offset_axis_y.bin_width[np.newaxis, :], n_bins_map, axis=0)
    solid_angle = 4. * (np.sin(bin_width_x / 2.) * np.sin(bin_width_y / 2.)) * u.steradian
    irf_map.data = irf_map.data * solid_angle.to_value(u.sr)[np.newaxis, :, :] * irf_energy_axis.bin_width.to_value(u.TeV)[:, np.newaxis, np.newaxis] * livetime 

    return irf_map

def get_cut_downsampled_irf_from_map(irf_map, irf_down_axes, cut_down_factors, bkg_dim, livetime, plot=True, verbose=True, FoV_alignment='ALTAZ'):
    irf_down_energy_axis, irf_down_offset_axis = irf_down_axes
    offset_factor, downsample_factor = cut_down_factors

    if offset_factor > 1: 
        irf_cut_map = irf_map.cutout(position=SkyCoord(ra=0*u.deg,dec=0*u.deg,frame='icrs'),width=2*irf_down_offset_axis.edges.max())
        irf_cut_map.data = irf_cut_map.data*offset_factor*irf_cut_map.data.shape[1]/irf_map.data.shape[1]
        irf_down_map = irf_cut_map.copy()
        
        if plot:
            fig, ax = plt.subplots(figsize = (6,6))
            irf_cut_map.sum_over_axes(['energy']).plot(ax=ax)
    else: irf_down_map = irf_map.copy()

    irf_down_map = irf_down_map.downsample(factor=downsample_factor).downsample(factor=downsample_factor,axis_name='energy')
    # irf_down_map.data = irf_down_map.data/downsample_factor**3

    if plot:
        fig, ax = plt.subplots(figsize = (6,6))
        irf_down_map.sum_over_axes(['energy']).plot(ax=ax)
    
    bkg_irf_down = get_bkg_irf(irf_down_map, 2*irf_down_offset_axis.nbin, irf_down_energy_axis, irf_down_offset_axis, bkg_dim,livetime,FoV_alignment=FoV_alignment)
    if verbose:
        print(f"irf_map: {irf_map.data.shape}\nirf_cut_map: {irf_cut_map.data.shape}\nirf_down_map: {irf_down_map.data.shape}")
    return irf_down_map, bkg_irf_down

def get_bkg_true_irf_from_config(config, downsample=True, downsample_only=True, plot=False, verbose=False):
    cfg_paths = config["paths"]

    cfg_simulation = config["simulation"]
    n_run = cfg_simulation["n_run"] # Multiplicative factor for each simulated wobble livetime
    livetime_simu = cfg_simulation["livetime"]

    cfg_source = config["source"]
    cfg_background = config["background"]
    cfg_irf = config["irf"]
    cfg_acceptance = config["acceptance"]

    catalog = SourceCatalogGammaCat(cfg_paths["gammapy_catalog"])

    src = catalog[cfg_source["catalog_name"]]
    if verbose:
        print(src.name,': ',src.position)
        print(src.spectral_model())
        print(src.spatial_model())

    bkg_dim=cfg_irf["dimension"]

    e_min, e_max = float(cfg_acceptance["energy"]["e_min"])*u.TeV, float(cfg_acceptance["energy"]["e_max"])*u.TeV
    size_fov_acc = float(cfg_acceptance["offset"]["offset_max"]) * u.deg
    nbin_E_acc, nbin_offset_acc = cfg_acceptance["energy"]["nbin"],cfg_acceptance["offset"]["nbin"]

    offset_axis_acceptance = MapAxis.from_bounds(0.*u.deg, size_fov_acc, nbin=nbin_offset_acc, name='offset')
    energy_axis_acceptance = MapAxis.from_energy_bounds(e_min, e_max, nbin=nbin_E_acc, name='energy')

    Ebin_mid = np.round(energy_axis_acceptance.edges[:-1]+(energy_axis_acceptance.edges[1:]-energy_axis_acceptance.edges[:-1])*0.5,1)
    ncols = int((len(Ebin_mid)+1)/2)
    nrows = int(len(Ebin_mid)/ncols)

    down_factor=cfg_irf["down_factor"]
    nbin_E_irf = nbin_E_acc * down_factor

    energy_axis_irf = MapAxis.from_energy_bounds(e_min, e_max, nbin=nbin_E_irf, name='energy')

    offset_factor=cfg_irf["offset_factor"]
    size_fov_irf = size_fov_acc * offset_factor
    nbin_offset_irf = nbin_offset_acc * down_factor * cfg_irf["offset_factor"]
    offset_max_irf= size_fov_irf

    offset_axis_irf = MapAxis.from_bounds(0.*u.deg, offset_max_irf, nbin=nbin_offset_irf, name='offset')

    factor = cfg_background["spectral_model"]["factor"]
    scale = cfg_background["spectral_model"]["scale"]
    bkg_tilt = factor * scale
    bkg_norm = Parameter("norm", cfg_background["spectral_model"]["norm"], unit=cfg_background["spectral_model"]["unit"], interp="log", is_norm=True)

    bkg_spectral_model = PowerLawNormSpectralModel(tilt=bkg_tilt, norm=bkg_norm, reference=cfg_background["spectral_model"]["reference"]) 

    if cfg_background['spatial_model']["model"] ==  "GaussianSpatialModel":
        bkg_spatial_model = GaussianSpatialModel(lon_0=cfg_background["spatial_model"]["lon_0"]*u.deg, lat_0=cfg_background["spatial_model"]["lat_0"]*u.deg, sigma=str(cfg_background["spatial_model"]["sigma"])+" "+cfg_background["spatial_model"]["unit"],e=cfg_background["spatial_model"]["e"],phi=cfg_background["spatial_model"]["phi"]*u.deg, frame="AltAz")
    elif cfg_background['spatial_model']["model"] ==  "GaussianSpatialModel_LinearGradient":
        bkg_spatial_model = GaussianSpatialModel_LinearGradient(lon_grad=cfg_background["spatial_model"]["lon_grad"]/u.deg,lat_grad=cfg_background["spatial_model"]["lat_grad"]/u.deg,lon_0=cfg_background["spatial_model"]["lon_0"]*u.deg, lat_0=cfg_background["spatial_model"]["lat_0"]*u.deg, sigma=str(cfg_background["spatial_model"]["sigma"])+" "+cfg_background["spatial_model"]["unit"],e=cfg_background["spatial_model"]["e"],phi=cfg_background["spatial_model"]["phi"]*u.deg, frame="AltAz")
    elif cfg_background['spatial_model']["model"] ==  "GaussianSpatialModel_LinearGradient_half":
        bkg_spatial_model = GaussianSpatialModel_LinearGradient_half(lon_grad=cfg_background["spatial_model"]["lon_grad"]/u.deg,lat_grad=cfg_background["spatial_model"]["lat_grad"]/u.deg,lon_0=cfg_background["spatial_model"]["lon_0"]*u.deg, lat_0=cfg_background["spatial_model"]["lat_0"]*u.deg, sigma=str(cfg_background["spatial_model"]["sigma"])+" "+cfg_background["spatial_model"]["unit"],e=cfg_background["spatial_model"]["e"],phi=cfg_background["spatial_model"]["phi"]*u.deg, frame="AltAz")
    bkg_true_model = FoVBackgroundModel(dataset_name="true_model", spatial_model=bkg_spatial_model, spectral_model=bkg_spectral_model)
    
    if verbose: print(bkg_true_model)
    
    bkg_true_rates = evaluate_bkg(bkg_dim,bkg_true_model,energy_axis_irf,offset_axis_irf)
    bkg_true_irf = get_bkg_irf(bkg_true_rates, 2*nbin_offset_irf, energy_axis_irf, offset_axis_irf, bkg_dim,FoV_alignment=cfg_irf["FoV_alignment"])
    
    if not downsample: 
        return bkg_true_irf
    else:
        bkg_true_map = get_irf_map(bkg_true_rates,[energy_axis_irf,offset_axis_irf],n_run*livetime_simu)
        _, bkg_true_down_irf = get_cut_downsampled_irf_from_map(bkg_true_map,[energy_axis_acceptance,offset_axis_acceptance], [offset_factor, down_factor], bkg_dim, n_run * livetime_simu, plot=plot, verbose=verbose,FoV_alignment=cfg_irf["FoV_alignment"])
        if downsample_only: 
            return bkg_true_down_irf
        else: 
            return bkg_true_irf, bkg_true_down_irf

def get_obs_collection(dir_path,pattern,multiple_simulation_subdir=False,from_index=False,with_datastore=True):
    path = Path(dir_path)
    paths = sorted(list(path.rglob(pattern)))
    
    if not multiple_simulation_subdir:
        if from_index:
            data_store = DataStore.from_dir(f"{dir_path}",hdu_table_filename=f'hdu-index{pattern}.fits.gz',obs_table_filename=f'obs-index{pattern}.fits.gz')
        else:
            data_store = DataStore.from_events_files(paths)
        obs_collection = data_store.get_observations(required_irf='all-optional')
    else:
        obs_collection = Observations()
        for path in paths:
            obs_collection.append(Observation.read(path))
        for iobs in range(len(obs_collection)):
            obs_id=iobs+1
            obs_collection[iobs].events.table.meta['OBS_ID'] = obs_id
            obs_collection[iobs].obs_id = obs_id
    
    if with_datastore:
        return data_store, obs_collection
    else: 
        return obs_collection


# PLOTS
    
def scale_value(x,xlim,ylim):
    y_min = ylim[0]
    y_max = ylim[1]
    x_min = xlim[0]
    x_max = xlim[1]
    
    # Apply the linear mapping formula
    y = (x - x_min) * ((y_max - y_min) / (x_max - x_min)) + y_min
    
    return y

def get_geom(Bkg_irf, axis_info, run_info):
    '''Return geom from bkg_irf axis or a set of given axis edges'''

    loc, source_pos, run, livetime, pointing, file_name = run_info
    
    if axis_info is not None: 
        e_min, e_max, offset_max, nbin_offset= axis_info
        energy_axis = MapAxis.from_energy_bounds(e_min, e_max, nbin=10, name='energy')
        nbins_map = 2*nbin_offset
    else:
        energy_axis = Bkg_irf.axes[0]
        e_min = energy_axis.edges.min()
        e_max = energy_axis.edges.max()
        offset_axis = Bkg_irf.axes[1]
        offset_max = offset_axis.edges.max()
        nbins_map = offset_axis.nbin
    
    binsize = offset_max / (nbins_map / 2)
    skydir=(pointing.ra.degree,pointing.dec.degree)
    
    geom = WcsGeom.create(
            skydir=skydir,
            binsz=binsize,
            npix=(nbins_map, nbins_map),
            frame='icrs',
            proj="CAR",
            axes=[energy_axis],
        )
    return geom

def get_value_at_threshold(data_to_bin,weights,threshold_percent,plot=False):
    # Filter NaN values
    data_to_bin_filtered = data_to_bin[~np.isnan(weights)]
    weights_filtered = weights[~np.isnan(weights)]

    # Compute histogram with weights
    counts, bin_edges = np.histogram(data_to_bin_filtered, bins=22, weights=weights_filtered)

    # Calculate cumulative sum of the histogram
    cumulative_counts = np.cumsum(counts)
    total_area = cumulative_counts[-1]

    # Find the x-axis value where cumulative sum reaches 90% of the total area
    threshold = threshold_percent/100 * total_area
    index_thr = np.searchsorted(cumulative_counts, threshold)
    data_value_at_thr = bin_edges[index_thr]

    if plot:
        # Plot the histogram
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(data_to_bin_filtered, weights=weights_filtered, bins=22)
        ax.axvline(x=data_value_at_thr, color='r', linestyle='--', label=f'{threshold_percent}% area at r={data_value_at_thr:.2f}°')
        ax.legend()
        ax.set_title(f"Histogram with {threshold_percent}% Area Marker")
        plt.show()
    return data_value_at_thr

def compute_residuals(out, true, residuals='diff/true',res_lim_for_nan=0.9) -> np.array:
    res_arr = np.zeros_like(true)
    for iEbin in range(true.shape[0]):
        diff = out[iEbin,:,:] - true[iEbin,:,:]
        true[iEbin,:,:][true[iEbin,:,:] == 0.0] = np.nan
        res = diff / true[iEbin,:,:]
        diff[np.abs(res) >= res_lim_for_nan] = np.nan # Limit to mask bins with "almost zero" statistics for truth model and 0 output count
        # Caveat: this will also mask very large bias, but the pattern should be visibly different

        if residuals == "diff/true": res = diff / true[iEbin,:,:]
        elif residuals == "diff/sqrt(true)": res = diff / np.sqrt(true[iEbin,:,:])

        # res[true[iEbin,:,:] == 0] = np.nan
        res_arr[iEbin,:,:] += res
    return res_arr


def get_exclusion_mask_from_dataset_geom(dataset, exclude_regions):
    geom_map = dataset.geoms['geom']
    energy_axis_map = dataset.geoms['geom'].axes[0]
    geom_image = geom_map.to_image().to_cube([energy_axis_map.squash()])
    exclusion_mask = geom_image.region_mask(exclude_regions, inside=False)
    return exclusion_mask

def get_lima_maps(dataset, correlation_radius, correlate_off):
    estimator = ExcessMapEstimator(correlation_radius*u.deg, correlate_off=correlate_off)
    lima_maps = estimator.run(dataset)
    return lima_maps

def get_dataset_maps_dict(dataset, results=['counts','background']):
    maps = {}
    if 'counts' in results: maps['counts'] = dataset.counts.sum_over_axes()
    if 'background' in results: maps['background'] = dataset.background.sum_over_axes()
    return maps

def get_high_level_maps_dict(lima_maps, exclusion_mask, results=['significance_all']):
    '''results='all', ['significance_all','significance_off','excess']'''
    if results == 'all': results = ['significance_all','significance_off','excess']
    
    significance_map = lima_maps["sqrt_ts"]
    excess_map = lima_maps["npred_excess"]
    
    maps = {}
    if 'significance_all' in results: maps['significance_all'] = significance_map
    if 'significance_off' in results: maps['significance_off'] = significance_map * exclusion_mask
    if 'excess' in results: maps['excess'] = excess_map
    
    return maps

def get_high_level_maps_from_dataset(dataset, exclude_regions, correlation_radius, correlate_off, results):
    exclusion_mask = get_exclusion_mask_from_dataset_geom(dataset, exclude_regions)
    lima_maps = get_lima_maps(dataset, correlation_radius, correlate_off)
    maps = get_high_level_maps_dict(lima_maps, exclusion_mask, results)
    return maps

def get_skymaps_dict(dataset, exclude_regions, correlation_radius, correlate_off, results):
    '''results='all', ['counts', 'background','significance_all','significance_off','excess']'''
    
    if results == 'all': results=['counts', 'background', 'significance_all', 'significance_off', 'excess']
    
    dataset_bool = 1*(('counts' in results) or ('background' in results))
    estimator_bool = -1*(('significance_all' in results) or ('significance_off' in results) or ('excess' in results))
    i_method = dataset_bool + estimator_bool # methods = {1: 'dataset_only', -1: 'estimator_only', 0: 'both'}
    
    if i_method >= 0: maps_dataset = get_dataset_maps_dict(dataset, results)
    if i_method <= 0: maps_high_level = get_high_level_maps_from_dataset(dataset, exclude_regions, correlation_radius, correlate_off, results)

    if i_method==0:  
        maps_both = maps_dataset.copy()
        maps_both.update(maps_high_level)
        return maps_both
    elif i_method==1: return maps_dataset
    else: return maps_high_level

def plot_skymap_from_dict(skymaps, key, crop_width=0 * u.deg, figsize=(5,5)):
    skymaps_args = {
        'counts': {
            'cbar_label': 'events',
            'title': 'Counts map'
        },
        'background': {
            'cbar_label': 'events',
            'title': 'Background map'
        },
        'significance': {
            'cbar_label': 'significance [$\sigma$]',
            'title': 'Significance map'
        },
        'significance_off': {
            'cbar_label': 'significance [$\sigma$]',
            'title': 'Off significance map'
        },
        'excess': {
            'cbar_label': 'events',
            'title': 'Excess map'
        }
    }

    skymap = skymaps[key]
    if crop_width != 0 * u.deg:
        width = 0.5 * skymap.geom.width[0][0]
        binsize = width / (0.5 * skymap.data.shape[-1])
        n_crop_px = int(((width - crop_width)/binsize).value)
        skymap = skymap.crop(n_crop_px)
    
    cbar_label, title = (skymaps_args[key]["cbar_label"], skymaps_args[key]["title"])
    
    fig,ax=plt.subplots(figsize=figsize,subplot_kw={"projection": skymap.geom.wcs})
    if (key=='counts') or (key=='background'): skymap.plot(ax=ax, add_cbar=True, stretch="linear",kwargs_colorbar={'label': cbar_label})
    elif key == 'excess': skymap.plot(ax=ax, add_cbar=True, stretch="linear", cmap='magma',kwargs_colorbar={'label': cbar_label})
    elif 'significance' in key: 
        skymap.plot(ax=ax, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma',kwargs_colorbar={'label': cbar_label})
        maxsig = np.nanmax(skymap.data)
        im = ax.images        
        cb = im[-1].colorbar 
        maxsig = np.nanmax(skymap.data)
        cb.ax.axhline(maxsig, c='g')
        cb.ax.text(1.1,maxsig - 0.07,'max', color = 'g')
    ax.set(title=title)
    plt.tight_layout()
    plt.show()

class BaseSimBMVCreator(ABC):

    def __init__(self) -> None:
        """
        Create the class to perform model validation.

        TO-DO: Parameters description
        ----------
        """

    def init_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        # Print the loaded configuration
        for key in config.keys(): print(f"{key}: {config[key]}")
        self.config = config
        self.config_path = config_path
        self.cfg_paths = config["paths"]
        self.cfg_data = self.config["data"]
        self.cfg_simulation = config["simulation"]
        self.cfg_wobble_1 = config["wobble_1"]
        self.cfg_wobble_2 = config["wobble_2"]
        self.cfg_source = config["source"]
        self.cfg_background = config["background"]
        self.cfg_irf = config["irf"]
        self.cfg_dataset = config["dataset"]
        self.cfg_acceptance = config["acceptance"]

        # Paths
        self.output_dir = self.cfg_paths["output_dir"]
        self.simulated_obs_dir = self.cfg_paths["simulated_obs_dir"]
        self.save_name_obs = self.cfg_paths["save_name_obs"]
        self.save_name_suffix = self.cfg_paths["save_name_suffix"]
        self.path_data=self.cfg_paths["path_data"] # path to files to get run information used for simulation

        # Data
        self.real_data = self.cfg_data["real"]
        self.run_list = np.array(self.cfg_data["run_list"])
        self.all_obs_ids = np.array([])
        self.obs_pattern = self.cfg_data["obs_pattern"]

        # Source
        self.source_name=self.cfg_source["catalog_name"]
        catalog = SourceCatalogGammaCat(self.cfg_paths["gammapy_catalog"])
        if self.source_name in catalog.table['common_name']:
            self.source = catalog[self.source_name]
            self.source_pos=self.source.position
        else:
            self.source = SourceCatalogObject(data={'Source_Name':self.source_name,'RA': self.cfg_source["coordinates"]["ra"]*u.deg, 'DEC':self.cfg_source["coordinates"]["dec"]})
            self.source_pos=self.source.position

        self.region_shape = self.cfg_source["exclusion_region"]["shape"]
        self.source_region = []

        if (self.region_shape=='noexclusion'):
            # Parts of the code don't work without an exlcusion region, so a dummy one is declared at the origin of the ICRS frame (and CircleSkyRegion object needs a non-zero value for the radius) 
            self.source_region.append(CircleSkyRegion(center=SkyCoord(ra=0. * u.deg, dec=0. * u.deg, frame='icrs'),radius=1*u.deg))
        
        elif (self.region_shape=='n_circles'):
            self.n_circles = len(self.cfg_source["exclusion_region"]["n_circles"])
            for i in range(self.n_circles):
                cfg_circle = self.cfg_source["exclusion_region"]["n_circles"][f"circle_{i+1}"]
                circle_pos = SkyCoord(ra=cfg_circle["ra"]*u.deg, dec=cfg_circle["dec"]*u.deg, frame='icrs')
                circle_rad = cfg_circle["radius"] * u.deg
                self.source_region.append(CircleSkyRegion(center=circle_pos, radius=circle_rad))
                if i == 0:
                    self.exclusion_radius=circle_rad
                    self.exclu_rad = circle_rad.to_value(u.deg)
        
        elif (self.region_shape=='ellipse'):
            self.width = self.cfg_source["exclusion_region"]["ellipse"]["width"] * u.deg
            self.height = self.cfg_source["exclusion_region"]["ellipse"]["height"] * u.deg
            self.angle = self.cfg_source["exclusion_region"]["ellipse"]["angle"] * u.deg
            self.source_region = [EllipseSkyRegion(center=self.source_pos, width=self.width, height=self.height, angle=self.angle)]

        self.single_region=True # Set it to False to mask additional regions in the FoV. The example here is for Crab + Zeta Tauri
        self.exclude_regions=[]
        for region in self.source_region: self.exclude_regions.append(region)
        if not self.single_region:
            # TO-DO: catalog of classic regions to mask. Here it is an example for masking zeta tauri on crab observations
            zeta_region = CircleSkyRegion(center=SkyCoord(ra=84.4125*u.deg, dec=21.1425*u.deg), radius=self.exclusion_radius)
            self.exclude_regions.append(zeta_region)

        self.source_info = [self.source_name,self.source_pos,self.source_region]
        self.flux_to_0 = self.cfg_source["flux_to_0"]
        
        # Background
        self.lon_grad = self.cfg_background["spatial_model"]["lon_grad"]
        self.correlation_radius = self.cfg_background["maker"]["correlation_radius"]
        self.correlate_off = self.cfg_background["maker"]["correlate_off"]
        self.ring_bkg_param = [self.cfg_background["maker"]["ring"]["internal_ring_radius"],self.cfg_background["maker"]["ring"]["width"]]
        self.fov_bkg_param = self.cfg_background["maker"]["fov"]["method"]
        
        # Acceptance binning
        self.e_min, self.e_max = float(self.cfg_acceptance["energy"]["e_min"])*u.TeV, float(self.cfg_acceptance["energy"]["e_max"])*u.TeV
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

        self.radius_edges = np.linspace(0,np.sqrt(2*self.size_fov_acc.value**2),10)
        self.radius_centers = self.radius_edges[:-1]+0.5*(self.radius_edges[1:]-self.radius_edges[:-1])
        
        # Background IRF
        self.bkg_dim = self.cfg_irf["dimension"]
        self.true_collection = self.cfg_irf["collection"]
        self.down_factor=self.cfg_irf["down_factor"]
        self.nbin_E_irf = self.nbin_E_acc * self.down_factor

        self.energy_axis_irf = MapAxis.from_energy_bounds(self.e_min, self.e_max, nbin=self.nbin_E_irf, name='energy')

        self.offset_factor=self.cfg_irf["offset_factor"]
        self.size_fov_irf = self.size_fov_acc * self.offset_factor
        self.nbin_offset_irf = self.nbin_offset_acc * self.down_factor * self.cfg_irf["offset_factor"]
        self.offset_axis_irf = MapAxis.from_bounds(0.*u.deg, self.size_fov_irf, nbin=self.nbin_offset_irf, name='offset')

        ## Spectral model
        factor = self.cfg_background["spectral_model"]["factor"]
        scale = self.cfg_background["spectral_model"]["scale"]
        norm = self.cfg_background["spectral_model"]["norm"]
        unit = self.cfg_background["spectral_model"]["unit"]
        reference = self.cfg_background["spectral_model"]["reference"]
        bkg_tilt = factor * scale
        bkg_norm = Parameter("norm", norm, unit=unit, interp="log", is_norm=True)
        self.bkg_spectral_model = PowerLawNormSpectralModel(tilt=bkg_tilt, norm=bkg_norm, reference=reference) 

        ## Spatial model
        spatial_model = self.cfg_background['spatial_model']["model"]
        if spatial_model ==  "GaussianSpatialModel":
            bkg_spatial_model = GaussianSpatialModel(lon_0=self.cfg_background["spatial_model"]["lon_0"]*u.deg, lat_0=self.cfg_background["spatial_model"]["lat_0"]*u.deg, sigma=str(self.cfg_background["spatial_model"]["sigma"])+" "+self.cfg_background["spatial_model"]["unit"],e=self.cfg_background["spatial_model"]["e"],phi=self.cfg_background["spatial_model"]["phi"]*u.deg, frame="AltAz")
        elif spatial_model ==  "GaussianSpatialModel_LinearGradient":
            bkg_spatial_model = GaussianSpatialModel_LinearGradient(lon_grad=self.cfg_background["spatial_model"]["lon_grad"]/u.deg,lat_grad=self.cfg_background["spatial_model"]["lat_grad"]/u.deg,lon_0=self.cfg_background["spatial_model"]["lon_0"]*u.deg, lat_0=self.cfg_background["spatial_model"]["lat_0"]*u.deg, sigma=str(self.cfg_background["spatial_model"]["sigma"])+" "+self.cfg_background["spatial_model"]["unit"],e=self.cfg_background["spatial_model"]["e"],phi=self.cfg_background["spatial_model"]["phi"]*u.deg, frame="AltAz")
        elif spatial_model ==  "GaussianSpatialModel_LinearGradient_half":
            bkg_spatial_model = GaussianSpatialModel_LinearGradient_half(lon_grad=self.cfg_background["spatial_model"]["lon_grad"]/u.deg,lat_grad=self.cfg_background["spatial_model"]["lat_grad"]/u.deg,lon_0=self.cfg_background["spatial_model"]["lon_0"]*u.deg, lat_0=self.cfg_background["spatial_model"]["lat_0"]*u.deg, sigma=str(self.cfg_background["spatial_model"]["sigma"])+" "+self.cfg_background["spatial_model"]["unit"],e=self.cfg_background["spatial_model"]["e"],phi=self.cfg_background["spatial_model"]["phi"]*u.deg, frame="AltAz")
        self.bkg_true_model = FoVBackgroundModel(dataset_name="true_model", spatial_model=bkg_spatial_model, spectral_model=self.bkg_spectral_model)

        # Some variables used for the custom plotting methods
        self.axis_info_acceptance = [self.e_min,self.e_max,self.size_fov_acc,self.nbin_offset_acc]
        self.axis_info_dataset = [self.e_min,self.e_max,self.size_fov_irf,self.nbin_offset_irf]
        self.axis_info_map = [self.e_min,self.e_max,self.size_fov_irf,self.nbin_offset_irf]

        # Simulation
        self.loc = EarthLocation.of_site('Roque de los Muchachos')
        self.location = observatory_locations["cta_north"]
        
        self.t_ref = self.cfg_simulation["t_ref"]
        self.delay = self.cfg_simulation["delay"]
        self.time_oversampling = self.cfg_simulation["time_oversampling"] * u.s
        self.fov_rotation_error_limit = self.cfg_simulation["fov_rotation_error_limit"] * u.deg
        
        self.single_pointing = self.cfg_simulation["single_pointing"]
        self.obs_collection_type = self.cfg_simulation["obs_collection_type"]
        self.two_obs = self.obs_collection_type == 'two_wobble_obs'
        self.n_run = self.cfg_simulation["n_run"]
        self.livetime_simu = self.cfg_simulation["livetime"]
        self.tot_livetime_simu = self.n_run*self.livetime_simu*u.s
        if not self.single_pointing: self.tot_livetime_simu *= 2
        
        #  Wobble 1
        self.seed_W1=self.cfg_wobble_1["seed"]
        run_W1=self.cfg_wobble_1["run"]
        livetime_W1,self.pointing_W1,file_name_W1 = get_run_info(self.path_data,self.obs_pattern,run_W1) 
        # The true livetime is retrieved by get_run_info in case you want to implement a very realistic simulation pipeline with a simulation for each true observation you have
        # Here we use the same info for every run, which is the simulation livetime
        # The get_run_info method is mostly use to have realistic pointings, but you can decide the values yourself by changing the next lines
        self.run_info_W1=[self.loc,self.source_pos,run_W1,self.livetime_simu*(self.n_run if self.two_obs else 1),self.pointing_W1,file_name_W1]
        
        #  Wobble 2
        self.seed_W2=self.cfg_wobble_2["seed"]
        run_W2=self.cfg_wobble_2["run"]
        livetime_W2,self.pointing_W2,file_name_W2 = get_run_info(self.path_data,self.obs_pattern,run_W2)
        self.run_info_W2=[self.loc,self.source_pos,run_W2,self.livetime_simu*(self.n_run if self.two_obs else 1),self.pointing_W2,file_name_W2]
        
        print(f"Total simulated livetime: {self.tot_livetime_simu.to(u.h):.1f}")

        # Naming scheme
        self.save_name_obs = f"{self.cfg_paths['save_name_obs']}"
        if self.two_obs: self.save_name_obs += f"_{0.5*self.tot_livetime_simu.to(u.h).value:.0f}h"

        self.multiple_simulation_subdir = False # TO-DO adapt for multiple subdirectories
        self.save_path_simu_joined = ''
        if self.multiple_simulation_subdir: self.save_path_simu = self.save_path_simu_joined
        elif not self.real_data: self.save_path_simu = f"{self.simulated_obs_dir}/{self.save_name_obs}/{self.obs_collection_type}"+f"_{self.save_name_suffix}"*(self.save_name_suffix is not None)
        else: self.save_path_simu = self.cfg_data['save_path_data']
        # Acceptance parameters
        self.method = self.cfg_acceptance["method"]

        self.out_collection = self.cfg_acceptance["collection"]
        self.single_file_path = self.cfg_acceptance["single_file_path"]

        self.cos_zenith_binning = self.cfg_acceptance["cos_zenith_binning"]
        self.zenith_binning=self.cos_zenith_binning["zenith_binning"]
        self.runwise_normalisation=self.cos_zenith_binning["runwise_normalisation"]
        self.initial_cos_zenith_binning=self.cos_zenith_binning['initial_cos_zenith_binning']
        self.cos_zenith_binning_method=self.cos_zenith_binning['cos_zenith_binning_method']
        self.cos_zenith_binning_parameter_value=self.cos_zenith_binning['cos_zenith_binning_parameter_value']

        self.fit_fnc=self.cfg_acceptance["fit"]["fnc"]
        self.fit_bounds=self.cfg_acceptance["fit"]["bounds"]

        # Output files
        region_shape = self.region_shape if self.region_shape != "n_circles" else f"{self.n_circles}_circle{'s'*(self.n_circles > 1)}"
        self.end_name=f'{self.bkg_dim}D_{region_shape}_Ebins_{self.nbin_E_acc}_offsetbins_{self.nbin_offset_acc}_offset_max_{self.size_fov_acc.value:.1f}'+f'_{self.cos_zenith_binning_parameter_value}sperW'*self.zenith_binning+f'_exclurad_{self.exclu_rad}'*((self.exclu_rad != 0) & (self.region_shape != 'noexclusion'))
        self.index_suffix = f"_with_bkg_{self.bkg_dim}d_{self.method}_{self.end_name[3:]}"
        if self.real_data:
            self.acceptance_files_dir = f"{self.output_dir}/{self.save_name_obs}/{self.end_name}/{self.method}/acceptances"
            self.plots_dir=f"{self.output_dir}/{self.save_name_obs}/{self.end_name}/{self.method}/plots"
        else:
            self.acceptance_files_dir=f"{self.save_path_simu}/{self.end_name}/{self.method}/acceptances"
            self.plots_dir=f"{self.save_path_simu}/{self.end_name}/{self.method}/plots"
        
        Path(self.acceptance_files_dir).mkdir(parents=True, exist_ok=True)
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
        print("acceptance files: ",self.acceptance_files_dir)
    
    def load_true_background_irfs(self, config_path=None) -> None:
        if config_path is not None: self.init_config(config_path)
        
        if self.true_collection:
            self.bkg_true_irf_collection = {}
            self.bkg_true_down_irf_collection = {}
            tmp_config = self.config
            self.lon_grad_step = np.diff(np.linspace(0,abs(self.lon_grad), self.n_run))[0]

            for i_step in range(self.n_run):
                lon_grad_new = self.lon_grad + i_step * self.lon_grad_step
                tmp_config["spatial_model"]["lon_grad"] = lon_grad_new

                if (i_step == 0) or  (i_step == self.n_run-1): plot,verbose=(False,True)
                else: plot,verbose=(False,False)

                bkg_true_irf, bkg_true_down_irf = get_bkg_true_irf_from_config(tmp_config,downsample=True,downsample_only=False,plot=plot,verbose=verbose)
                self.bkg_true_irf_collection[i_step] = bkg_true_irf
                self.bkg_true_irf_down_collection[i_step] = bkg_true_down_irf
                logging.INFO(tmp_config["spatial_model"]["lon_grad"])
        else:
            self.bkg_true_irf, self.bkg_true_down_irf = get_bkg_true_irf_from_config(self.config,downsample=True,downsample_only=False,plot=False,verbose=True)

    def load_output_background_irfs(self, config_path=None) -> None:
        if config_path is not None: self.init_config(config_path)

        if self.out_collection:
            self.bkg_output_irf_collection = {}

            self.obs_ids = self.all_obs_ids if self.run_list.shape[0] == 0 else self.run_list
            if self.real_data:
                for iobs,obs_id in enumerate(self.obs_ids): self.bkg_output_irf_collection[iobs] = Background3D.read(f'{self.acceptance_files_dir}/acceptance_obs-{obs_id}.fits')
            else:
                for irun in range(self.n_run): self.bkg_output_irf_collection[irun+1] = Background3D.read(f'{self.acceptance_files_dir}/acceptance_obs-{irun+1}.fits')
        else:
            self.bkg_output_irf = Background3D.read(self.single_file_path)
    
    def write_datastore_with_new_model(self, model_path=''):
        # For each observation get the acceptance map, save it and add the saved file path to the data store as a background map
        data_store = DataStore.from_dir(f"{self.save_path_simu}",hdu_table_filename=f'hdu-index.fits.gz',obs_table_filename=f'obs-index.fits.gz')
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
        print(file_name)
        for i in range(len(all_obs_ids)):
            obs_id=all_obs_ids[i]
            data_store.hdu_table.add_row({'OBS_ID': f"{obs_id}", 
                                            'HDU_TYPE': 'bkg',
                                            "HDU_CLASS": f"bkg_{self.bkg_dim}d",
                                            "FILE_DIR": str(file_dir),
                                            "FILE_NAME": str(file_name[obs_id]) if path.is_dir() else str(file_name),
                                            "HDU_NAME": "BACKGROUND"})

        # Save the new data store for future use
        data_store.hdu_table.write(f"{self.save_path_simu}/hdu-index{self.index_suffix}.fits.gz",format="fits",overwrite=True) 
        data_store.obs_table.write(f"{self.save_path_simu}/obs-index{self.index_suffix}.fits.gz",format="fits",overwrite=True)

    def load_observation_collection(self, config_path=None, from_index=False) -> None:
        if config_path is not None: self.init_config(config_path)
        
        if not self.multiple_simulation_subdir:
            if from_index: self.pattern = self.index_suffix
            elif not self.real_data: self.pattern = f"obs_*{self.save_name_obs}.fits"
            else: self.pattern = self.obs_pattern
            print(f"Obs collection loading pattern: {self.pattern}")
            self.data_store, self.obs_collection = get_obs_collection(self.save_path_simu,self.pattern,self.multiple_simulation_subdir,from_index=from_index,with_datastore=True)
            self.obs_table = self.data_store.obs_table
            
            # Save the new data store for future use
            if not pathlib.Path(f"{self.save_path_simu}/hdu-index.fits.gz").exists(): 
                self.data_store.hdu_table.write(f"{self.save_path_simu}/hdu-index.fits.gz",format="fits")
            if not pathlib.Path(f"{self.save_path_simu}/obs-index.fits.gz").exists(): 
                self.data_store.obs_table.write(f"{self.save_path_simu}/obs-index.fits.gz",format="fits")

            self.all_sources = np.unique(self.data_store.obs_table["OBJECT"])
            self.all_obs_ids = np.array(self.obs_table["OBS_ID"].data)
            print("Available sources: ", self.all_sources)
            print(f"{len(self.all_obs_ids)} available runs: ",self.all_obs_ids)
            self.tot_livetime_simu = 2*self.n_run*self.livetime_simu*u.s

            self.obs_ids = self.all_obs_ids if self.run_list.shape[0] == 0 else self.run_list
            if self.run_list.shape[0] != 0: 
                self.obs_collection = self.data_store.get_observations(self.obs_ids, required_irf=['aeff', 'edisp'])
                self.obs_table = self.data_store.obs_table.select_obs_id(self.obs_ids)
                print(f"{len(self.obs_ids)} selected runs: ",self.obs_ids)
            else: print("All runs selected")
            
            if self.real_data:
                # Add telescope position to observations
                for iobs in range(len(self.obs_collection)):
                    meta_dict = self.obs_collection[iobs].events.table.meta
                    meta_dict.__setitem__('GEOLON',str(self.loc.lon.value))
                    meta_dict.__setitem__('GEOLAT',str(self.loc.lat.value))
                    meta_dict.__setitem__('GEOALT',str(self.loc.height.to_value(u.m)))
                    meta_dict.__setitem__('deadtime_fraction',str(1-meta_dict['DEADC']))
                    self.obs_collection[iobs]._meta = ObservationMetaData.from_header(meta_dict)
                    self.obs_collection[iobs]._location = self.loc
                    self.obs_collection[iobs].pointing._location = self.loc
                    # self.obs_collection[iobs].obs_info['observatory_earth_location'] = self.loc # <- modifié pour être accessible à l'intérieur de la méthode qui récupère le pointé

        else:
            self.pattern = f"{self.obs_collection_type}_{self.save_name_suffix[:-8]}*/obs_*{self.save_name_obs}.fits" # Change pattern according to your sub directories
            self.obs_collection = get_obs_collection(self.simulated_obs_dir,self.pattern,self.multiple_simulation_subdir,with_datastore=False)
            self.all_obs_ids = np.arange(1,len(self.obs_collection)+1,1)
            self.tot_livetime_simu = 2*len(self.obs_collection)*self.livetime_simu*u.s
        self.total_livetime = sum([obs.observation_live_time_duration for obs in self.obs_collection])
        if not isinstance(self.total_livetime, u.Quantity): self.total_livetime *= u.s
        print(f"Total livetime: {self.total_livetime.to(u.h):.1f}")
    
    def do_simulation(self, config_path=None):
        if config_path is not None: self.init_config(config_path)

        Path(self.save_path_simu).mkdir(parents=True, exist_ok=True)
        files = glob.glob(str(self.save_path_simu)+'/*fit*')
        for f in files:
            Path(f).unlink()
        
        bkg_irf_W1=self.bkg_true_irf
        bkg_irf_W2=self.bkg_true_irf # In case you want different IRFs you can change it here

        for wobble,bkg_irf,run_info,random_state in zip([1,2],[bkg_irf_W1,bkg_irf_W2],[self.run_info_W1,self.run_info_W2],[self.seed_W1,self.seed_W2]):
            i = 0
            if self.single_pointing and (wobble ==  2): break

            # Loop pour résolution temporelle. Paramètre "oversampling"
            for iobs in range(1 if self.two_obs else self.n_run):
                print(iobs)
                events_all = None
                oversampling = self.time_oversampling
                obs_id = iobs + 1 + (1 if self.two_obs else self.n_run)*(wobble==2)
                verbose = iobs == 0
                sampler = MapDatasetEventSampler(random_state=random_state+iobs)
                if self.true_collection: obs = get_empty_obs_simu(self.bkg_true_collection[iobs],None,run_info,self.source,self.path_data,self.flux_to_0,self.t_ref,i*self.delay,verbose)
                else: obs = get_empty_obs_simu(bkg_irf,None,run_info,self.source,self.path_data,self.flux_to_0,self.t_ref,i*self.delay,verbose)
                n = int(run_info[3]/oversampling.to_value("s")) + 1
                oversampling = (run_info[3]/n) * u.s
                run_info_over = deepcopy(run_info)
                run_info_over[3] = oversampling.to_value("s")
                for j in range(n):
                    print(j)
                    if self.true_collection: tdataset, tobs = get_empty_dataset_and_obs_simu(self.bkg_true_collection[iobs],None,run_info_over,self.source,self.path_data,self.flux_to_0,self.t_ref,i*self.delay+j*oversampling.to_value("s"),verbose=False)
                    else: tdataset, tobs = get_empty_dataset_and_obs_simu(bkg_irf,None,run_info_over,self.source,self.path_data,self.flux_to_0,self.t_ref,i*self.delay+j*oversampling.to_value("s"),verbose=False)
                    tdataset.fake(random_state=random_state+iobs)

                    events = sampler.run(tdataset, tobs)
                    events.table.meta['ALT_PNT']=None
                    events.table.meta['AZ_PNT']=None
                    if events_all is None:
                        events_all = events
                    else:
                        events_all = stack_with_meta(events_all, events)
                
                events_all.table.meta['OBS_ID'] = obs_id
                obs._events = events_all
                obs.write(f"{self.save_path_simu}/obs_W{wobble}_{'0'*(obs_id<10)+'0'*(obs_id<100)}{obs_id}_{self.save_name_obs}.fits",overwrite=True)
                i+=1
        del(obs)
        shutil.copyfile(self.config_path, f'{self.save_path_simu}/config_simu.yaml')
        print(f"Simulation dir: {self.save_path_simu}")
        path = Path(self.save_path_simu) 
        paths = sorted(list(path.rglob("obs_*.fits")))
        data_store = DataStore.from_events_files(paths)
        data_store.obs_table.write(f"{self.save_path_simu}/obs-index.fits.gz", format='fits',overwrite=True)
        data_store.hdu_table.write(f"{self.save_path_simu}/hdu-index.fits.gz", format='fits',overwrite=True)
        self.load_observation_collection()

    def do_acceptance_modelisation(self, config_path=None):
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
            self.data_store_out = DataStore.from_dir(f"{self.save_path_simu}",hdu_table_filename=f'hdu-index.fits.gz',obs_table_filename=f'obs-index.fits.gz')

            # Bkg row cannot be overwritten, if it exists it needs to be removed before adding the new one
            if 'bkg' in self.data_store_out.hdu_table['HDU_TYPE']:
                self.data_store_out.hdu_table.remove_rows(self.data_store_out.hdu_table['HDU_TYPE']=='bkg')

            for i in range(len(self.obs_ids)):
                obs_id=self.all_obs_ids[i]
                hdu_acceptance = acceptance_model[obs_id].to_table_hdu()
                hdu_acceptance.writeto(f'{self.acceptance_files_dir}/acceptance_obs-{obs_id}.fits', overwrite=True)
                self.data_store_out.hdu_table.add_row({'OBS_ID': f"{obs_id}", 
                                                'HDU_TYPE': 'bkg',
                                                "HDU_CLASS": f"bkg_{self.bkg_dim}d",
                                                "FILE_DIR": self.acceptance_files_dir,
                                                "FILE_NAME": f'acceptance_obs-{obs_id}.fits',
                                                "HDU_NAME": "BACKGROUND"})

            # Save the new data store for future use
            self.data_store_out.hdu_table.write(f"{self.save_path_simu}/hdu-index{self.index_suffix}.fits.gz",format="fits",overwrite=True) 
            self.data_store_out.obs_table.write(f"{self.save_path_simu}/obs-index{self.index_suffix}.fits.gz",format="fits",overwrite=True)
            self.load_observation_collection(from_index=True)
        else:
            # I still don't know how to create a new datastore from an observation collection, without having to save them again and store them twice
            for i in range(len(self.all_obs_ids)):
                obs_id=self.all_obs_ids[i]
                hdu_acceptance = acceptance_model[obs_id].to_table_hdu()
                hdu_acceptance.writeto(f'{self.acceptance_files_dir}/acceptance_obs-{obs_id}.fits', overwrite=True)
        self.load_output_background_irfs()
    
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
        geom_image = geom.to_image().to_cube([energy_axis.squash()])

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
                dataset_on_off = ring_bkg_maker.run(dataset_loc.to_image())
                stacked_on_off.stack(dataset_on_off)
            else:
                if bkg_method == 'FoV':  
                    dataset_loc.counts.data[~dataset_loc.mask_safe] = 0
                    dataset_loc = FoV_background_maker.run(dataset_loc)
                stacked_on_off.append(dataset_loc)
        
        if bkg_method == 'ring': return stacked_on_off
        else: return stacked_on_off.stack_reduce()
            
    def get_background_irf(self, type='true', downsampled=True) -> BackgroundIRF:
        if self.true_collection:
            if type=='true': 
                if downsampled: return self.bkg_true_down_irf_collection
                else: return self.bkg_true_collection
            elif type=='output':
                return self.bkg_output_irf_collection
        else:
            if type=='true': 
                if downsampled: return self.bkg_true_down_irf
                else: return self.bkg_true_irf
            elif type=='output':
                return self.bkg_output_irf
    
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
    
    def plot_profile(self, irf='both', profile='both', stat='sum', bias=False, ratio_lim = [0.95,1.05], all_Ebins=False, fig_save_path=''):
        if self.true_collection: bkg_true_irf = deepcopy(self.bkg_true_down_irf_collection[1])
        else: bkg_true_irf = deepcopy(self.bkg_true_down_irf)
        
        profile_true = self.get_dfprofile(bkg_true_irf)
        
        if (irf=='output') or (irf=='both'): 
            if self.out_collection: bkg_out_irf = deepcopy(self.bkg_output_irf_collection[1])
            else: bkg_out_irf = deepcopy(self.bkg_output_irf)
            
            profile_out = self.get_dfprofile(bkg_out_irf)
        
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
        fov_edges = self.bkg_true_down_irf.axes['fov_lon'].edges.value
        fov_centers = self.bkg_true_down_irf.axes['fov_lon'].center.value
        offset_edges = fov_edges[fov_edges >= 0]
        offset_centers = fov_centers[fov_centers >= 0]
        radius_centers = self.radius_centers
        radius_edges = self.radius_edges
        
        # Flatten and filter NaNs
        x_centers = fov_centers
        y_centers = fov_centers
        X_, Y_ = np.meshgrid(x_centers, y_centers)
        r_center = np.sqrt(X_**2 + Y_**2).flatten()
        weights_Ebinall = np.sum(self.bkg_true_down_irf.data, axis=0)[np.newaxis, :, :].flatten()
        r_threshold = 99.8
        r_3sig = get_value_at_threshold(r_center,weights_Ebinall,r_threshold,plot=False)

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
                if bias: fig, (ax_lon,ax_lat) = plt.subplots(1,2,figsize=(12,(2.5/4)*6))
                else: fig, ((ax_lon,ax_lat),(ax_lon_ratio,ax_lat_ratio)) = plt.subplots(2,2,figsize=(12,6), gridspec_kw={'height_ratios': [2.5, 1.5]})
            if ((irf=='true') or (irf=='both')) and not bias: fig_true, (ax_lon_true,ax_lat_true) = plt.subplots(1,2,figsize=(12,(2.5/4)*6))
            for iEbin,Ebin in enumerate(np.concatenate((E_centers,[-1]))):
                if not all_Ebins and Ebin != -1: continue
                else:
                    if (Ebin != -1): label=self.Ebin_labels[iEbin]
                    else: label=f"{self.e_min.value:.1f}-{self.e_max.value:.1f} {self.energy_axis_acceptance.unit}"

                    fov_is_in_3sig = np.abs(profile_true.xs((Ebin), axis=1).xs(('lon_lat'),axis=1).loc[('fov_lon','sum')].index) < r_3sig
                    
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

                    if not bias:
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
                    
                    if ((irf=='output') or  (irf=='both')) and not bias:
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
                if bias: fig, ax = plt.subplots(figsize=(12,(2.5/4)*6))
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

                
                    if not bias:
                        y_true=profile_true.xs((Ebin), axis=1).xs(('offset'),axis=1).loc[('fov_offset',stat)]
                        
                        if  ((irf=='true') or  (irf=='both')) and  (all_Ebins or (Ebin == -1)):
                            sns.lineplot(x=radius_centers, y=y_true, ax=ax_true, label=label, lw=lw, ls=ls_out, marker=m_lon)

                            ax_true.set(xlim=xlim,title=title,xlabel=xlabel,ylabel=ylabel)
                            ax_true.grid(True, alpha=0.2)
                            if not bias: ax_true.set(yscale='log')
                            fig_true.suptitle(suptitle+": true")
                            fig_true.tight_layout()
                
                if ((irf=='output') or (irf=='both')) and not bias:
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

    def create_zenith_binned_collections(self, collections=["output", "observation"], zenith_bins='auto'):
        '''Initialise zenith binned collections: observations and models
        collections = ["true", "output", "observation"]: List of the collections you want binned with
        zenith_bins = 'auto', 'config'
        Observations need to be loaded previously'''
        
        self.cos_zenith_observations = np.array(
                    [np.cos(obs.get_pointing_altaz(obs.tmid).zen) for obs in self.obs_collection])
        
        if zenith_bins=='auto':
            cos_min, cos_max = self.cos_zenith_observations.min(), self.cos_zenith_observations.max()
            self.cos_zenith_bin_edges = np.flip(np.linspace(cos_min, cos_max, 5))
            self.cos_zenith_bin_centers = np.flip(self.cos_zenith_bin_edges[:-1] + 0.5*(self.cos_zenith_bin_edges[1:]-self.cos_zenith_bin_edges[:-1]))
        elif zenith_bins=='config':
            self.cos_zenith_bin_edges = np.flip(self.cfg_data["cos_zenith_bin_edges"])
            self.cos_zenith_bin_centers = np.flip(self.cfg_data["cos_zenith_bin_centers"])

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

    def plot_model(self, data='acceptance', irf='true', residuals='none', profile='none', downsampled=True, i_irf=1, zenith_binned=False, title='', fig_save_path='', plot_hist=False) -> None:
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
        if (irf == 'true') or (irf == 'both'):
            if zenith_binned:
                for i, model in enumerate(self.zenith_binned_bkg_true_down_irf_collection[i_irf]):
                    if i==0: true = self.zenith_binned_bkg_true_down_irf_collection[i_irf][0]
                    else: true.data += model.data
                true.data /= (i+1)
            else: 
                if self.true_collection: true = self.bkg_true_down_irf_collection[i_irf]
                else: true = self.bkg_true_down_irf
        if (irf == 'output') or (irf == 'both'): 
            if zenith_binned:
                for i, model in enumerate(self.zenith_binned_bkg_output_irf_collection[i_irf]):
                    if i==0: out = self.zenith_binned_bkg_output_irf_collection[i_irf][0]
                    else: out.data += model.data
                out.data /= (i+1)
            else: 
                if self.out_collection: out = self.bkg_output_irf_collection[i_irf]
                else: out = self.bkg_output_irf
        
        radec_map = (data == 'bkg_map')
        if radec_map:            
            # By default the map is for W1 pointing
            # TO-DO: option to chose the pointing
            pointing, run_info = self.pointing_W1, self.run_info_W1
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

            if (irf == 'true') or (irf == 'both'):
                map_true = make_map_background_irf(pointing_info, ontime, true, geom, oversampling=oversampling, use_region_center=True, obstime=obstime, fov_rotation_error_limit=self.fov_rotation_error_limit)
                map_true_cut = map_true.cutout(position=pointing,width=2*offset_max)
                map_true_cut.sum_over_axes(["energy"]).plot()
                true = map_true.data

            if (irf == 'output') or (irf == 'both'):
                map_out = make_map_background_irf(pointing_info, ontime, out, geom, oversampling=oversampling, use_region_center=True, obstime=obstime, fov_rotation_error_limit=self.fov_rotation_error_limit)
                map_out_cut = map_true.cutout(position=pointing,width=2*offset_max)
                map_out_cut.sum_over_axes(["energy"]).plot()
                out = map_out.data
             
            xlabel,ylabel=("Ra offset [°]", "Dec offset [°]")
            cbar_label = 'Counts'
        else: 
            xlabel,ylabel=("FoV Lat [°]", "FoV Lon [°]")
            cbar_label = 'Background [MeV$^{-1}$s$^{-1}$sr$^{-1}$]'
            if (irf == 'true') or (irf == 'both'): true = true.data
            if (irf == 'output') or (irf == 'both'): out = out.data
        
        res_type_label = " diff / true" if residuals == "diff/true" else "diff / $\sqrt{true}$"

        rot = 65
        nncols = 3
        n = self.nbin_E_acc
        cols = min(nncols, n)
        rows = 1 + (n - 1) // cols
        width = 16
        cfraction = 0.15

        if residuals != "none": self.res_arr = compute_residuals(out, true, residuals=residuals,res_lim_for_nan=0.9)

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(width, rows * width // (cols * (1 + cfraction))))
        for iax, ax in enumerate(axs.flat[:n]):
            if residuals != "none":
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

        if (residuals != "none"): 
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
    
    def plot_zenith_binned_model(self, data='acceptance', irf='output', i_bin=-1, zenith_bins='auto', residuals='none', profile='none', fig_save_path='') -> None:
        '''Create a zenith binned collection and plot model data
        By default all zenith bins are plotted with i_bin == -1
        Set it to bin index value to plot a single bin'''
        plot_all_bins = (i_bin ==-1)
        collections = ["true", "output"] if irf == "both" else [irf]
        self.create_zenith_binned_collections(collections=collections, zenith_bins=zenith_bins)
        for icos,cos_center in enumerate(self.cos_zenith_bin_centers):
            if plot_all_bins or (icos == i_bin):
                zd_bin_center = np.rad2deg(np.arccos(cos_center))
                title = f"Zenith binned averaged model data\nzd = {zd_bin_center:.1f}°, {self.obs_in_coszd_bin[icos].shape[0]} runs"
                if fig_save_path == '': fig_save_path=f"{self.plots_dir}/averaged_binned_acceptance_zd_{zd_bin_center:.0f}.png"
                else:  fig_save_path=f"{fig_save_path[:-4]}_{zd_bin_center:.0f}.png"
                self.plot_model(data=data, irf=irf, residuals=residuals, profile=profile, downsampled=True, i_irf=cos_center, zenith_binned=True, title=title, fig_save_path=fig_save_path, plot_hist=False)

    def plot_exclusion_mask(self):
        geom=get_geom(None,self.axis_info_dataset,self.run_info_W1)
        geom_image = geom.to_image().to_cube([self.energy_axis_irf.squash()])

        # Make the exclusion mask
        exclusion_mask = geom_image.region_mask(self.exclude_regions, inside=False)
        exclusion_mask.cutout(self.source_pos, self.size_fov_acc).plot()
        plt.show()
    
    def plot_lima_maps(self, dataset, axis_info, cutout=False, method='FoV', fig_save_path=''):
        '''Compute and store skymaps'''
        emin_map, emax_map, offset_max, nbin_offset = axis_info
        internal_ring_radius,width_ring = self.ring_bkg_param
        if cutout: dataset = dataset.cutout(position = self.source_pos, width=offset_max)

        self.exclusion_mask = get_exclusion_mask_from_dataset_geom(dataset, self.exclude_regions)
        self.maps = get_skymaps_dict(dataset, self.exclude_regions, self.correlation_radius, self.correlate_off, 'all')
        
        significance_all = self.maps["significance_all"]

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
        self.maps["significance_all"].plot(ax=ax4, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')
        
        ax5.set_title("Off significance map")
        #significance_map.plot(ax=ax1, add_cbar=True, stretch="linear")
        self.maps["significance_off"].plot(ax=ax5, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')

        ax6.set_title("Excess map")
        self.maps["excess"].plot(ax=ax6, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')
        
        # Background and counts

        ax2.set_title("Background map")
        self.maps["background"].plot(ax=ax2, add_cbar=True, stretch="linear")

        ax3.set_title("Counts map")
        self.maps["counts"].plot(ax=ax3, add_cbar=True, stretch="linear")
        
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
        significance_all = self.maps["significance_all"].data[np.isfinite(self.maps["significance_all"].data)]
        significance_off = self.maps["significance_off"].data[np.logical_and(np.isfinite(self.maps["significance_all"].data), 
                                                                self.exclusion_mask.data)]
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
        fig.savefig(f"{fig_save_path[:-4]}_sigma_residuals.png", dpi=300, transparent=False, bbox_inches='tight')

    def plot_skymaps(self, bkg_method='ring', stacked_dataset=None, axis_info_dataset=None, axis_info_map=None):
        if axis_info_dataset is None: axis_info_dataset = self.axis_info_dataset
        if axis_info_map is None: axis_info_map = self.axis_info_map
        cutout = (axis_info_dataset[2] != axis_info_map[2]) & (axis_info_dataset[2] > axis_info_map[2])
        if stacked_dataset is None: self.stacked_dataset = self.get_stacked_dataset(bkg_method=bkg_method, axis_info=axis_info_dataset)
        self.plot_lima_maps(self.stacked_dataset, axis_info_map, cutout, bkg_method)
