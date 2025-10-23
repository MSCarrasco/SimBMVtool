import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord,EarthLocation, angular_separation, position_angle,Angle
from astropy.time import Time
from astropy.visualization.wcsaxes import SphericalCircle
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, LinearSegmentedColormap

from copy import deepcopy
import pandas as pd
from pathlib import Path
import os, pathlib, yaml
import pickle as pk
import seaborn as sns
from itertools import product

# %matplotlib inline

from gammapy.data import FixedPointingInfo, Observations, Observation, PointingMode, ObservationTable
from gammapy.datasets import MapDataset, MapDatasetOnOff
from gammapy.irf import load_irf_dict_from_file, Background2D, Background3D, FoVAlignment
from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, WcsGeom, WcsNDMap
from regions import CircleAnnulusSkyRegion, CircleSkyRegion, Regions
from gammapy.estimators import ExcessMapEstimator, TSMapEstimator
from gammapy.maps.region.geom import RegionGeom
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
from gammapy.catalog import SourceCatalogGammaCat

from scipy.special import erfcinv, erfinv
from scipy.stats import chi2, mode
from scipy.stats import norm as norm_stats
from scipy.optimize import curve_fit
from math import sqrt

from gammapy.stats import CashCountsStatistic
from gammapy.modeling import Parameter, Parameters

import logging
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------------
# Data management
#-------------------------------------------------------------------------------------

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

def load_yaml(save_path:str):
    '''Read yaml file'''
    path = Path(save_path)
    if path.exists():
        with open(save_path, 'r') as file:
            yaml_file = yaml.safe_load(file)
            return yaml_file
    else: 
        print('Error : No yaml file found at '+ os.fspath(path))
        return path.exists()

#-------------------------------------------------------------------------------------
# Observations
#-------------------------------------------------------------------------------------

def get_data_store(dir_path, pattern, from_index=False):
    if from_index:
        data_store = DataStore.from_dir(f"{dir_path}",hdu_table_filename=f'hdu-index{pattern}.fits.gz',obs_table_filename=f'obs-index{pattern}.fits.gz')
    else:
        path = Path(dir_path)
        paths = sorted(list(path.rglob(pattern)))
        data_store = DataStore.from_events_files(paths)
    
    for i in range(len(data_store.hdu_table)):
        if data_store.hdu_table["HDU_NAME"][i] == 'POINT SPREAD FUNCTION': data_store.hdu_table["HDU_NAME"][i] = 'PSF'
        if data_store.hdu_table["HDU_CLASS"][i] == 'psf_3gauss': data_store.hdu_table["HDU_CLASS"][i] = 'psf_table'
    
    return data_store

def get_dfobs_table(obs_table:ObservationTable):
    str_columns = ['OBJECT', 'OBS_MODE', 'TELLIST', 'INSTRUME']
    datetime_columns = ['DATE-OBS', 'TIME-OBS', 'DATE-END', 'TIME-END']
    datetime_formats = ["%Y-%m-%d","%H:%M:%S.%f","%Y-%m-%d","%H:%M:%S.%f"]
    dfobs_table = obs_table.to_pandas().set_index("OBS_ID")
    for col in str_columns: 
        if col in dfobs_table.columns:
            if (dfobs_table[col].apply(lambda x: isinstance(x, bytes)).all()): dfobs_table[col] = dfobs_table[col].str.decode('utf-8')
    for col,format in zip(datetime_columns,datetime_formats): 
        if col in dfobs_table.columns:
            if 'TIME' in col:
                if (dfobs_table[col].apply(lambda x: isinstance(x, bytes)).all()):
                    col_strings = dfobs_table[col].str.decode('utf-8')
                    if not np.any(col_strings == 'NOT AVAILABLE'):
                        dfobs_table[col] = pd.to_datetime(col_strings,format=format).dt.time
            elif 'DATE' in col:
                if (dfobs_table[col].apply(lambda x: isinstance(x, bytes)).all()):
                    col_strings = dfobs_table[col].str.decode('utf-8')
                    if not np.any(col_strings == 'NOT AVAILABLE'):
                        dfobs_table[col] = pd.to_datetime(col_strings,format=format).dt.date
    return dfobs_table

def get_obs_collection(dir_path,pattern,multiple_simulation_subdir=False,from_index=False,with_datastore=True, obs_ids=np.array([])):
    path = Path(dir_path)
    paths = sorted(list(path.rglob(pattern)))
    obs_ids = None if (obs_ids.shape[0]==0) else obs_ids
    if not multiple_simulation_subdir:
        if not from_index:
            try: data_store = get_data_store(dir_path, '', from_index=True)
            except: data_store = get_data_store(dir_path=dir_path, pattern=pattern, from_index=from_index)
        else:  data_store = get_data_store(dir_path=dir_path, pattern=pattern, from_index=from_index)
        obs_collection = data_store.get_observations(obs_id=obs_ids, required_irf=['aeff','edisp'])
        
        # Because DL3s from different experiments can have different names for hdus we need to check if some which are missing can be retrieved
        file_dirs = data_store.hdu_table['FILE_DIR'].astype(str).data
        for irf_filename in ['IRF_FILENAME','FILE_NAME']:
            if irf_filename in data_store.obs_table.columns:
                irf_files = data_store.obs_table[irf_filename].astype(str).data
                break
            elif irf_filename in data_store.hdu_table.columns:
                irf_files = data_store.hdu_table[irf_filename].astype(str).data
                break
        if obs_ids is None: obs_ids = np.array(data_store.obs_table['OBS_ID'].data)
        for iobs, obs_id in enumerate(obs_ids):
            if (obs_collection[iobs].aeff is None) or (obs_collection[iobs].edisp is None) or (obs_collection[iobs].psf is None) or (obs_collection[iobs].bkg is None) or (obs_collection[iobs].gti is None):
                try:
                    irf_dict_path = str(irf_files[iobs]) if dir_path in str(irf_files[iobs]) else dir_path+'/'+ str(irf_files[iobs])
                    irf_dict = load_irf_dict_from_file(irf_dict_path)
                    if (obs_collection[iobs].aeff is None) & ('aeff' in irf_dict): obs_collection[iobs].aeff = deepcopy(irf_dict['aeff'])
                    if (obs_collection[iobs].edisp is None) & ('edisp' in irf_dict): obs_collection[iobs].edisp = deepcopy(irf_dict['edisp'])
                    if (obs_collection[iobs].psf is None) & ('psf' in irf_dict): obs_collection[iobs].psf = deepcopy(irf_dict['psf'])
                    if (obs_collection[iobs].bkg is None) & ('bkg' in irf_dict): obs_collection[iobs].bkg = deepcopy(irf_dict['bkg'])
                    if (obs_collection[iobs].gti is None) & ('gti' in irf_dict): obs_collection[iobs].bkg = deepcopy(irf_dict['gti'])
                except:
                    logger.info('Error when loading irf dictionnary')
    else:
        # This handles the case where multiple observations have the same obs_id
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

#-------------------------------------------------------------------------------------
# Simulation
#-------------------------------------------------------------------------------------

def get_empty_obs_simu(Bkg_irf, axis_info, run_info, src_models, path_data:str,flux_to_0=True, t_ref_str="2000-01-01 00:00:00", t_delay=0,verbose=False):
    '''Loads irf from file and return a simulated observation with its associated dataset'''

    loc, source_pos, run, livetime, pointing, file_name = run_info

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

def get_empty_dataset_and_obs_simu(Bkg_irf, axis_info, run_info, src_models, path_data:str,flux_to_0=True, t_ref_str="2000-01-01 00:00:00", t_delay=0, fov_rotation_error_limit=1 * u.deg, verbose=False):
    '''Loads irf from file and return a simulated observation with its associated dataset'''

    loc, source_pos, run, livetime, pointing, file_name = run_info
    # print(livetime)
    
    if axis_info is not None: 
        e_min, e_max, nbin_E, offset_max, width = axis_info
        energy_axis = MapAxis.from_energy_edges(
            np.logspace(np.log10(e_min.to_value(u.TeV)), np.log10(e_max.to_value(u.TeV)), nbin_E)*u.TeV,
            name="energy",
        )
        binsz = 0.02
        npix = (int(width[0]/binsz), int(width[1]/binsz))
    else:
        energy_axis = Bkg_irf.axes[0]
        e_min = energy_axis.edges.min()
        e_max = energy_axis.edges.max()
        offset_axis = Bkg_irf.axes[1]
        offset_max = offset_axis.edges.max()
        nbins_map = offset_axis.nbin
        npix = (nbins_map, nbins_map)
        width = offset_max.to_value(u.deg) * 2
        binsz = width / nbins_map
    
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

    # Create the geometry with the additional axes
    geom = WcsGeom.create(
        skydir=(pointing.ra.degree,pointing.dec.degree),
        binsz=binsz, 
        npix=npix,
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

    maps=["exposure", "background", "edisp", "psf"]
    if hasattr(MapDatasetMaker(),'fov_rotation_error_limit'):
        maker = MapDatasetMaker(selection=maps, fov_rotation_error_limit=fov_rotation_error_limit)
    else:
        maker = MapDatasetMaker(selection=maps)
    
    dataset = maker.run(empty, obs)
    if verbose: print(obs.obs_info)
    if verbose: print(dataset)

    # Models
    try: spatial_model = src_models.spatial_model() # If source models from Catalog object
    except: spatial_model = src_models.spatial_model    # If source models from SkyRegion object

    # Uncomment if you want to use the true source spectral model. Here we declare a model with an amplitude of 0 cm-2 s-1 TeV-1 to simulate only background
    if not flux_to_0: 
        try: spectral_model = src_models.spectral_model()
        except: spectral_model = src_models.spectral_model
    else: spectral_model = LogParabolaSpectralModel(
            alpha=2,beta=0.2 / np.log(10), amplitude=0. * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
    )

    model_simu = SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        name=src_models.name,
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

#-------------------------------------------------------------------------------------
# Skymaps
#-------------------------------------------------------------------------------------

def get_regions_from_dict(regions_dict:dict):
    '''Takes a dictionnary with SkyRegion parameters and returns list of regions
    Supported regions: circle, ellipse, rectangle
    Note that BAccMod doesn't handle rectangle regions yet'''
    regions_ds9 = ""
    n_regions = len(regions_dict)
    for iregion, region_key in enumerate(regions_dict):
        shape = region_key.split("_")[0]
        region = regions_dict[region_key]
        if shape == "circle": regions_ds9 += f"icrs;circle({region['ra']}, {region['dec']}, {region['radius']})"
        elif shape == "ellipse": regions_ds9 += f"icrs;ellipse({region['ra']}, {region['dec']}, {0.5*region['width']}, {0.5*region['height']}, {0.5*region['angle']})"
        elif shape == "rectangle": regions_ds9 += f"icrs;box({region['ra']}, {region['dec']}, {region['width']}, {region['height']}, {region['angle']})"
        if iregion < n_regions - 1: regions_ds9 += ";"
    return Regions.parse(regions_ds9, format="ds9")

def get_geom(Bkg_irf, axis_info=None, pointing_radec=None):
    '''Return geom from bkg_irf axis or a set of given axis edges'''
    
    if axis_info is not None: 
        e_min, e_max, nbin_E, offset_max, width = axis_info
        energy_axis = MapAxis.from_energy_edges(
            np.logspace(np.log10(e_min.to_value(u.TeV)), np.log10(e_max.to_value(u.TeV)), nbin_E)*u.TeV,
            name="energy",
        )
        binsz = 0.02
        npix = (int(width[0]/binsz), int(width[1]/binsz))
    else:
        energy_axis = Bkg_irf.axes[0]
        e_min = energy_axis.edges.min()
        e_max = energy_axis.edges.max()
        offset_axis = Bkg_irf.axes[1]
        offset_max = offset_axis.edges.max()
        nbins_map = offset_axis.nbin
        npix = (nbins_map, nbins_map)
        width = offset_max.to_value(u.deg) * 2
        binsz = width / nbins_map
    
    if not isinstance(pointing_radec, SkyCoord): pointing_radec= SkyCoord(pointing_radec[0] * u.degree, pointing_radec[1] * u.degree)
    
    geom = WcsGeom.create(
            skydir=(pointing_radec.ra.degree, pointing_radec.dec.degree),
            binsz=binsz,
            npix=npix,
            frame='icrs',
            proj="CAR",
            axes=[energy_axis],
        )
    return geom

def get_exclusion_mask_from_dataset_geom(dataset, exclude_regions):
    geom_map = dataset.geoms['geom']
    energy_axis_map = dataset.geoms['geom'].axes[0]
    geom_image = geom_map.to_image().to_cube([energy_axis_map.squash()])
    exclusion_mask = geom_image.region_mask(exclude_regions, inside=False)
    return exclusion_mask

def get_lima_maps(stacked_dataset, correlation_radius, correlate_off, estimator='excess', model_source=None):
    if estimator=='excess': map_estimator = ExcessMapEstimator(correlation_radius*u.deg, correlate_off=correlate_off)
    elif estimator=='ts':
        # if isinstance(stacked_dataset, MapDatasetOnOff): stacked_dataset = stacked_dataset.to_map_dataset()
        map_estimator = TSMapEstimator(
                                        model_source,
                                        kernel_width=correlation_radius * u.deg,
                                        energy_edges=stacked_dataset._geom.axes['energy'].bounds,
                                    )
    try: lima_maps = map_estimator.run(stacked_dataset)
    except:
        stacked_dataset=stacked_dataset.to_map_dataset()
        stacked_dataset.model = model_source
        lima_maps = map_estimator.run(stacked_dataset)
    return lima_maps

def get_dataset_maps_dict(dataset, results=['counts','background']):
    maps = {}
    if 'counts' in results: maps['counts'] = dataset.counts.sum_over_axes()
    if 'background' in results: maps['background'] = dataset.background.sum_over_axes()
    return maps

def get_high_level_maps_dict(lima_maps, exclusion_mask, exclusion_mask_not_source=None, results=['significance_all']):
    '''results='all', ['significance_all','significance_off','excess','ts','flux']'''
    if results == 'all': results = ['significance_all','significance_off','excess','ts','flux']
    
    ts_map = lima_maps["ts"]
    significance_map = lima_maps["sqrt_ts"]
    excess_map = lima_maps["npred_excess"]
    flux_map = lima_maps["flux"]
    
    maps = {}
    if 'significance_all' in results: maps['significance_all'] = significance_map if exclusion_mask_not_source is None else significance_map * exclusion_mask_not_source
    if 'significance_off' in results: maps['significance_off'] = significance_map * exclusion_mask
    if 'excess' in results: maps['excess'] = excess_map
    if 'ts' in results: maps['ts'] = ts_map
    if 'flux' in results: maps['flux'] = flux_map

    return maps

def get_high_level_maps_from_dataset(dataset, exclude_regions, exclude_regions_not_source, correlation_radius, correlate_off, results, estimator='excess', model_source=None):
    exclusion_mask = get_exclusion_mask_from_dataset_geom(dataset, exclude_regions)
    exclusion_mask_not_source = get_exclusion_mask_from_dataset_geom(dataset, exclude_regions_not_source)
    lima_maps = get_lima_maps(dataset, correlation_radius, correlate_off, estimator, model_source)
    maps = get_high_level_maps_dict(lima_maps, exclusion_mask, exclusion_mask_not_source, results)
    return maps

def get_skymaps_dict(dataset, exclude_regions, exclude_regions_not_source, correlation_radius, correlate_off, results, estimator='excess', model_source=None):
    '''results='all', ['counts', 'background','significance_all','significance_off','excess','ts','flux']'''
    
    if results == 'all': results=['counts', 'background', 'significance_all', 'significance_off', 'excess', 'ts', 'flux']
    
    dataset_bool = 1*(('counts' in results) or ('background' in results))
    estimator_bool = -1*(('significance_all' in results) or ('significance_off' in results) or ('excess' in results) or ('ts' in results) or ('flux' in results))
    i_method = dataset_bool + estimator_bool # methods = {1: 'dataset_only', -1: 'estimator_only', 0: 'both'}
    
    if i_method >= 0: maps_dataset = get_dataset_maps_dict(dataset, results)
    if i_method <= 0: maps_high_level = get_high_level_maps_from_dataset(dataset, exclude_regions, exclude_regions_not_source, correlation_radius, correlate_off, results, estimator, model_source)

    if i_method==0:  
        maps_both = maps_dataset.copy()
        maps_both.update(maps_high_level)
        return maps_both
    elif i_method==1: return maps_dataset
    else: return maps_high_level

def plot_skymap_from_dict(skymaps, key, crop_width=0 * u.deg, ring_bkg_param=None, figsize=(5,5)):
    skymaps_args = {
        'counts': {
            'cbar_label': 'events',
            'title': 'Counts map'
        },
        'background': {
            'cbar_label': 'events',
            'title': 'Background map'
        },
        'significance_all': {
            'cbar_label': 'significance [$\sigma = \sqrt{TS}$]',
            'title': 'Significance map'
        },
        'significance_off': {
            'cbar_label': 'significance [$\sigma = \sqrt{TS}$]',
            'title': 'Off significance map'
        },
        'excess': {
            'cbar_label': 'events',
            'title': 'Excess map'
        },
        'ts': {
            'cbar_label': 'TS',
            'title': 'TS map'
        },
        'flux': {
            'cbar_label': 'flux [$s^{-1}cm^{-2}$]',
            'title': 'Flux map'
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
    if key in ['counts', 'background']:
        skymap.plot(ax=ax, add_cbar=False, stretch="linear")
        im = ax.images[-1]
        cax = fig.add_axes([ax.get_position().x1+0.05,ax.get_position().y0,0.04,ax.get_position().height])
        plt.colorbar(im, cax=cax, label=cbar_label) # Similar to fig.colorbar(im, cax = cax)
    elif key in ['excess', 'ts', 'flux']:
        skymap.plot(ax=ax, add_cbar=False, stretch="linear", cmap='magma')
        im = ax.images[-1]
        cax = fig.add_axes([ax.get_position().x1+0.05,ax.get_position().y0,0.04,ax.get_position().height])
        plt.colorbar(im, cax=cax, label=cbar_label) # Similar to fig.colorbar(im, cax = cax)
    elif 'significance' in key: 
        skymap.plot(ax=ax, add_cbar=False, stretch="linear",norm=CenteredNorm(), cmap='magma')
        im = ax.images[-1]
        cax = fig.add_axes([ax.get_position().x1+0.05,ax.get_position().y0,0.04,ax.get_position().height])
        plt.colorbar(im, cax=cax, label=cbar_label) # Similar to fig.colorbar(im, cax = cax)
        ax.contour(skymap.data[0], levels=[3,5], colors=['white', 'red'], alpha=0.5)
        maxsig = np.nanmax(skymap.data)
        minsig = np.nanmin(skymap.data)
        cax.axhline(maxsig, c='g')
        cax.text(1.1,maxsig - 0.07,' max', color = 'g')
        cax.axhline(minsig, c='g')
        cax.text(1.1,minsig - 0.07,' min', color = 'g')
        
    if (ring_bkg_param is not None):
            if hasattr(ring_bkg_param,'__len__') & (len(ring_bkg_param)==2):
                int_rad, width = ring_bkg_param
                ring_center_pos = SkyCoord(ra=skymap._geom.center_coord[0],dec=skymap._geom.center_coord[1],frame='icrs')
                r2 = SphericalCircle(ring_center_pos, int_rad * u.deg,
                                    edgecolor='white', facecolor='none',
                                    transform=ax.get_transform('icrs'))
                r3 = SphericalCircle(ring_center_pos, int_rad * u.deg + width * u.deg,
                                    edgecolor='white', facecolor='none',
                                    transform=ax.get_transform('icrs'))
                ax.add_patch(r2)
                ax.add_patch(r3)
    ax.set(title=title)
    plt.tight_layout()
    plt.show()

def plot_significance_residuals(xlabel, lima_maps, title="", figsize=(11, 5), fontsize=15, stat='density', n_bins=30, n_sigma=3, width_left = 1.8, cutout_width=1.7*u.deg, save_plot=False, return_minmax=True):
    skymap = lima_maps["sqrt_ts"].cutout(lima_maps["sqrt_ts"]._geom.center_skydir, cutout_width)
    
    fig, (ax1,ax) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [width_left, 3-width_left]})
    fig.delaxes(ax1)
    ax1 = fig.add_axes([0.1, 0.1, 0.8*width_left/3, 0.8], projection=skymap.geom.wcs)
    skymap.plot(cmap="coolwarm", add_cbar=True, vmin=-5, vmax=5, ax=ax1)
    ax1.set_title(label='Significance residuals ', fontsize=fontsize-1)
    residuals_distrib = lima_maps["sqrt_ts"].data.flatten()
    distribution = residuals_distrib[~np.isnan(residuals_distrib) & (residuals_distrib != 0)]
    
    # Compute histogram
    hist, bin_edges = np.histogram(distribution, bins=n_bins, density=True)  # density=True scales the histogram to density
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initial guesses for parameters [A, mu, sigma]
    p0 = [hist.max(), np.mean(distribution), np.std(distribution)]
    
    # Fit the Gaussian to the histogram
    popt, pcov = curve_fit(gauss, bin_centers, hist, p0=p0)
    psigma = np.sqrt(np.diag(pcov))
    
    # Extract parameters from the fit
    A = popt[0]
    mu = popt[1]
    sigma = np.sqrt(popt[2] ** 2)

    # Generate Gaussian fit data
    xdata = np.linspace(bin_edges.min(), bin_edges.max(), 1000)  # Higher resolution for plotting
    y_gauss = gauss(xdata, *popt)  # Compute Gaussian values
    
    # Normalize Gaussian fit to match histogram area
    norm_factor = np.trapz(hist, bin_centers)  # Calculate area under the histogram (integral)
    y_gauss *= norm_factor  # Scale Gaussian fit to match area under the histogram
    
    # Define the range for containment based on the number of sigmas
    min_val = mu - n_sigma * sigma
    max_val = mu + n_sigma * sigma

    # Plotting
    # fig, ax = plt.subplots(figsize=(4, 4))
    
    # Plot the standard normal distribution for comparison
    xdata_norm = np.linspace(bin_edges.min(), bin_edges.max(), 1000)
    y_gauss_norm = norm_stats.pdf(xdata_norm)  # Standard normal distribution
    ax.plot(xdata_norm, y_gauss_norm, color='midnightblue', ls='--', label="Standard normal distribution")
    
    # Plot the Gaussian fit to the histogram
    ax.plot(xdata, y_gauss, color='crimson', label=f"Best-fit normal distribution\n$\mu$={mu:.2f}, $\sigma$={sigma:.2f}")
    
    # Plot the histogram
    sns.histplot(x=distribution, bins=n_bins, ax=ax, element='step', fill=False, stat=stat, color='cornflowerblue', label='Distribution')
    
    # Set plot labels and title
    ax.set_xlabel(xlabel=xlabel, fontsize=fontsize-2)
    ax.set_ylabel(ylabel=stat.capitalize(), fontsize=fontsize-2)
    
    ax.legend(loc='upper center', frameon=False, fontsize=fontsize-1)
    ax.set(ylim=[0, 0.6])
    ax.set_title(label='Fitted distribution', fontsize=fontsize-1)
    plt.suptitle(t=title, fontsize=fontsize)
    # Save plot if needed
    if save_plot:
        fig.savefig(f'/gaussfit_{xlabel}.png')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    if return_minmax: return min_val, max_val

#-------------------------------------------------------------------------------------
# Model validation
#-------------------------------------------------------------------------------------

def plot_residuals_histogram(maps:dict, exclude_regions = None):
    exclusion_mask =  maps["significance_all"].geom.region_mask(exclude_regions, inside=False)
    exclusion_mask_not_source =  maps["significance_all"].geom.region_mask(exclude_regions[-1], inside=False)
    significance_all = maps["significance_all"].data[np.logical_and(np.isfinite(maps["significance_all"].data), 
                                                            exclusion_mask_not_source.data)]
    significance_off = maps["significance_off"].data[np.logical_and(np.isfinite(maps["significance_all"].data), 
                                                            exclusion_mask.data)]
    emin_map, emax_map = significance_all.geom.axes['energy'].bounds
    
    fig, ax1 = plt.subplots(figsize=(4,4))
    ax1.hist(
        significance_all[significance_all != 0],
        range=(-8,8),
        density=True,
        alpha=0.5,
        color="red",
        label="all bins",
        bins=30,
    )

    ax1.hist(
        significance_off[significance_off != 0],
        range=(-8,8),
        density=True,
        alpha=0.5,
        color="blue",
        label="off bins",
        bins=30,
    )

    # Now, fit the off distribution with a Gaussian
    mu, std = norm_stats.fit(significance_off[significance_off != 0])
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
    plt.show()

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

def compute_residuals(out, true, residuals='diff/true',res_lim_for_nan=1., norm_factor=1, dfobs_stat=None) -> np.array:
    res_arr = np.zeros_like(true)
    
    for iEbin in range(true.shape[0]):
        count_tot = dfobs_stat["count"].to_numpy()[iEbin] if dfobs_stat is not None else norm_factor
        flux_tot = np.nansum(true[iEbin,:,:].flatten()) if dfobs_stat is not None else 1
        diff = out[iEbin,:,:] - true[iEbin,:,:]
        true[iEbin,:,:][true[iEbin,:,:] == 0.0] = np.nan
        res = diff / true[iEbin,:,:]
        N = count_tot / flux_tot
        diff[np.abs(res) >= res_lim_for_nan] = np.nan # Limit to mask bins with "almost zero" statistics for truth model and 0 output count
        # Caveat: this will also mask very large bias, but the pattern should be visibly different

        if residuals == "diff": res = diff
        elif residuals == "diff/true": res = diff / true[iEbin,:,:]
        elif residuals == "diff/sqrt(true)": res = diff / np.sqrt(true[iEbin,:,:])
        elif residuals == "diff/sqrt(out)": res = diff / np.sqrt(out[iEbin,:,:])

        # res[true[iEbin,:,:] == 0] = np.nan
        res_arr[iEbin,:,:] += res * np.sqrt(N)
    return res_arr

def distance(x, y, x0, y0):
    """
    Return distance between point
    P[x0,y0] and a curve (x,y)
    """
    d_x = x - x0
    d_y = y - y0
    dis = np.sqrt( d_x**2 + d_y**2 )
    return dis

def min_distance(x, y, P, precision=30):
    """
    Compute minimum/a distance/s between
    a point P[x0,y0] and a curve (x,y)
    rounded at `precision`.
    
    ARGS:
        x, y      (array)
        P         (tuple)
        precision (int)
        
    Returns min indexes and distances array.
    """
    # compute distance
    d = distance(x, y, P[0], P[1])
    d = np.round(d, precision)
    # find the minima
    glob_min_idxs = np.argwhere(d==np.min(d)).ravel()
    sign = []
    for idx in glob_min_idxs:
        sign.append(np.sign(P[1] - y[idx]))
    return glob_min_idxs, sign, d

def linear_func(x, a, b):
        return a * x + b

def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def get_gaussian_containment(xlabel,distribution,bins=30,n_sigma=1,weight_hist=1,  ax=None,save_plot=True, plot=True):
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    hist, bin_edges = np.histogram(distribution,bins)
    # p0 = [hist.max(), np.mean(distribution), np.std(distribution)]
    p0 = [hist.max(), mode(np.round(distribution,3),keepdims=False,nan_policy='omit')[0], np.std(distribution)]
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    try:
        popt, pcov = curve_fit(gauss, bin_centers, hist, p0=p0)
        psigma = np.sqrt(np.diag(pcov))
        print(popt,psigma)

        xdata = np.linspace(bin_edges.min(),bin_edges.max(),100)
        y_gauss = gauss(xdata,*popt)

        min = popt[1]-n_sigma*sqrt(popt[2]**2)
        max = popt[1]+n_sigma*sqrt(popt[2]**2)

        if plot:
            if ax is None: fig,ax=plt.subplots(figsize=(5,5))
            ax.fill_betweenx(y=[0,np.max([y_gauss.max(),hist.max()])],x1=min,x2=max,color='g',alpha=0.1,label=f'$\pm${n_sigma}$\sigma$')
            sns.histplot(x=distribution,bins=bins,ax=ax, element='step',weights=weight_hist)
            ax.plot(xdata,y_gauss,color='r')
            ax.set(xlabel=xlabel)
            ax.legend(loc='best')
        # if save_plot: fig.savefig(f'{save_path_plots}/gaussfit_{xlabel}')

        return min, max, sqrt(popt[2]**2)
    except:
        return np.nan,np.nan, np.nan

def calculate_pcc(true, pred, mask=None):
    if mask is None:
        mask = (~np.isnan(true)) & (~np.isnan(pred))
    else:
        mask = mask & (~np.isnan(true)) & (~np.isnan(pred))
    if np.sum(mask) == 0:
        return np.nan
    return pearsonr(true[mask], pred[mask])[0]

def calculate_nrmse(true, pred, mask=None):
    if mask is None:
        mask = (~np.isnan(true)) & (~np.isnan(pred))
    else:
        mask = mask & (~np.isnan(true)) & (~np.isnan(pred))
    if np.sum(mask) == 0:
        return np.nan
    mse = np.mean((true[mask] - pred[mask])**2)
    rmse = np.sqrt(mse)
    norm = np.mean(true[mask])
    return rmse / norm

#-------------------------------------------------------------------------------------
# High level analysis
#-------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------
# Misc
#-------------------------------------------------------------------------------------

def scale_value(x,xlim,ylim):
    y_min = ylim[0]
    y_max = ylim[1]
    x_min = xlim[0]
    x_max = xlim[1]
    
    # Apply the linear mapping formula
    y = (x - x_min) * ((y_max - y_min) / (x_max - x_min)) + y_min
    
    return y
