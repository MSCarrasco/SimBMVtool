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
from scipy.stats import chi2
from scipy.stats import norm as norm_stats
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
                    irf_dict = load_irf_dict_from_file(dir_path+'/'+str(irf_files[iobs]))
                    if (obs_collection[iobs].aeff is None) & ('aeff' in irf_dict): obs_collection[iobs].aeff = irf_dict['aeff']
                    if (obs_collection[iobs].edisp is None) & ('edisp' in irf_dict): obs_collection[iobs].edisp = irf_dict['edisp']
                    if (obs_collection[iobs].psf is None) & ('psf' in irf_dict): obs_collection[iobs].psf = irf_dict['psf']
                    if (obs_collection[iobs].bkg is None) & ('bkg' in irf_dict): obs_collection[iobs].bkg = irf_dict['bkg']
                    if (obs_collection[iobs].gti is None) & ('gti' in irf_dict): obs_collection[iobs].bkg = irf_dict['gti']
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

def get_geom(Bkg_irf, axis_info, pointing_radec):
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

def compute_residuals(out, true, residuals='diff/true',res_lim_for_nan=1.) -> np.array:
    res_arr = np.zeros_like(true)
    for iEbin in range(true.shape[0]):
        diff = out[iEbin,:,:] - true[iEbin,:,:]
        true[iEbin,:,:][true[iEbin,:,:] == 0.0] = np.nan
        res = diff / true[iEbin,:,:]
        diff[np.abs(res) >= res_lim_for_nan] = np.nan # Limit to mask bins with "almost zero" statistics for truth model and 0 output count
        # Caveat: this will also mask very large bias, but the pattern should be visibly different

        if residuals == "diff": res = diff
        elif residuals == "diff/true": res = diff / true[iEbin,:,:]
        elif residuals == "diff/sqrt(true)": res = diff / np.sqrt(true[iEbin,:,:])
        elif residuals == "diff/sqrt(out)": res = diff / np.sqrt(out[iEbin,:,:])

        # res[true[iEbin,:,:] == 0] = np.nan
        res_arr[iEbin,:,:] += res
    return res_arr

#-------------------------------------------------------------------------------------
# High level analysis
#-------------------------------------------------------------------------------------

def sigma_to_ts(sigma, df=1):
    """Convert sigma to delta ts"""
    p_value = 2 * norm_stats.sf(sigma)
    return chi2.isf(p_value, df=df)

def ts_to_sigma(ts, df=1):
    """Convert delta ts to sigma"""
    p_value = chi2.sf(ts, df=df)
    return norm_stats.isf(0.5 * p_value)

def is_nested_model(null,alt):
    null_free_params = pd.Series(null.parameters.free_parameters.to_table()['name'])
    alt_free_params = pd.Series(alt.parameters.free_parameters.to_table()['name'])
    if 'alpha' in null_free_params.values: null_free_params.loc[null_free_params == 'alpha'] = 'index'
    if 'alpha' in alt_free_params.values: alt_free_params.loc[alt_free_params == 'alpha'] = 'index'
    return null_free_params.isin(alt_free_params).all()

def get_nested_model_wilk_significance(fitted_null_model, fitted_alternative_model, return_dof=True):
    D = fitted_null_model.total_stat - fitted_alternative_model.total_stat
    null_free_parameters = pd.Series(fitted_null_model.parameters.free_parameters.to_table()['name'])
    alt_free_parameters = pd.Series(fitted_alternative_model.parameters.free_parameters.to_table()['name'])
    is_nested = is_nested_model(fitted_null_model, fitted_alternative_model)
    if is_nested:
        delta_dof = len(alt_free_parameters) - len(null_free_parameters)
        wilk_sig = ts_to_sigma(D, delta_dof)
        if return_dof: return wilk_sig, delta_dof
        else: return wilk_sig
    else:
        if return_dof: return -1, -1
        else: return -1

def get_dfmodels_wilk(results:dict, bkg_method='ring', fit_method='stacked'):
    models_list = list(results[bkg_method]['results'].keys())
    dfmodels_0 = pd.DataFrame(index=pd.Index(models_list, name='null_model'), columns=pd.Index(models_list, name='alt_model'))
    dfmodels_wilk = dfmodels_0.copy()

    for null_model, alt_model in product(models_list, models_list):
        fitted_null = results[bkg_method]['results'][null_model][fit_method]
        fitted_alt = results[bkg_method]['results'][alt_model][fit_method]
        fit_fail = (("fit_success" in fitted_null.keys()) and not fitted_null["fit_success"]) or (("fit_success" in fitted_alt.keys()) and not fitted_alt["fit_success"])
        if fit_fail:
            dfmodels_wilk.loc[null_model, alt_model] = -1
            continue
        
        null_tail, null_head = null_model.split(' - ')
        alt_tail, alt_head = alt_model.split(' - ')

        if null_head == 'No source': is_nested_head = True
        elif alt_head == 'No source': is_nested_head = False
        else:
            is_nested_head = is_nested_model(fitted_null['models'][null_head + ' head'],fitted_alt['models'][alt_head + ' head'])
        
        if null_tail == 'No source': is_nested_tail = True
        elif alt_tail == 'No source': is_nested_tail = False
        else:
            is_nested_tail = is_nested_model(fitted_null['models'][null_tail + ' tail'],fitted_alt['models'][alt_tail + ' tail'])

        if is_nested_head and is_nested_tail:
            dfmodels_wilk.loc[null_model, alt_model] = get_nested_model_wilk_significance(fitted_null['fit_result'], fitted_alt['fit_result'])
    return dfmodels_wilk

def get_dfmodels_wilk_dof(results:dict, bkg_method='ring', fit_method='stacked'):
    models_list = list(results[bkg_method]['results'].keys())
    dfmodels_0 = pd.DataFrame(index=pd.Index(models_list, name='null_model'), columns=pd.MultiIndex.from_product([["wilk","delta_dof"],models_list], names=['stat','alt_model']))
    dfmodels_wilk_dof = dfmodels_0.copy()

    for null_model, alt_model in product(models_list, models_list):
        fitted_null = results[bkg_method]['results'][null_model][fit_method]
        fitted_alt = results[bkg_method]['results'][alt_model][fit_method]
        fit_fail = (("fit_success" in fitted_null.keys()) and not fitted_null["fit_success"]) or (("fit_success" in fitted_alt.keys()) and not fitted_alt["fit_success"])
        if fit_fail:
            dfmodels_wilk_dof.loc[null_model, ("wilk", alt_model)] = -1
            continue
        null_tail, null_head = null_model.split(' - ')[:2]
        alt_tail, alt_head = alt_model.split(' - ')[:2]

        if null_head == 'No source': is_nested_head = True
        elif alt_head == 'No source': is_nested_head = False
        else:
            is_nested_head = is_nested_model(fitted_null['models'][null_head + ' head'],fitted_alt['models'][alt_head + ' head'])
        
        if null_tail == 'No source': is_nested_tail = True
        elif alt_tail == 'No source': is_nested_tail = False
        else:
            is_nested_tail = is_nested_model(fitted_null['models'][null_tail + ' tail'],fitted_alt['models'][alt_tail + ' tail'])

        if is_nested_head and is_nested_tail:
            if 'PSR' in null_model:
                is_nested_psr = True if 'PSR' in alt_model else False
                if not is_nested_psr: continue
            wilk_sig, delta_dof = get_nested_model_wilk_significance(fitted_null['fit_result'], fitted_alt['fit_result'])
            dfmodels_wilk_dof.loc[null_model,  ("wilk", alt_model)] = wilk_sig
            dfmodels_wilk_dof.loc[null_model,  ("delta_dof", alt_model)] = delta_dof
    return dfmodels_wilk_dof

def plot_dfmodels_wilk(dfmodels_wilk:pd.DataFrame, fontsize=15, rotation=45, figsize = (20,20), fig_ax=(None,None), shrink=0.7, labels=["","",""], annot=True, square=False):
    title, xlabel, ylabel = labels
    if fig_ax == (None,None): fig,ax = plt.subplots(figsize=figsize)
    else: fig, ax = fig_ax
    dfplot = dfmodels_wilk.where(dfmodels_wilk >= 0, np.nan).copy()
    c = ["darkred","red","lightcoral", "gold", "yellow", "palegreen", "lightgreen","green","darkgreen"]
    vmax=dfplot.max().max()
    v3sig = 3./vmax
    v5sig = 5./vmax
    v = [0, v3sig-0.5*v3sig, v3sig-0.001*v3sig, v3sig, v3sig+0.999*(v5sig-v3sig), v5sig, v5sig+0.33*(1-v5sig), v5sig+0.66*(1-v5sig),1.]
    l = list(zip(v,c))
    cmap=LinearSegmentedColormap.from_list('rg',l, N=256*2)
    sns.heatmap(dfplot.T.astype(float), annot=annot,annot_kws={'fontweight':'bold','fontsize':fontsize-2}, cbar=True, vmin=0, vmax=vmax,cmap=cmap, square=square, mask=dfplot.T.isna(),cbar_kws={"shrink": shrink, 'label': "significance [$\sigma$]"},ax=ax)

    ax.invert_yaxis()
    ax.set_xlabel(f"Null model" if xlabel == "" else xlabel, fontsize=fontsize)
    ax.set_ylabel(f"Alternative model" if ylabel == "" else ylabel, fontsize=fontsize)
    xticks = ax.get_xticklabels()
    yticks = ax.get_yticklabels()
    ax.set_xticklabels(xticks, rotation=rotation, fontsize=fontsize-3, ha='right')
    ax.set_yticklabels(yticks, rotation=0, fontsize=fontsize-3, ha='right')
    ax.grid(True, alpha=0.1)
    ax.set_title(label="Wilk significance matrix\n" if title == "" else title, fontsize=fontsize+2)
    plt.tight_layout()
    plt.show()

def get_dfbest_models(dfmodels_wilk:pd.DataFrame, results:dict, bkg_method='ring', fit_method='stacked', rel_L_tsh=1, sig_method='wilk'):
    models_list = list(results[bkg_method]['results'].keys())
    dfmodels = pd.DataFrame(index=pd.Index(models_list,name='model'))
    models_to_remove = []
    for tested_model in models_list:
        fitted_model = results[bkg_method]['results'][tested_model][fit_method]
        if (("fit_success" in fitted_model.keys()) and not fitted_model["fit_success"]):
            dfmodels.loc[tested_model, "AIC"] = 1e3
            dfmodels.loc[tested_model, "AIC"] = 1e3
            models_to_remove.append(tested_model)
            continue
        fitted_model_res = fitted_model['fit_result']
        dfmodels.loc[tested_model, "AIC"] = 2*len(fitted_model_res.parameters.free_parameters) + fitted_model_res.total_stat
        # dfmodels['rel_L'] = np.exp((dfmodels.AIC.min() - dfmodels.AIC) * 0.5)
        dfmodels.loc[tested_model, 'wilk'] = dfmodels_wilk.loc['No source - No source', tested_model]

    if sig_method == 'wilk': best_models = dfmodels[~dfmodels.index.isin(models_to_remove)].index.to_list()
    else:
        best_models = dfmodels[~dfmodels.index.isin(models_to_remove) & (dfmodels.rel_L > np.exp(-rel_L_tsh))].sort_values(by="AIC",ascending=True).index.to_list()
    
    models_to_keep = []

    for ibest in range(len(best_models)):
        tested_model = best_models[ibest]
        print(f"Tested model: {tested_model}")
        
        tested_model_wilk_as_null = dfmodels_wilk.loc[tested_model].copy()
        tested_model_wilk_as_alt = dfmodels_wilk[tested_model].copy()

        
        better_alt_models = tested_model_wilk_as_null.loc[(tested_model_wilk_as_null > 3) & (tested_model_wilk_as_null != np.nan)]
        better_null_models = tested_model_wilk_as_alt.loc[(tested_model_wilk_as_alt < 3) & (tested_model_wilk_as_alt >= 0) & (tested_model_wilk_as_alt != np.nan)]

        worse_alt_models = tested_model_wilk_as_null.loc[(tested_model_wilk_as_null <= 3) & (tested_model_wilk_as_null >= 0) & (tested_model_wilk_as_null != np.nan)]
        worse_null_models = tested_model_wilk_as_alt.loc[(tested_model_wilk_as_alt >= 3) & (tested_model_wilk_as_alt != np.nan)]
        
        if len(better_alt_models) > 0: better_alt_str = f"{len(better_alt_models)} more significant nested models"
        else: better_alt_str = "No nested model is more significant"
        if len(better_null_models) > 0: better_null_str = f"Model not more significant than {len(better_null_models)} nesting models"
        else: better_null_str = "More significant than all nesting models"

        if (len(better_alt_models) > 0) or (len(better_null_models) > 0):
            models_to_remove.append(tested_model)
            print(f"{better_null_str}\n{better_alt_str}\nModel removed\n")
        else:
            models_to_keep.append(tested_model)
            print(f"{better_null_str}\n{better_alt_str}\nModel kept\n")

    dfbest_models = dfmodels.reset_index()
    dfbest_models = dfbest_models[dfbest_models.model.isin(models_to_keep)]
    dfbest_models = dfbest_models[~dfbest_models.model.isin(models_to_remove)][["model", "wilk", "AIC"]]
    dfbest_models['relative_L'] = np.exp((dfbest_models.AIC.min() - dfbest_models.AIC) * 0.5)
    
    return dfmodels, dfbest_models.sort_values(by="AIC")

def get_plot_kwargs_from_models_dict(models_dict):    
    plot_kwargs = models_dict["plot_kwargs"].copy()
    plot_kwargs["energy_bounds"] = plot_kwargs["energy_bounds"] * u.Unit(plot_kwargs["energy_bounds_unit"])
    plot_kwargs.pop("energy_bounds_unit", None)
    plot_kwargs["yunits"] = u.Unit(plot_kwargs["yunits"])
    return plot_kwargs

def plot_ref(ax, ref_models, ref_source_name, ref_model_name, plot_type = "spectral"):
    ref_source_dict = ref_models[ref_source_name]["published"][ref_model_name]

    ref_source_model_dict=ref_source_dict["models"].copy()
    ref_source_model_dict["name"] = ref_model_name
    ref_source_model = SkyModel.from_dict(ref_source_model_dict)
    plot_kwargs = get_plot_kwargs_from_models_dict(ref_source_dict)

    if plot_type == "spectral":
        plot_kwargs.pop("color_sigma", None)
        plot_kwargs.pop("ls_sigma", None)
        ref_source_model.spectral_model.plot(ax=ax, **plot_kwargs)
        plot_kwargs.pop("label", None)
        edgecolor = plot_kwargs["color"]
        plot_kwargs.pop("color", None)
        ref_source_model.spectral_model.plot_error(ax=ax, alpha=0.1, linewidth=1, edgecolor=edgecolor, facecolor="none", hatch='/', **plot_kwargs)
    elif plot_type=='spatial':
        center = SkyCoord(ra=ref_source_model.spatial_model.lon_0.value * u.deg,dec=ref_source_model.spatial_model.lat_0.value * u.deg, frame="icrs")
        sigma = ref_source_model.spatial_model.sigma.value
        r = SphericalCircle(center, sigma * u.deg,
                                edgecolor=plot_kwargs["color_sigma"], facecolor='none',
                                lw = 2,
                                ls = plot_kwargs["ls_sigma"],
                                transform=ax.get_transform('icrs'), label=plot_kwargs["label"])
        ax.add_patch(r)

def plot_spectra_from_models_dict(ax, results, ref_models, ref_source_name, ref_models_to_plot, results_to_plot=['all'], bkg_methods_to_plot=['all'], fit_methods_to_plot=['all'], colors = ['blue', 'darkorange', 'purple', 'green'], plot_flux_points=True):

    for ref_model_name in ref_models_to_plot:
        plot_ref(ax, ref_models, ref_source_name, ref_model_name, plot_type = "spectral")
    
    bkg_methods = ['ring', 'FoV'] if bkg_methods_to_plot == ['all'] else bkg_methods_to_plot
    fit_methods = ['stacked', 'joint'] if fit_methods_to_plot == ['all'] else fit_methods_to_plot
    tested_models = list(results[bkg_methods[0]]['results'].keys()) if results_to_plot == ['all'] else results_to_plot
    i=0
    for bkg_method in bkg_methods:
        for tested_model in tested_models:
            for fit_method in fit_methods:
                if (len(results[bkg_method]['results'][tested_model][fit_method]) == 0): continue
                results_model = results[bkg_method]['results'][tested_model][fit_method]['models'].copy()
                for model in results_model[:-1]:
                    model_name = model.name
                    # print(model_name)
                    flux_points = results[bkg_method]['results'][tested_model][fit_method][model_name]["flux_points"]
                    i_color = 0 if 'tail' in model_name else 1
                    plot_kwargs = {
                        "energy_bounds": [flux_points.energy_min[flux_points.success.data.flatten()][0],flux_points.energy_max[flux_points.success.data.flatten()][-1]] * u.TeV,
                        "sed_type": "e2dnde",
                        "ls" : "-",
                        "color": colors[i_color],
                        "yunits": u.Unit("TeV cm-2 s-1"),
                    }
                    label = f"{model_name} ({bkg_method}, {fit_method})"
                    if plot_flux_points:
                        model.spectral_model.plot(ax=ax,**plot_kwargs.copy())
                        model.spectral_model.plot_error(ax=ax, alpha=0.1, facecolor=colors[i_color], **plot_kwargs.copy())
                        flux_points.plot(ax=ax, sed_type="e2dnde", label=label, color=colors[i_color], marker='o' if 'tail' in model_name else "x", markersize=5 if 'tail' in model_name else 7)
                    else:
                        model.spectral_model.plot(ax=ax,**plot_kwargs.copy(), label=label)
                        model.spectral_model.plot_error(ax=ax, alpha=0.1, facecolor=colors[i_color], **plot_kwargs.copy())
                i+=1

def plot_spatial_model_from_dict(bkg_method, key, results, ref_models, ref_source_name, ref_models_to_plot, results_to_plot=['all'], bkg_methods_to_plot=['all'], fit_methods_to_plot=['all'], crop_width=0 * u.deg, estimator='excess', ring_bkg_param=None, figsize=(5,5),bbox_to_anchor=(1,1),fontsize=15, colors = ['blue', 'darkorange', 'purple', 'green']):
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
        },
        'ts': {
            'cbar_label': 'TS',
            'title': 'TS map'
        },
        'flux': {
            'cbar_label': 'flux [$s^{-1}cm^{-2}$]',
            'title': f'Flux map ({bkg_method}, {estimator})\nwith 3$\sigma$ (white) and 5$\sigma$ (black) contour'
        }
    }

    skymaps_dict = results[bkg_method][f'skymaps_{estimator}']
    skymap = skymaps_dict[key]
    skymap_sig = skymaps_dict["significance_all"]
    if crop_width != 0 * u.deg:
        width = 0.5 * skymap.geom.width[0][0]
        binsize = width / (0.5 * skymap.data.shape[-1])
        n_crop_px = int(((width - crop_width)/binsize).value)
        skymap = skymap.crop(n_crop_px)
        skymap_sig = skymap_sig.crop(n_crop_px)
    
    cbar_label, title = (skymaps_args[key]["cbar_label"], skymaps_args[key]["title"])

    fig,ax=plt.subplots(figsize=figsize,subplot_kw={"projection": skymap.geom.wcs})
    if key in ['counts', 'background']: skymap.smooth(0.01 * u.deg).plot(ax=ax, add_cbar=True, stretch="linear",kwargs_colorbar={'label': cbar_label, 'shrink':0.75})
    elif key in ['excess', 'ts', 'flux']:
        skymap.smooth(0.01 * u.deg).plot(ax=ax, add_cbar=True, stretch="linear", cmap='Greys_r')
        im = ax.images        
        cb = im[-1].colorbar
        cb.set_label(cbar_label)
    elif 'significance' in key: 
        skymap.smooth(0.01 * u.deg).plot(ax=ax, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma',kwargs_colorbar={'label': cbar_label, 'shrink':0.75})
        maxsig = np.nanmax(skymap.data)
        minsig = np.nanmin(skymap.data)
        im = ax.images        
        cb = im[-1].colorbar 
        cb.ax.axhline(maxsig, c='g')
        cb.ax.text(1.1,maxsig - 0.07,' max', color = 'g')
        cb.ax.axhline(minsig, c='g')
        cb.ax.text(1.1,minsig - 0.07,' min', color = 'g')
    
    ax.contour(skymap_sig.data[0], levels=[3,5], colors=['white', 'black'], alpha=0.5)

    for ref_model_name in ref_models_to_plot:
        plot_ref(ax, ref_models, ref_source_name, ref_model_name, plot_type = "spatial")
    
    both_bkg_methods = bkg_methods_to_plot == ['all']
    both_fit_methods = fit_methods_to_plot == ['all']

    bkg_methods = ['ring', 'FoV'] if both_bkg_methods else bkg_methods_to_plot
    fit_methods = ['stacked', 'joint'] if fit_methods_to_plot == ['all'] else fit_methods_to_plot
    tested_models = list(results[bkg_methods[0]]['results'].keys()) if results_to_plot == ['all'] else results_to_plot

    i=0

    for bkg_method in bkg_methods:
        for tested_model in tested_models:
            for fit_method in fit_methods:
                fitted_null_model = results[bkg_method]['results']['No source - No source'][fit_method]["fit_result"]
                label_methods = " ("* (both_bkg_methods or both_fit_methods) + bkg_method * both_bkg_methods +","*(both_bkg_methods and both_fit_methods)+ fit_method * both_fit_methods +")"* (both_bkg_methods or both_fit_methods)

                if (len(results[bkg_method]['results'][tested_model][fit_method]) == 0): continue
                results_model = results[bkg_method]['results'][tested_model][fit_method]['models'].copy()
                j=0
                for model in results_model[:-1]:
                    model_name = model.name
                    center = SkyCoord(ra=model.spatial_model.lon_0.value * u.deg,dec=model.spatial_model.lat_0.value * u.deg, frame="icrs")
                    label=f"{model_name}{label_methods}"
                    i_color = 0 if 'tail' in model_name else 1
                    if j==0:
                        fitted_alternative_model = results[bkg_method]['results'][tested_model][fit_method]['fit_result']
                        wilk_sig = get_nested_model_wilk_significance(fitted_null_model, fitted_alternative_model, False)
                        label_wilk= ", $\sqrt{TS}$" + f"={wilk_sig:.2f}"
                    
                    if "Point" in model_name:
                        r = SphericalCircle(center, 0.04 * u.deg,
                                            edgecolor='black', facecolor=colors[i_color],
                                            ls = "-",
                                            lw = 1,
                                            transform=ax.get_transform('icrs'), label=label + label_wilk*(j==0))
                        ax.add_patch(r)
                    elif "Gauss" in model_name:
                        sigma = model.spatial_model.sigma.value
                        if "1D" in model_name:
                            label += f": $\sigma$={sigma:.2f}°"
                        
                            r = SphericalCircle(center, sigma * u.deg,
                                                edgecolor=colors[i_color], facecolor='none',
                                                ls = "-",
                                                lw = 3,
                                                transform=ax.get_transform('icrs'), label=label + label_wilk*(j==0))
                            ax.add_patch(r)
                        else:
                            sky_region = model.spatial_model.to_region(x_sigma=1.)
                            pixel_region = sky_region.to_pixel(skymap.geom.wcs)
                            label += ": $\sigma_{eff}$"+f"={sigma:.2f}°"
                            pixel_region.plot(ax=ax,
                                            color = colors[i_color],
                                            lw=3,
                                            ls = "-",
                                            label=label + label_wilk*(j==0))
                    elif ("Disk" in model_name) or ("Ellipse" in model_name):
                        sky_region = model.spatial_model.to_region()
                        pixel_region = sky_region.to_pixel(skymap.geom.wcs)
                        if "Disk" in model_name:
                            r_0 = model.spatial_model.r_0.value
                            label += f": $r_0$={r_0:.2f}°"
                        else:
                            width_ellipse = sky_region.width.value
                            height_ellipse = sky_region.height.value
                            label += f": width,heigth={width_ellipse:.2f}°,{height_ellipse:.2f}°"

                        pixel_region.plot(ax=ax,
                                        color = colors[i_color],
                                        lw=3,
                                        ls = "-",
                                        label=label + label_wilk*(j==0))
                    j+=1
                i+=1
    pulsar_pos = SkyCoord.from_name('PSR J2229+6114')
    if ref_source_name == "boomerang":
        ax.errorbar(pulsar_pos.ra.value, pulsar_pos.dec.value, fmt='o', ms=5, color='lightgreen', label = 'PSR J2229+6114',
            transform=ax.get_transform('icrs'))
    ax.set_title(label=title,fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.show()

def get_on_region_geom(results:dict, tested_model_name:str, bkg_method='ring', fit_method='stacked', i_component=0, plot=False):
    skymap = results[bkg_method]['skymaps_excess']['flux']
    energy_axis = skymap.geom.axes['energy']
    tested_model = results[bkg_method]['results'][tested_model_name][fit_method]['models'][i_component].copy()
    model_name = tested_model_name.split(' - ')[i_component]
    label = model_name
    if plot:
        fig,ax=plt.subplots(figsize=(7,5),subplot_kw={"projection": skymap.geom.wcs})
        colors = ["red"]
        skymap.smooth(0.01 * u.deg).plot(ax=ax, add_cbar=True, stretch="linear", cmap='Greys_r')
        i_color=0

    if "Gauss" in model_name:
        sigma = tested_model.spatial_model.sigma.value
        sky_region = tested_model.spatial_model.to_region(x_sigma=1.)
        on_geom = RegionGeom(sky_region, axes=[energy_axis])
        pixel_region = sky_region.to_pixel(results['ring']['skymaps_excess']['ts'].geom.wcs)

        if "1D" in model_name: label += f": $\sigma$={sigma:.2f}°"
        else: label += ": $\sigma_{eff}$"+f"={sigma:.2f}°"
    
    elif ("Disk" in model_name) or ("Ellipse" in model_name):
        sky_region = tested_model.spatial_model.to_region()
        on_geom = RegionGeom(sky_region, axes=[energy_axis])
        pixel_region = sky_region.to_pixel(results['ring']['skymaps_excess']['ts'].geom.wcs)
        
        if "Disk" in model_name:
            r_0 = tested_model.spatial_model.r_0.value
            label += f": $r_0$={r_0:.2f}°"
        else:
            width_ellipse = sky_region.width.value
            height_ellipse = sky_region.height.value
            label += f": width,heigth={width_ellipse:.2f}°,{height_ellipse:.2f}°"

    if plot:
        pixel_region.plot(ax=ax,
                    color = colors[i_color],
                    lw=1,
                    ls = "-",
                    label=label)
        ax.set(title=f"Flux map")
        plt.legend(loc='upper left', title = 'on region')
        plt.tight_layout()
        plt.show()
    return on_geom

def get_map_residuals_spatial(
    stacked_dataset,
    ax=None,
    method="diff",
    smooth_kernel="gauss",
    smooth_radius="0.1 deg",
    **kwargs,
):
    """Plot spatial residuals.

    The normalization used for the residuals computation can be controlled
    using the method parameter.

    Parameters
    ----------
    ax : `~astropy.visualization.wcsaxes.WCSAxes`
        Axes to plot on.
    method : {"diff", "diff/model", "diff/sqrt(model)"}
        Normalization used to compute the residuals, see `MapDataset.residuals`.
    smooth_kernel : {"gauss", "box"}
        Kernel shape.
    smooth_radius: `~astropy.units.Quantity`, str or float
        Smoothing width given as quantity or float. If a float is given, it
        is interpreted as smoothing width in pixels.
    **kwargs : dict
        Keyword arguments passed to `~matplotlib.axes.Axes.imshow`.

    Returns
    -------
    ax : `~astropy.visualization.wcsaxes.WCSAxes`
        WCSAxes object.

    Examples
    --------
    >>> from gammapy.datasets import MapDataset
    >>> dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
    >>> kwargs = {"cmap": "RdBu_r", "vmin":-5, "vmax":5, "add_cbar": True}
    >>> dataset.plot_residuals_spatial(method="diff/sqrt(model)", **kwargs) # doctest: +SKIP
    """
    counts, npred = stacked_dataset.counts.copy(), stacked_dataset.npred()

    if counts.geom.is_region:
        raise ValueError("Cannot plot spatial residuals for RegionNDMap")

    if stacked_dataset.mask is not None:
        counts *= stacked_dataset.mask
        npred *= stacked_dataset.mask

    counts_spatial = counts.sum_over_axes().smooth(
        width=smooth_radius, kernel=smooth_kernel
    )
    npred_spatial = npred.sum_over_axes().smooth(
        width=smooth_radius, kernel=smooth_kernel
    )
    residuals = stacked_dataset._compute_residuals(counts_spatial, npred_spatial, method)

    if stacked_dataset.mask_safe is not None:
        mask = stacked_dataset.mask_safe.reduce_over_axes(func=np.logical_or, keepdims=True)
        residuals.data[~mask.data] = np.nan
    return residuals

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
