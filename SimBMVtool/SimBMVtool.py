from gammapy.data import Observations, PointingMode
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
from regions import CircleAnnulusSkyRegion, CircleSkyRegion, SkyRegion, Regions

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
from gammapy.catalog import SourceCatalogGammaCat

from scipy.stats import norm as norm_stats
from gammapy.stats import CashCountsStatistic
from gammapy.modeling import Parameter, Parameters
from gammapy.utils.compat import COPY_IF_NEEDED

from itertools import product
from gammapy.catalog import SourceCatalogGammaCat
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

def get_run_info(path_data:str, obs_id:int):
    """returns array with [livetime,pointing_radec,pointing_altaz]"""
    loc = EarthLocation.of_site('Roque de los Muchachos')
    data_store = DataStore.from_dir(f"{path_data}",hdu_table_filename=f'hdu-index.fits.gz',obs_table_filename=f'obs-index.fits.gz')
    obs_table = data_store.obs_table
    livetime = obs_table[obs_table["OBS_ID"] == obs_id]["LIVETIME"].data[0]
    ra = obs_table[obs_table["OBS_ID"] == obs_id]["RA_PNT"].data[0]
    dec = obs_table[obs_table["OBS_ID"] == obs_id]["DEC_PNT"].data[0]
    alt = obs_table[obs_table["OBS_ID"] == obs_id]["ALT_PNT"].data[0]
    az = obs_table[obs_table["OBS_ID"] == obs_id]["AZ_PNT"].data[0]
    pointing= SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
    pointing_altaz = AltAz(alt=alt * u.deg, az=az * u.deg,location=loc)
    print(f"--Run {obs_id}--\nlivetime: {livetime}\npointing radec: {pointing}\npointing altaz: {pointing_altaz}")
    return livetime, pointing, pointing_altaz

# +
# Les functions modifiées (pas proprement)

def get_empty_obs_simu(Bkg_irf, axis_info, run_info, src_models, run_dir:str,flux_to_0=True, t_ref_str="2000-01-01 00:00:00", t_delay=0,verbose=False):
    '''Loads irf from file and return a simulated observation with its associated dataset'''

    loc, source_pos, run, livetime, pointing, pointing_altaz = run_info
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
        f"{run_dir}/dl3_LST-1.Run{'0'*(len(str(run))==4)}{run}.fits"
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

def get_empty_dataset_and_obs_simu(Bkg_irf, axis_info, run_info, src_models, run_dir:str,flux_to_0=True, t_ref_str="2000-01-01 00:00:00", t_delay=0,verbose=False):
    '''Loads irf from file and return a simulated observation with its associated dataset'''

    loc, source_pos, run, livetime, pointing, pointing_altaz = run_info
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
        f"{run_dir}/dl3_LST-1.Run{'0'*(len(str(run))==4)}{run}.fits"
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

    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"], fov_rotation_error_limit=1 * u.deg)
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

def get_stacked_dataset(observations,source_info,exclusion_region,axis_info, bkg_method=None, bkg_param = None):

    source,source_pos,source_region = source_info
    emin,emax,offset_max,nbin_offset = axis_info

    exclusion = exclusion_region is not None

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
    # maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=2.*offset_max)
    
    for obs in observations:
        dataset_map = maker.run(base_map_dataset.copy(), obs)
        # dataset_map = maker_safe_mask.run(dataset_map, obs)
        unstacked_datasets.append(dataset_map)

    # Stack the datasets
    unstacked_datasets_local = unstacked_datasets.copy()

    ## Make the exclusion mask
    if exclusion: exclusion_mask = geom_image.region_mask(exclusion_region, inside=False)
    else: exclusion_mask=None
    
    ## Make the MapDatasetOnOff

    if bkg_method == 'ring': 
        internal_ring_radius,width_ring = bkg_param
        ring_bkg_maker = RingBackgroundMaker(r_in=internal_ring_radius,
                                                        width=width_ring,
                                                        exclusion_mask=exclusion_mask)
        stacked_on_off = MapDatasetOnOff.create(geom=geom_image, energy_axis_true=energy_axis_true, name="stacked")
        print("Ring background method is applied")
    else: 
        stacked_on_off = Datasets()
        if bkg_method == 'FoV':
            FoV_background_maker = FoVBackgroundMaker(method=bkg_param, exclusion_mask=exclusion_mask)
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

def plot_bkg_irf_residuals_at_energy(
        bkg_model,bkg_true, energy=None,max_residual_percent=100, max_fov=2.5*u.deg, add_cbar=True, ncols=3, figsize=None, **kwargs
    ):
        """Plot the residual background rate in Field of view coordinates at a given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            list of Energy
        add_cbar : bool
            Add color bar?
        ncols : int
            Number of columns to plot
        figsize : tuple
            Figure size
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`.
        """
        n = len(energy)
        cols = min(ncols, n)
        rows = 1 + (n - 1) // cols
        width = 12
        cfraction = 0.0
        if add_cbar:
            cfraction = 0.15
        if figsize is None:
            figsize = (width, rows * width // (cols * (1 + cfraction)))

        fig, axes = plt.subplots(
            ncols=cols,
            nrows=rows,
            figsize=figsize,
            gridspec_kw={"hspace": 0.2, "wspace": 0.3},
        )
        fig.suptitle("Background rates residuals")
        if isinstance(bkg_model,Background2D): bkg_model = bkg_model.to_3d()
        if isinstance(bkg_true,Background2D): bkg_true = bkg_true.to_3d()
        x = bkg_model.axes["fov_lat"].edges
        y = bkg_model.axes["fov_lon"].edges
        X, Y = np.meshgrid(x, y)

        for i, ee in enumerate(energy):
            if len(energy) == 1:
                ax = axes
            else:
                ax = axes.flat[i]
            bkg = 100*(bkg_model.evaluate(energy=ee)-bkg_true.evaluate(energy=ee))/bkg_true.evaluate(energy=ee)
            with quantity_support():
                caxes = ax.pcolormesh(X, Y, bkg.squeeze(), norm=CenteredNorm(halfrange=max_residual_percent), cmap='coolwarm', **kwargs)

            bkg_model.axes["fov_lat"].format_plot_xaxis(ax)
            bkg_model.axes["fov_lon"].format_plot_yaxis(ax)
            ax.set_title(str(ee))
            if add_cbar:
                label = f"(output - true) / true [%]"
                cbar = ax.figure.colorbar(caxes, ax=ax, label=label, fraction=cfraction,shrink=1.)
                # cbar.formatter.set_powerlimits((0, 0))

            row, col = np.unravel_index(i, shape=(rows, cols))
            if col > 0:
                ax.set_ylabel("")
            if row < rows - 1:
                ax.set_xlabel("")
            ax.set_xlim([-max_fov,max_fov])
            ax.set_ylim([-max_fov,max_fov])
            ax.set_aspect("equal", "box")

def plot_lima_maps(dataset,correlation_radius,source_pos,exclusion_region,exclusion_radius,method,ring_bkg_param,axis_info,cutout=False):
    emin_map,emax_map,offset_max = axis_info
    exclusion = exclusion_region is not None
    is_ring_bkg = ring_bkg_param is not None
    if is_ring_bkg: internal_ring_radius,width_ring = ring_bkg_param
    if cutout: dataset = dataset.cutout(position = source_pos, width=offset_max)
    
    geom_map = dataset.geoms['geom']
    energy_axis_map = dataset.geoms['geom'].axes[0]
    geom_image = geom_map.to_image().to_cube([energy_axis_map.squash()])

    # Make the exclusion mask
    if exclusion: exclusion_mask = geom_image.region_mask(exclusion_region, inside=False)

    # Get the maps
    estimator = ExcessMapEstimator(correlation_radius*u.deg, correlate_off=True)
    lima_maps = estimator.run(dataset)
    significance_map = lima_maps["sqrt_ts"]
    excess_map = lima_maps["npred_excess"]
    # Significance and excess
    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(
        figsize=(18, 11),subplot_kw={"projection": lima_maps.geom.wcs}, ncols=3, nrows=2
    )
    # fig.delaxes(ax1)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    ax1.set_title("Spatial residuals map: diff/sqrt(model)")
    g = dataset.plot_residuals_spatial(method='diff/sqrt(model)',ax=ax1, add_cbar=True, stretch="linear",norm=CenteredNorm())
    # plt.colorbar(g,ax=ax1, shrink=1, label='diff/sqrt(model)')
    
    ax4.set_title("Significance map")
    #significance_map.plot(ax=ax1, add_cbar=True, stretch="linear")
    significance_map.plot(ax=ax4, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')

    if exclusion: 
        significance_map_off = significance_map * exclusion_mask
        ax5.set_title("Off significance map")
        #significance_map.plot(ax=ax1, add_cbar=True, stretch="linear")
        significance_map_off.plot(ax=ax5, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')

    ax6.set_title("Excess map")
    excess_map.plot(ax=ax6, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')
    
    # Background and counts

    ax2.set_title("Background map")
    dataset.background.sum_over_axes().plot(ax=ax2, add_cbar=True, stretch="linear")

    ax3.set_title("Counts map")
    dataset.counts.sum_over_axes().plot(ax=ax3, add_cbar=True, stretch="linear")
    
    if is_ring_bkg:
        if cutout: ring_center_pos = (source_pos.ra + 0.5*offset_max*u.deg, source_pos.dec + 0.5*offset_max*u.deg)
        else: (source_pos.ra + offset_max*u.deg, source_pos.dec + offset_max*u.deg)
        r1 = SphericalCircle((source_pos.ra, source_pos.dec), exclusion_radius * u.deg,
                        edgecolor='yellow', facecolor='none',
                        transform=ax6.get_transform('icrs'))
        r2 = SphericalCircle(ring_center_pos, internal_ring_radius * u.deg,
                            edgecolor='white', facecolor='none',
                            transform=ax6.get_transform('icrs'))
        r3 = SphericalCircle(ring_center_pos, internal_ring_radius * u.deg + width_ring * u.deg,
                            edgecolor='white', facecolor='none',
                            transform=ax6.get_transform('icrs'))
        ax2.add_patch(r2)
        ax2.add_patch(r1)
        ax2.add_patch(r3)
    plt.tight_layout()

    # Residuals

    significance_all = significance_map.data[np.isfinite(significance_map.data)]
    if exclusion: 
        significance_map_off = significance_map * exclusion_mask
        significance_off = significance_map.data[np.logical_and(np.isfinite(significance_map.data), 
                                                            exclusion_mask.data)]
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

    if exclusion:
        ax1.hist(
            significance_off,
            range=(-8,8),
            density=True,
            alpha=0.5,
            color="blue",
            label="off bins",
            bins=30,
        )

    if exclusion: significance=significance_off
    else: significance=significance_all
    # Now, fit the off distribution with a Gaussian
    mu, std = norm_stats.fit(significance)
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

def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(int(n / i))
    for divisor in reversed(large_divisors):
        yield divisor

def plot_lima_off_residuals_map(dataset,dataset_true,correlation_radius,source_pos,exclusion_region,axis_info,halfrange=100,cutout=False):
    emin_map,emax_map,_ = axis_info
    dataset_true=dataset_true.to_image()
    if cutout: 
        dataset = dataset.cutout(position = source_pos, width=2.)
        dataset_true = dataset_true.cutout(position = source_pos, width=2.)
   
    geom_map = dataset.geoms['geom']
    energy_axis_map = dataset.geoms['geom'].axes[0]
    geom_image = geom_map.to_image().to_cube([energy_axis_map.squash()])

    # Make the exclusion mask
    exclusion_mask = geom_image.region_mask(exclusion_region, inside=False)

    # Get the maps
    estimator = ExcessMapEstimator(correlation_radius*u.deg, correlate_off=True)

    lima_maps = estimator.run(dataset)
    significance_map = lima_maps["sqrt_ts"]
    significance_map_off = significance_map * exclusion_mask

    geom_map = dataset_true.geoms['geom']
    energy_axis_map = dataset_true.geoms['geom'].axes[0]
    geom_image = geom_map.to_image().to_cube([energy_axis_map.squash()])

    # Make the exclusion mask
    exclusion_mask = geom_image.region_mask(exclusion_region, inside=False)
    lima_maps_true = estimator.run(dataset_true)
    significance_map_true = lima_maps_true["sqrt_ts"]
    significance_map_off_true = significance_map_true * exclusion_mask

    # Significance and excess
    fig, (ax1,ax2,ax3) = plt.subplots(
        figsize=(17, 5),subplot_kw={"projection": lima_maps.geom.wcs}, ncols=3, nrows=1
    )
    fig.suptitle("Off significance maps")
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    ax1.set_title("Model")
    significance_map_off_true.plot(ax=ax1, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')

    ax2.set_title("Data")
    significance_map_off.plot(ax=ax2, add_cbar=True, stretch="linear",norm=CenteredNorm(), cmap='magma')

    # We are not sure wether this plot means anything, but it exists
    ax3.set_title("Residuals: (output - true) / true [%]")
    # Due to the very large amount of pixel, the maps are downsampled
    # divisors = np.array(list(divisorGenerator(significance_map_off.geom.npix[0][0])))
    # down_factor = divisors[-5]
    # significance_map_residuals = significance_map_off.downsample(down_factor)
    # significance_map_residuals.data = 100*(significance_map_off.downsample(down_factor).data-significance_map_off_true.downsample(down_factor).data)/significance_map_off_true.downsample(down_factor).data
    significance_map_residuals = significance_map_off.copy()
    significance_map_residuals.data = 100*(significance_map_off.data-significance_map_off_true.data)/significance_map_off_true.data
    significance_map_residuals.plot(ax=ax3, add_cbar=True, norm=CenteredNorm(halfrange=halfrange), stretch="linear", cmap='coolwarm')

    plt.tight_layout()

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

def get_map(data, energy_axis, offset_axis, center_radec = (0.,0.)):
    max_offset = np.max(offset_axis.edges)
    n_bins_map = data.shape[1]
    spatial_bin_size = max_offset / (n_bins_map / 2)
    center_map = SkyCoord(ra=center_radec[0] * u.deg, dec=center_radec[1] * u.deg, frame='icrs')
    geom = WcsGeom.create(skydir=center_map, npix=(n_bins_map, n_bins_map),
                                binsz=spatial_bin_size, frame="icrs", axes=[energy_axis])
    map = WcsNDMap(data=data,geom=geom,unit=u.Unit("sr-1 s-1 TeV-1"))

    return map

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
        obs_collection = data_store.get_observations()
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

def get_dfprofile(bkg_irf):
    E_centers = bkg_irf.axes["energy"].center.value.round(1)
    fov_edges = bkg_irf.axes['fov_lon'].edges.value
    fov_centers = bkg_irf.axes['fov_lon'].center.value.round(1)
    offset_edges = fov_edges[fov_edges >= 0]
    offset_centers = fov_centers[fov_centers >= 0]

    dfcoord = pd.DataFrame(index=pd.Index(fov_centers,name='fov_lon'),columns=pd.Index(fov_centers,name='fov_lat'))
    for (lon,lat) in product(dfcoord.index,dfcoord.columns): dfcoord.loc[lon,lat] = np.sqrt(lon**2 + lat**2)
    dfcoord = dfcoord.apply(pd.to_numeric, errors='coerce')
    dfdata = pd.DataFrame(index=pd.Index(fov_centers,name='fov_lon'),columns=pd.MultiIndex.from_product([E_centers,fov_centers],names=['E','fov_lat']))
    # Check and convert if necessary (accmodel output seems to be problematic)
    if bkg_irf.data.dtype.byteorder == '>':
        bkg_irf.data = bkg_irf.data.view(bkg_irf.data.dtype.newbyteorder('S'))

    for iEbin,Ebin in enumerate(E_centers):
        dfdata[(Ebin.value)] = bkg_irf.data[iEbin,:,:]
        dfprofile_lat = dfdata[(Ebin.value)].describe()
        dfprofile_lat.loc['sum'] = dfdata[(Ebin.value)].sum()
        dfprofile_lon= dfdata[(Ebin.value)].describe()
        dfprofile_lon.loc['sum'] = dfdata[(Ebin.value)].T.sum()

        for r_mid, r_low, r_high in zip(offset_centers,offset_edges[:-1],offset_edges[1:]):
            values = dfdata[(Ebin.value)][(dfcoord >= r_low) & (dfcoord < r_high)].values
            dfvalues = pd.Series(values[~np.isnan(values)]).describe().to_frame().rename(columns={0:r_mid})
            dfvalues.loc['sum'] = np.nansum(values)
            if r_mid == offset_centers[0]: dfprofile_rad = dfvalues.copy()
            else: dfprofile_rad = dfprofile_rad.join(dfvalues.copy())

        dfprofile_Ebin = pd.concat([dfprofile_lat, dfprofile_lon, dfprofile_rad], keys=['fov_lat','fov_lon','fov_offset'],join='outer')
        dfprofile_Ebin.columns = pd.MultiIndex.from_product([[(Ebin.value)],dfprofile_Ebin.columns])
        if iEbin == 0: dfprofile = dfprofile_Ebin.copy()
        else:
            dfprofile=pd.concat([dfprofile,dfprofile_Ebin],axis=1)
    
    return dfprofile

def plot_acceptances(bkg_true, bkg_out, config, res_type='diff/true', res_lim_for_nan=0.9, title='', fig_save_path='', stack=False):
    '''res_type: none, diff/true, diff/sqrt(true)'''
    emin, emax = float(config["acceptance"]["energy"]["e_min"])*u.TeV, float(config["acceptance"]["energy"]["e_max"])*u.TeV
    size_fov = float(config["acceptance"]["offset"]["offset_max"]) * u.deg
    nbin_E, nbin_offset = config["acceptance"]["energy"]["nbin"],config["acceptance"]["offset"]["nbin"]

    energy_axis = MapAxis.from_energy_bounds(emin, emax, nbin=nbin_E, name='energy')

    fov_max = size_fov.to_value(u.deg)
    fov_lim = [-fov_max,fov_max]
    fov_bin_edges = np.linspace(-fov_max,fov_max,7)
    Ebin_labels = [f"{ebin_min:.1f}-{ebin_max:.1f} {energy_axis.unit}" for ebin_min, ebin_max in zip(energy_axis.edges.value[:-1],energy_axis.edges.value[1:])]

    if bkg_true is not None:
        if isinstance(bkg_true, Background3D): bkg_true = bkg_true.data
        elif bkg_true.shape == (2 * nbin_offset, 2 * nbin_offset, nbin_E): bkg_true = bkg_true.T

    if bkg_out is not None:
        if isinstance(bkg_out, Background3D): bkg_out = bkg_out.data
        elif bkg_out.shape == (2 * nbin_offset, 2 * nbin_offset, nbin_E): bkg_out = bkg_out.T

    rot = 65
    nncols = 3
    n = nbin_E
    cols = min(nncols, n)
    rows = 1 + (n - 1) // cols
    width = 16
    cfraction = 0.15
    res_types = ["none", "diff/true", "diff/sqrt(true)"]

    itype = np.argwhere(np.array(res_types) == res_type)[0]
    if len(itype) > 0: itype = itype[0]
    
    res_type_labels = [" diff / true", r"diff / $\sqrt{true}$"]
    radec_map = 'map' in title
    
    if radec_map: xlabel,ylabel=("Ra offset [°]", "Dec offset [°]")
    else: xlabel,ylabel=("FoV Lat [°]", "FoV Lon [°]")

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(width, rows * width // (cols * (1 + cfraction))))
    res_arr = []
    for iax, ax in enumerate(axs.flat[:n]):
        if res_type != "none":
            res_label = f'{res_type_labels[itype-1]} [%]'
            diff = bkg_out[iax,:,:] - bkg_true[iax,:,:]
            bkg_true[iax,:,:][np.where(bkg_true[iax,:,:] == 0.0)] = np.nan
            res = diff / bkg_true[iax,:,:]
            diff[np.abs(res) >= res_lim_for_nan] = np.nan

            if res_type == "diff/true": res = diff / bkg_true[iax,:,:]
            elif res_type == "diff/sqrt(true)": res = diff / np.sqrt(bkg_true[iax,:,:])
                # res_lim_for_nan = np.sqrt(res_lim_for_nan)
    
            res[bkg_true[iax,:,:] == 0] = np.nan
            res *= 100
            res_arr.append(res)
            vlim = np.nanmax(np.abs(res))
            colorbarticks = np.concatenate((np.flip(-np.logspace(-1,3,5)),[0],np.logspace(-1,3,5)))
            # colorbarticks[np.where(np.abs(colorbarticks) < vlim)]
            map = ax.imshow(res, origin='lower', norm=SymLogNorm(linthresh=0.03, linscale=0.03, vmin=-vlim, vmax=vlim), cmap='coolwarm')
            plt.colorbar(map,ax=ax, shrink=1, ticks=colorbarticks, label=res_label)
        else:
            if bkg_true is not None:
                bkg = bkg_true
                if bkg_out is not None: print("Two bkg IRF were given, only the first one will be plotted")
            elif bkg_out is not None: bkg = bkg_out

            bkg[iax,:,:][np.where(bkg[iax,:,:] == 0.0)] = np.nan
            map = ax.imshow(bkg[iax,:,:], origin='lower', cmap='viridis')

            plt.colorbar(map,ax=ax, shrink=1, label='Background [MeV$^{-1}$s$^{-1}$sr$^{-1}$]')

        x_lim = ax.get_xlim()
        xticks_new = scale_value(fov_bin_edges,fov_lim,x_lim).round(1)
        if radec_map: ax.set_xticks(rotation=rot, ticks=xticks_new, labels=np.flip(fov_bin_edges.round(1)))
        else: ax.set_xticks(rotation=rot, ticks=xticks_new, labels=fov_bin_edges.round(1))
        ax.set_yticks(ticks=xticks_new, labels=fov_bin_edges.round(1))
        ax.set(title=Ebin_labels[iax],xlabel=xlabel,ylabel=ylabel)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    if fig_save_path != '': fig.savefig(fig_save_path, dpi=300, transparent=False, bbox_inches='tight')

    if res_type != "none": 
        fig_hist, ax_hist = plt.subplots(figsize=(4,3))
        res_arr_flat = np.array(res_arr).flatten()
        res_arr_flat = res_arr_flat[~np.isnan(res_arr_flat)]
        res_arr_flat_absmax = np.max(np.abs(res_arr_flat))
        sns.histplot(res_arr_flat, bins=np.linspace(-res_arr_flat_absmax,res_arr_flat_absmax,20), ax=ax_hist, element='step', fill=False, stat='density', color='k',line_kws={'color':'k'})
        ax_hist.set(title='Residuals distribution', xlabel=res_label, yscale='log')
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
        return fig, fig_hist
    else: return fig

def get_bkg_irf_profile_ratio_plot(bkg_true, bkg_out, config, iEbin, ratio_lim=(0.98,1.02),fig_save_path=''):
    emin, emax = float(config["acceptance"]["energy"]["e_min"])*u.TeV, float(config["acceptance"]["energy"]["e_max"])*u.TeV
    size_fov = float(config["acceptance"]["offset"]["offset_max"]) * u.deg
    nbin_E, nbin_offset = config["acceptance"]["energy"]["nbin"],config["acceptance"]["offset"]["nbin"]

    energy_axis = MapAxis.from_energy_bounds(emin, emax, nbin=nbin_E, name='energy')
    offset_axis = MapAxis.from_bounds(0.*u.deg, size_fov, nbin=nbin_offset, name='offset')

    edges = offset_axis.edges
    extended_edges = np.concatenate((-np.flip(edges), edges[1:]), axis=None)
    extended_offset_axis = MapAxis.from_edges(extended_edges, name='fov_coord')

    fov_max = size_fov.to_value(u.deg)
    fov_lim = [-fov_max,fov_max]
    fov_bin_edges = np.linspace(-fov_max,fov_max,7)
    Ebin_labels = [f"{ebin_min:.1f}-{ebin_max:.1f} {energy_axis.unit}" for ebin_min, ebin_max in zip(energy_axis.edges.value[:-1],energy_axis.edges.value[1:])]

    if isinstance(bkg_true, Background3D): bkg_true = bkg_true.data
    elif bkg_true.shape == (2 * nbin_offset, 2 * nbin_offset, nbin_E): bkg_true = bkg_true.T

    if isinstance(bkg_out, Background3D): bkg_out = bkg_out.data
    elif bkg_out.shape == (2 * nbin_offset, 2 * nbin_offset, nbin_E): bkg_out = bkg_out.T
    
    if iEbin != -1:
        array = bkg_true[iEbin,:,:]
        array2 = bkg_out[iEbin,:,:]
        Ebin_label = Ebin_labels[iEbin]
    else:
        array = np.sum(bkg_true, axis=0)
        array2 = np.sum(bkg_out, axis=0)
        Ebin_label = f"{emin.value:.1f}-{emax.value:.1f} {energy_axis.unit}"
    
    array[np.where(array == 0.0)] = np.nan

    row_mean_profile = np.nansum(array, axis=1)
    column_mean_profile = np.nansum(array, axis=0)

    array2[np.where(array2 == 0.0)] = np.nan

    row_mean_profile2 = np.nansum(array2, axis=1)
    column_mean_profile2 = np.nansum(array2, axis=0)
    
    fig, (ax,ax2) = plt.subplots(2,1,figsize=(6,6), gridspec_kw={'height_ratios': [3, 1]})

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fov_bin_edges_plot = extended_offset_axis.edges.to_value(u.deg)
    fov_bin_centers_plot = extended_offset_axis.center.to_value(u.deg)
    ax.set(xlim=[fov_bin_edges_plot[0],fov_bin_edges_plot[-1]])

    lw=1
    ls_true,ls_out = ("--","-")
    m_lat,m_lon = ("o","^")

    sns.lineplot(x=fov_bin_centers_plot, y=row_mean_profile, ax=ax, label='lon true', lw=lw, ls=ls_true, marker=m_lon)
    sns.lineplot(x=fov_bin_centers_plot, y=row_mean_profile2, ax=ax, label='lon out', lw=lw, ls=ls_out, marker=m_lon)
    sns.lineplot(x=fov_bin_centers_plot,y=column_mean_profile, ax=ax, label='lat true', lw=lw, ls=ls_true, marker=m_lat)
    sns.lineplot(x=fov_bin_centers_plot, y=column_mean_profile2, ax=ax, label='lat out', lw=lw, ls=ls_out, marker=m_lat)

    ax.set(yscale='log', title=f'Integrated profiles: {Ebin_label}', ylabel='Background [MeV$^{-1}$s$^{-1}$sr$^{-1}$]')

    rot=0
    x_lim = ax.get_xlim()
    xticks_new = scale_value(fov_bin_edges,fov_lim,x_lim)
    ax.set_xticks(rotation=rot, ticks=xticks_new, labels=fov_bin_edges.round(1))
    ax.grid(True,alpha=0.2)

    row_mean_profile2[np.where(row_mean_profile == 0)] = np.nan
    row_mean_profile[np.where(row_mean_profile == 0)] = np.nan
    row_ratio = row_mean_profile2/row_mean_profile
    # row_ratio[np.where((row_ratio < ratio_lim[0]) | (row_ratio > ratio_lim[1]))] = np.nan

    column_mean_profile2[np.where(column_mean_profile == 0)] = np.nan
    column_mean_profile[np.where(column_mean_profile == 0)] = np.nan
    column_ratio = column_mean_profile2/column_mean_profile
    # column_ratio[np.where((column_ratio < ratio_lim[0]) | (column_ratio > ratio_lim[1]))] = np.nan
    
    ls_ratio = ':'
    sns.lineplot(x=fov_bin_centers_plot, y=row_ratio, ax=ax2, label='lon', lw=lw, ls=ls_ratio, marker=m_lon, color=colors[0])
    sns.lineplot(x=fov_bin_centers_plot, y=column_ratio, ax=ax2, label='lat', lw=lw, ls=ls_ratio, marker=m_lat, color=colors[2])

    ax2.set(xlim=x_lim, ylim=ratio_lim, xlabel='FoV coordinate [°]', ylabel='Ratio (out / true)')
    ax2.set_xticks(rotation=rot, ticks=xticks_new, labels=fov_bin_edges.round(1))
    ax2.grid(True,alpha=0.2)

    plt.show()
    if fig_save_path != '': fig.savefig(fig_save_path, dpi=300, transparent=False, bbox_inches='tight')
    
    return fig

def get_geom(Bkg_irf, axis_info, run_info, frame='icrs'):
    '''Return geom from bkg_irf axis or a set of given axis edges'''

    loc, source_pos, run, livetime, pointing, pointing_altaz = run_info
    
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
    if frame == 'icrs': skydir=(pointing.ra.degree,pointing.dec.degree)
    if frame == 'altaz': skydir=(pointing_altaz.az.degree,pointing_altaz.alt.degree)
    
    geom = WcsGeom.create(
            skydir=skydir,
            binsz=binsize,
            npix=(nbins_map, nbins_map),
            frame=frame,
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

def get_radial_dfbias_and_plot_bias_mean_std(irf_output,irf_true,cut_down_factors,save_path_plot,thr_percent=99.8,res_type='diff/true'):
    cut_factor,downsample_factor = cut_down_factors
    energy_axis = irf_output.axes[0]
    Ebin_mid = np.round(energy_axis.edges[:-1]+(energy_axis.edges[1:]-energy_axis.edges[:-1])*0.5,1)
    offset_axis = irf_output.axes[1]
    max_offset = offset_axis.edges.max().to_value(u.deg)
    x_centers = irf_true.axes['fov_lon'].center.value
    y_centers = irf_true.axes['fov_lon'].center.value
    X_, Y_ = np.meshgrid(x_centers, y_centers)

    center_x_, center_y_ = 0, 0  # Center of the circle

    radius_arr=np.linspace(0,np.sqrt(2*max_offset**2),10)
    r_low_arr = radius_arr[:-1]
    r_up_arr = radius_arr[1:]
    dfbias = pd.DataFrame(index=[r_low_arr,r_up_arr],columns=pd.MultiIndex.from_product([Ebin_mid,['in','out','dr','dr_std']],names=['Ebin','bias_mean']))
    dfbias.index.set_names(['r_low','r_up'],inplace=True)

    irf_output = irf_output.data
    irf_true = irf_true.data
    res_arr = []
    for iEbin,Ebin in enumerate(Ebin_mid):
        diff = irf_output[iEbin,:,:] - irf_true[iEbin,:,:]
        irf_true[iEbin,:,:][np.where(irf_true[iEbin,:,:] == 0.0)] = np.nan
        res = diff / irf_true[iEbin,:,:]
        diff[np.abs(res) >= 0.9] = np.nan

        if res_type == "diff/true": res = diff / irf_true[iEbin,:,:]
        elif res_type == "diff/sqrt(true)": res = diff / np.sqrt(irf_true[iEbin,:,:])
            # res_lim_for_nan = np.sqrt(res_lim_for_nan)

        res[irf_true[iEbin,:,:] == 0] = np.nan
        bkg_bias = 100*res
        res_arr.append(bkg_bias)
        dfbias[(Ebin.value,'in')] = [np.nanmean(bkg_bias[(X_ - center_x_)**2 + (Y_ - center_y_)**2 < r_up**2]) for r_up in r_up_arr]
        dfbias[(Ebin.value,'out')] = [np.nanmean(bkg_bias[~((X_ - center_x_)**2 + (Y_ - center_y_)**2 < r_up**2)]) for r_up in r_up_arr]
        dfbias[(Ebin.value,'dr')] = [np.nanmean(bkg_bias[((X_ - center_x_)**2 + (Y_ - center_y_)**2 >= r_low**2) & ((X_ - center_x_)**2 + (Y_ - center_y_)**2 < r_up**2)]) for r_low,r_up in zip(r_low_arr,r_up_arr)]
        dfbias[(Ebin.value,'dr_std')] = [np.nanstd(bkg_bias[((X_ - center_x_)**2 + (Y_ - center_y_)**2 >= r_low**2) & ((X_ - center_x_)**2 + (Y_ - center_y_)**2 < r_up**2)]) for r_low,r_up in zip(r_low_arr,r_up_arr)]
    display(dfbias)

    colors = plt.cm.jet(np.linspace(0, 1, irf_true.shape[0]))
    r_mid = r_low_arr+0.5*(r_up_arr-r_low_arr)
    dfbias_T = dfbias.T
    mean_zoom_min = dfbias_T.loc[(dfbias_T.index.get_level_values('bias_mean') == 'dr'), dfbias_T.columns[dfbias_T.columns.get_level_values('r_up') <= max_offset]].min().min()
    mean_zoom_max = dfbias_T.loc[(dfbias_T.index.get_level_values('bias_mean') == 'dr'), dfbias_T.columns[dfbias_T.columns.get_level_values('r_up') <= max_offset]].max().max()
    mean_zoom_min *= 0.8 + 0.4*(np.sign(mean_zoom_min) < 0)
    mean_zoom_max *= 0.8 + 0.3*(np.sign(mean_zoom_max) > 0)
    std_zoom_min = dfbias_T.loc[(dfbias_T.index.get_level_values('bias_mean') == 'dr_std'), dfbias_T.columns[dfbias_T.columns.get_level_values('r_up') <= max_offset]].min().min()
    std_zoom_max = dfbias_T.loc[(dfbias_T.index.get_level_values('bias_mean') == 'dr_std'), dfbias_T.columns[dfbias_T.columns.get_level_values('r_up') <= max_offset]].max().max()
    std_zoom_min *= 0.8 + 0.4*(np.sign(std_zoom_min) < 0)
    std_zoom_max *= 0.8 + 0.3*(np.sign(std_zoom_max) > 0)
    bbox = (1.02,1.02)

    # Flatten and filter NaNs
    r_center = np.sqrt(X_**2 + Y_**2).flatten()
    weights_Ebinall = np.sum(irf_true, axis=0)[np.newaxis, :, :].flatten()
    r_threshold = thr_percent
    r_thr_value = get_value_at_threshold(r_center,weights_Ebinall,r_threshold)

    for var,var_label,var_min,var_max in zip(['dr','dr_std'],['mean','std'],[mean_zoom_min,std_zoom_min],[mean_zoom_max,std_zoom_max]):
        fig,ax=plt.subplots(figsize=(7,6))
        for i,Ebin in enumerate(Ebin_mid):
            ax.errorbar(x = r_mid,
                                y=dfbias[Ebin.value,var], 
                                xerr=0.5 * (r_up_arr-r_low_arr),color=colors[i],label=str(Ebin),marker='.',lw=0.5)
        ylim = ax.get_ylim()
        ax.legend(bbox_to_anchor=bbox, loc='upper left', title='E bin mid')

        ax.vlines(x=r_thr_value,ymin=ylim[0],ymax=ylim[1],label=f'{r_threshold}% containment radius (true)',color='r')
        ax.set(xlabel='radius [°]',ylabel=f'bias {var_label} [%]')
        ax.set_title(f"Bias {var_label} between r and r+dr")
        ax.grid(True,lw=0.2)
        ax.set(xlabel='radius [°]',ylabel=f'bias {var_label} [%]',xlim=[0,max_offset],ylim=[var_min,var_max])
        fig.savefig(f'{save_path_plot}/bias_{var_label}.png', dpi=300, transparent=False, bbox_inches='tight')
        plt.show()

    mean_zoom_min = dfbias_T.loc[(dfbias_T.index.get_level_values('bias_mean') == 'dr'), dfbias_T.columns[dfbias_T.columns.get_level_values('r_up') <= r_thr_value]].min().min()
    mean_zoom_max = dfbias_T.loc[(dfbias_T.index.get_level_values('bias_mean') == 'dr'), dfbias_T.columns[dfbias_T.columns.get_level_values('r_up') <= r_thr_value]].max().max()
    mean_zoom_min *= 0.8 + 0.4*(np.sign(mean_zoom_min) < 0)
    mean_zoom_max *= 0.8 + 0.3*(np.sign(mean_zoom_max) > 0)
    std_zoom_min = dfbias_T.loc[(dfbias_T.index.get_level_values('bias_mean') == 'dr_std'), dfbias_T.columns[dfbias_T.columns.get_level_values('r_up') <= r_thr_value]].min().min()
    std_zoom_max = dfbias_T.loc[(dfbias_T.index.get_level_values('bias_mean') == 'dr_std'), dfbias_T.columns[dfbias_T.columns.get_level_values('r_up') <= r_thr_value]].max().max()
    std_zoom_min *= 0.8 + 0.4*(np.sign(std_zoom_min) < 0)
    std_zoom_max *= 0.8 + 0.3*(np.sign(std_zoom_max) > 0)

    for var,var_label,var_min,var_max in zip(['dr','dr_std'],['mean','std'],[mean_zoom_min,std_zoom_min],[mean_zoom_max,std_zoom_max]):
        fig,ax=plt.subplots(figsize=(7,6))
        for i,Ebin in enumerate(Ebin_mid):
            ax.errorbar(x = r_mid,
                                y=dfbias[Ebin.value,var], 
                                xerr=0.5 * (r_up_arr-r_low_arr),color=colors[i],label=str(Ebin),marker='.',lw=0.5)
        ylim = ax.get_ylim()
        ax.set(xlabel='radius [°]',ylabel=f'bias {var_label} [%]')
        ax.set_title(f"Bias {var_label} between r and r+dr")
        ax.legend(bbox_to_anchor=bbox, loc='upper left', title='E bin mid')
        ax.grid(True,lw=0.2)

        ax.set(xlabel='radius [°]',ylabel=f'bias {var_label} [%]',xlim=[0,r_thr_value],ylim=[var_min,var_max])
        fig.savefig(f'{save_path_plot}/bias_{var_label}_zoom.png', dpi=300, transparent=False, bbox_inches='tight')
        plt.show()
    return dfbias

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
        self.cfg_simulation = config["simulation"]
        self.cfg_wobble_1 = config["wobble_1"]
        self.cfg_wobble_2 = config["wobble_2"]
        self.cfg_source = config["source"]
        self.cfg_background = config["background"]
        self.cfg_irf = config["irf"]
        self.cfg_dataset = config["dataset"]
        self.cfg_acceptance = config["acceptance"]

        # Paths
        self.simulated_obs_dir = self.cfg_paths["simulated_obs_dir"]
        self.save_name_obs = self.cfg_paths["save_name_obs"]
        self.save_name_suffix = self.cfg_paths["save_name_suffix"]

        self.run_dir=self.cfg_paths["irf"] # path to files to get the irfs used for simulation
        self.path_data=self.cfg_paths["irf"] # path to files to get run information used for simulation

        # Source
        self.source_name=self.cfg_source["catalog_name"]
        self.source = SourceCatalogGammaCat(self.cfg_paths["gammapy_catalog"])[self.source_name]
        self.source_pos=self.source.position

        self.region_shape = self.cfg_source["exclusion_region"]["shape"]
        self.exclu_rad = self.cfg_source["exclusion_region"]["radius"]

        if (self.region_shape=='noexclusion'):
            # Parts of the code don't work without an exlcusion region, so a dummy one is declared at the origin of the ICRS frame (and CircleSkyRegion object needs a non-zero value for the radius) 
            self.source_region = CircleSkyRegion(center=SkyCoord(ra=0. * u.deg, dec=0. * u.deg, frame='icrs'),radius=1*u.deg)
        elif (self.region_shape=='circle'):
            self.exclusion_radius=self.exclu_rad * u.deg
            self.source_region = CircleSkyRegion(center=self.source_pos, radius=self.exclusion_radius)

        self.single_region=True # Set it to False to mask additional regions in the FoV. The example here is for Crab + Zeta Tauri
        if self.single_region:
            self.exclude_regions=[self.source_region] # The modelisation tool needs an array of regions to work
        else:
            zeta_region = CircleSkyRegion(center=SkyCoord(ra=84.4125*u.deg, dec=21.1425*u.deg), radius=self.exclusion_radius)
            self.exclude_regions = [self.source_region, zeta_region]

        self.source_info = [self.source_name,self.source_pos,self.source_region]
        self.flux_to_0 = self.cfg_source["flux_to_0"]
        
        # Background
        self.lon_grad = self.cfg_background["spatial_model"]["lon_grad"]
        self.correlation_radius = self.cfg_background["maker"]["correlation_radius"]
        self.ring_bkg_param = [self.cfg_background["maker"]["ring"]["internal_ring_radius"],self.cfg_background["maker"]["ring"]["width"]]
        self.fov_bkg_param = self.cfg_background["maker"]["fov"]["method"]
        
        # Acceptance binning
        self.e_min, self.e_max = float(self.cfg_acceptance["energy"]["e_min"])*u.TeV, float(self.cfg_acceptance["energy"]["e_max"])*u.TeV
        self.size_fov_acc = float(self.cfg_acceptance["offset"]["offset_max"]) * u.deg
        self.nbin_E_acc, self.nbin_offset_acc = self.cfg_acceptance["energy"]["nbin"],self.cfg_acceptance["offset"]["nbin"]

        self.offset_axis_acceptance = MapAxis.from_bounds(0.*u.deg, self.size_fov_acc, nbin=self.nbin_offset_acc, name='offset')
        self.energy_axis_acceptance = MapAxis.from_energy_bounds(self.e_min, self.e_max, nbin=self.nbin_E_acc, name='energy')
        
        self.Ebin_labels = [f"{ebin_min:.1f}-{ebin_max:.1f} {self.energy_axis_acceptance.unit}" for ebin_min, ebin_max in zip(self.energy_axis_acceptance.edges.value[:-1],self.energy_axis_acceptance.edges.value[1:])]

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
        self.axis_info_acceptance = [self.e_min,self.e_max,self.size_fov_acc.value,self.nbin_offset_acc]
        self.axis_info_irf = [self.e_min,self.e_max,self.size_fov_irf.value,self.nbin_offset_irf]
        self.axis_info_map = [self.e_min,self.e_max,self.size_fov_irf.value,self.nbin_offset_irf]

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
        livetime_W1,self.pointing_W1,pointing_altaz_W1 = get_run_info(self.path_data,run_W1) 
        # The true livetime is retrieved by get_run_info in case you want to implement a very realistic simulation pipeline with a simulation for each true observation you have
        # Here we use the same info for every run, which is the simulation livetime
        # The get_run_info method is mostly use to have realistic pointings, but you can decide the values yourself by changing the next lines
        self.run_info_W1=[self.loc,self.source_pos,run_W1,self.livetime_simu*(self.n_run if self.two_obs else 1),self.pointing_W1,pointing_altaz_W1]
        
        #  Wobble 2
        self.seed_W2=self.cfg_wobble_2["seed"]
        run_W2=self.cfg_wobble_2["run"]
        livetime_W2,self.pointing_W2,pointing_altaz_W2 = get_run_info(self.path_data,run_W2)
        self.run_info_W2=[self.loc,self.source_pos,run_W2,self.livetime_simu*(self.n_run if self.two_obs else 1),self.pointing_W2,pointing_altaz_W2]
        
        print(f"Total simulated livetime: {self.tot_livetime_simu.to(u.h):.1f}")

        # Naming scheme
        self.save_name_obs = f"{self.cfg_paths['save_name_obs']}"
        if self.two_obs: self.save_name_obs += f"_{0.5*self.tot_livetime_simu.to(u.h).value:.0f}h"

        self.multiple_simulation_subdir = False # TO-DO adapt for multiple subdirectories
        self.save_path_simu_joined = ''
        if self.multiple_simulation_subdir: self.save_path_simu = self.save_path_simu_joined
        else: self.save_path_simu = f"{self.simulated_obs_dir}/{self.save_name_obs}/{self.obs_collection_type}"+f"_{self.save_name_suffix}"*(self.save_name_suffix is not None)

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
        self.end_name=f'{self.bkg_dim}D_{self.region_shape}_Ebins_{self.nbin_E_acc}_offsetbins_{self.nbin_offset_acc}_offset_max_{self.size_fov_acc.value:.1f}'+f'_{self.cos_zenith_binning_parameter_value}sperW'*self.zenith_binning+f'_exclurad_0{self.exclu_rad}'*((self.exclu_rad != 0) & (self.region_shape != 'noexclusion'))
        self.index_suffix = f"_with_bkg_{self.bkg_dim}d_{self.method}_{self.end_name[3:]}"
        self.acceptance_files_dir=f"{self.save_path_simu}/{self.end_name}/{self.method}/acceptances"
        self.plots_dir=f"{self.save_path_simu}/{self.end_name}/{self.method}/plots"
        
        Path(self.acceptance_files_dir).mkdir(parents=True, exist_ok=True)
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
        print("acceptance files: ",self.acceptance_files_dir)
    
    def load_true_background_irfs(self, config_path=None) -> None:
        if config_path is not None: self.init_config(config_path)
        
        if self.true_collection:
            self.bkg_true_irf_collection = BackgroundCollectionZenith()
            self.bkg_true_down_irf_collection = BackgroundCollectionZenith()
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
            self.bkg_output_irf_collection = BackgroundCollectionZenith()
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
            else: self.pattern = f"obs_*{self.save_name_obs}.fits"
            print(f"Obs collection loading pattern: {self.pattern}")
            self.data_store, self.obs_collection = get_obs_collection(self.save_path_simu,self.pattern,self.multiple_simulation_subdir,from_index=from_index,with_datastore=True)
            self.obs_table = self.data_store.obs_table
            self.all_sources = np.unique(self.data_store.obs_table["OBJECT"])
            self.all_obs_ids = np.array(self.obs_table["OBS_ID"].data)
            print("Available sources: ", self.all_sources)
            print(f"{len(self.all_obs_ids)} available runs: ",self.all_obs_ids)
            self.tot_livetime_simu = 2*self.n_run*self.livetime_simu*u.s
        else:
            self.pattern = f"{self.obs_collection_type}_{self.save_name_suffix[:-8]}*/obs_*{self.save_name_obs}.fits" # Change pattern according to your sub directories
            self.obs_collection = get_obs_collection(self.simulated_obs_dir,self.pattern,self.multiple_simulation_subdir,with_datastore=False)
            self.all_obs_ids = np.arange(1,len(self.obs_collection)+1,1)
            self.tot_livetime_simu = 2*len(self.obs_collection)*self.livetime_simu*u.s
        print(f"Total simulated livetime: {self.tot_livetime_simu.to(u.h):.1f}")
    
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
                if self.true_collection: obs = get_empty_obs_simu(self.bkg_true_collection[iobs],None,run_info,self.source,self.run_dir,self.flux_to_0,self.t_ref,i*self.delay,verbose)
                else: obs = get_empty_obs_simu(bkg_irf,None,run_info,self.source,self.run_dir,self.flux_to_0,self.t_ref,i*self.delay,verbose)
                n = int(run_info[3]/oversampling.to_value("s")) + 1
                oversampling = (run_info[3]/n) * u.s
                run_info_over = deepcopy(run_info)
                run_info_over[3] = oversampling.to_value("s")
                for j in range(n):
                    print(j)
                    if self.true_collection: tdataset, tobs = get_empty_dataset_and_obs_simu(self.bkg_true_collection[iobs],None,run_info_over,self.source,self.run_dir,self.flux_to_0,self.t_ref,i*self.delay+j*oversampling.to_value("s"),verbose=False)
                    else: tdataset, tobs = get_empty_dataset_and_obs_simu(bkg_irf,None,run_info_over,self.source,self.run_dir,self.flux_to_0,self.t_ref,i*self.delay+j*oversampling.to_value("s"),verbose=False)
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

            for i in range(len(self.all_obs_ids)):
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

    def plot(self, data='acceptance', irf='true', residuals='none', profile='none', downsampled=True, title='', fig_save_path='', plot_hist=False) -> None:
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
            if self.true_collection: true = self.bkg_true_down_irf_collection[1]
            else: true = self.bkg_true_down_irf
        if (irf == 'output') or (irf == 'both'): 
            if self.out_collection: out = self.bkg_output_irf_collection[1]
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
