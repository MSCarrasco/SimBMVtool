import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

# %matplotlib inline

from gammapy.catalog import SourceCatalogGammaCat
from gammapy.irf import Background2D, Background3D, FoVAlignment
from gammapy.maps import MapAxis, WcsGeom, WcsNDMap
from regions import CircleAnnulusSkyRegion, CircleSkyRegion

from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    FoVBackgroundModel,
    SpatialModel,
    GaussianSpatialModel,
    PowerLawNormSpectralModel,
)

from bkg_irf_models import GaussianSpatialModel_LinearGradient, GaussianSpatialModel_LinearGradient_half

def evaluate_bkg(bkg_dim,bkg_model,energy_axis,offset_axis):
    ecenters = energy_axis.center.to_value(u.TeV)
    centers = offset_axis.center.to_value(u.deg)

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
    return bkg_irf if bkg_dim == 3 else bkg_irf.to_2d()

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
            irf_cut_map.sum_over_axes(['energy']).plot(ax=ax, cbar=True)
    else: irf_down_map = irf_map.copy()

    irf_down_map = irf_down_map.downsample(factor=downsample_factor).downsample(factor=downsample_factor,axis_name='energy')

    if plot:
        fig, ax = plt.subplots(figsize = (6,6))
        irf_down_map.sum_over_axes(['energy']).plot(ax=ax, cbar=True)
    
    bkg_irf_down = get_bkg_irf(irf_down_map, 2*irf_down_offset_axis.nbin, irf_down_energy_axis, irf_down_offset_axis, bkg_dim,livetime,FoV_alignment=FoV_alignment)
    if verbose:
        print(f"irf_map: {irf_map.data.shape}\nirf_cut_map: {irf_cut_map.data.shape}\nirf_down_map: {irf_down_map.data.shape}")
    return irf_down_map, bkg_irf_down

def get_bkg_true_irf_from_config(config, downsample=True, downsample_only=True, plot=False, verbose=False, return_map=False):
    cfg_paths = config["paths"]

    cfg_simulation = config["simulation"]
    n_run = cfg_simulation["n_run"] # Multiplicative factor for each simulated wobble livetime
    livetime_simu = cfg_simulation["livetime"]

    cfg_background = config["background"]
    cfg_irf = cfg_background["irf"]
    cfg_acceptance = config["acceptance"]

    bkg_dim=cfg_acceptance["dimension"]

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
    nbin_offset_irf = nbin_offset_acc * down_factor * offset_factor
    offset_max_irf= size_fov_irf

    offset_axis_irf = MapAxis.from_bounds(0.*u.deg, offset_max_irf, nbin=nbin_offset_irf, name='offset')

    ## Spectral model
    bkg_model_dict = cfg_background["custom_source"]
    bkg_spectral_model = PowerLawNormSpectralModel.from_dict(bkg_model_dict["spectral"]) 

    ## Spatial model
    spatial_model = cfg_background['custom_source']["spatial"]["type"]
    if spatial_model ==  "GaussianSpatialModel_LinearGradient":
        bkg_spatial_model = GaussianSpatialModel_LinearGradient.from_dict(bkg_model_dict["spatial"])
    else: bkg_spatial_model = SpatialModel.from_dict(bkg_model_dict["spatial"])

    bkg_true_model = FoVBackgroundModel(dataset_name=bkg_model_dict["name"], spatial_model=bkg_spatial_model, spectral_model=bkg_spectral_model)
    
    if verbose: print(bkg_true_model)
    
    bkg_true_rates = evaluate_bkg(bkg_dim,bkg_true_model,energy_axis_irf,offset_axis_irf)
    bkg_true_irf = get_bkg_irf(bkg_true_rates, 2*nbin_offset_irf, energy_axis_irf, offset_axis_irf, bkg_dim, FoV_alignment=cfg_acceptance["FoV_alignment"])
    
    if not downsample:
        return bkg_true_irf
    else:
        bkg_true_map = get_irf_map(bkg_true_rates,[energy_axis_irf,offset_axis_irf],n_run*livetime_simu)
        bkg_true_down_map, bkg_true_down_irf = get_cut_downsampled_irf_from_map(bkg_true_map,[energy_axis_acceptance,offset_axis_acceptance], [offset_factor, down_factor], bkg_dim, n_run * livetime_simu, plot=plot, verbose=verbose,FoV_alignment=cfg_acceptance["FoV_alignment"])
        bkg_true_map.data /= n_run
        bkg_true_down_map.data /= n_run
        if downsample_only:
            if return_map: return bkg_true_down_irf, bkg_true_down_map
            else: return bkg_true_down_irf
        else:
            if return_map: return bkg_true_irf, bkg_true_down_irf, bkg_true_map, bkg_true_down_map
            else: return bkg_true_irf, bkg_true_down_irf
