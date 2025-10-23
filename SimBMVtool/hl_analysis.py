import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, LinearSegmentedColormap

from copy import deepcopy
from scipy.stats import chi2, mode
from scipy.stats import norm as norm_stats
from scipy.optimize import curve_fit
from math import sqrt

import astropy.units as u
from astropy.coordinates import SkyCoord,EarthLocation
from astropy.visualization.wcsaxes import SphericalCircle
from gammapy.maps.region.geom import RegionGeom

from gammapy.modeling.models import (
    FoVBackgroundModel,
    PiecewiseNormSpectralModel,
    PointSpatialModel,
    DiskSpatialModel,
    GaussianSpatialModel,
    Models,
    SpatialModel,
    PowerLawSpectralModel,
    PowerLawNormSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    LogParabolaSpectralModel,
    SkyModel,
)

from itertools import combinations_with_replacement, product, chain

def get_models_names_list(spatial_models=["Point"], spectral_models=["PL"], n_sources=2, same_seeds=True):
    # Generate base source names
    spatial_models = np.concatenate((["No source"], spatial_models))
    source_names = [f"{s}{'' if s=='No source' else f' {sp}' }" for s in spatial_models for sp in spectral_models]

    # Generate all product combinations
    all_combinations = product(source_names, repeat=n_sources)

    # Keep only unique combinations (unordered)
    unique_combinations = set()
    models_names = []

    for comb in all_combinations:
        canonical = tuple(sorted(comb)) if same_seeds else comb
        if canonical not in unique_combinations:
            unique_combinations.add(canonical)
            models_names.append(" - ".join(comb[::-1]))
    return models_names

def add_model_to_models_dict(models_seeds:list, models_dict={}, model_name="Point PL", same_seeds=True):
    spatial_types_dict = dict([("Point", "PointSpatialModel"), ("1D Gauss", "GaussianSpatialModel"), ("2D Gauss", "GaussianSpatialModel"),
                           ("Disk", "DiskSpatialModel"),("Ellipse", "DiskSpatialModel")])
    spectral_types_dict = dict([("PL", "PowerLawSpectralModel"), ("ECPL", "ExpCutoffPowerLawSpectralModel"), ("LP", "LogParabolaSpectralModel")])

    model_component_names = []
    components = model_name.split(" - ")
    models = Models()
    for i, component in enumerate(components):
        if component != "No source":
            model_spatial = ' '.join(component.split(" ")[:-1])
            model_spectral = component.split(' ')[-1]
            model_name_component = f"{component} {i+1}"
            model_component_names.append(model_name_component)

            ref_source_model = models_seeds[0] if same_seeds else models_seeds[i]
            ref_source_model_dict = ref_source_model.to_dict().copy()
            ref_spatial_type, ref_spectral_type = (ref_source_model_dict['spatial']['type'],ref_source_model_dict['spectral']['type'])
            is_same_spatial_type = (spatial_types_dict[model_spatial]==ref_spatial_type)
            is_same_spectral_type = (spectral_types_dict[model_spectral]==ref_spectral_type)
            
            if (model_spatial != "Point") or is_same_spatial_type: model = ref_source_model.copy(name=model_name_component)
            else:
                model_new = PointSpatialModel(lon_0=ref_source_model.spatial_model.lon_0, lat_0=ref_source_model.spatial_model.lat_0, frame="icrs")
                model = SkyModel(name=model_name_component, spatial_model=model_new.copy(), spectral_model=ref_source_model.spectral_model.copy())

            if 'Gauss' in model_spatial:
                model_dim = int(model_spatial[0])
                if not is_same_spatial_type:
                    model.spatial_model = GaussianSpatialModel(lon_0=ref_source_model.spatial_model.lon_0, lat_0=ref_source_model.spatial_model.lat_0, sigma=0.15*u.deg, e=0, phi=0*u.deg, frame="icrs")

                if model_dim == 1:
                    model.spatial_model.e.frozen = True
                    model.spatial_model.phi.frozen = True

                elif model_dim == 2:
                    model.spatial_model.e.frozen = False
                    model.spatial_model.phi.frozen = False

                    model.parameters['e'].min = 0.
                    model.parameters['e'].max = 1.

                    model.parameters['phi'].min = 0.
                    model.parameters['phi'].max = 360

                model.parameters['sigma'].min = 0.05
                model.parameters['sigma'].max = 0.4
            
            elif model_spatial in ["Disk", "Ellipse"]:
                if not is_same_spatial_type:
                    model.spatial_model = DiskSpatialModel(lon_0=ref_source_model.spatial_model.lon_0, lat_0=ref_source_model.spatial_model.lat_0, r_0=0.2*u.deg, e=0, phi=0*u.deg, edge_width=0.01, frame="icrs")
                
                model.parameters['r_0'].min = 0.01
                model.parameters['r_0'].max = 1.
                
                model.spatial_model.e.frozen = True
                model.spatial_model.phi.frozen = True

                if model_spatial == "Ellipse":
                    model.spatial_model.e.frozen = False
                    model.spatial_model.phi.frozen = False

                    model.parameters['e'].min = 0.
                    model.parameters['e'].max = 1.

                    model.parameters['phi'].min = 0.
                    model.parameters['phi'].max = 360

            index =  model.parameters['index'] if ref_source_model_dict['spectral']['type'] in ['PowerLawSpectralModel', 'ExpCutoffPowerLawSpectralModel'] else 2.
            lambda_ =  model.parameters['lambda_'] if ref_source_model_dict['spectral']['type'] == 'ExpCutoffPowerLawSpectralModel' else 1e-2/u.TeV
            alpha =  model.parameters['alpha'] if ref_source_model_dict['spectral']['type'] in ['ExpCutoffPowerLawSpectralModel','LogParabolaSpectralModel'] else (1. if model_spectral=='ECPL' else 2.)
            beta =  model.parameters['beta'] if ref_source_model_dict['spectral']['type'] == 'LogParabolaSpectralModel' else 1.
            
            if model_spectral in ["PL", "ECPL"]:
                if not is_same_spectral_type:
                    if model_spectral == "PL":
                        model.spectral_model = PowerLawSpectralModel(index=index, amplitude=model.parameters['amplitude'],reference=model.parameters['reference'])
                    else:
                        model.spectral_model = ExpCutoffPowerLawSpectralModel(index=index, lambda_=lambda_, alpha=alpha, amplitude=model.parameters['amplitude'],reference=model.parameters['reference'])
                model.parameters['index'].min = model.parameters['index'].value - 1
                model.parameters['index'].max = model.parameters['index'].value + 1
                
                if model_spectral == "ECPL":
                    model.parameters['lambda_'].min = -0.1
                    model.parameters['lambda_'].max = 1
            
            elif model_spectral == "LP":
                if not is_same_spectral_type:
                    model.spectral_model = LogParabolaSpectralModel(alpha=alpha, beta=beta, amplitude=model.parameters['amplitude'],reference=model.parameters['reference'])
            
                model.parameters['alpha'].min = 1.
                model.parameters['alpha'].max = 3.
                model.parameters['beta'].min = 0.
                model.parameters['beta'].max = 2.

            model.parameters['lon_0'].min = model.spatial_model.lon_0.value - 0.4
            model.parameters['lon_0'].max = model.spatial_model.lon_0.value + 0.4

            model.parameters['lat_0'].min = model.spatial_model.lat_0.value - 0.4
            model.parameters['lat_0'].max = model.spatial_model.lat_0.value + 0.4

            model.parameters['amplitude'].min = model.parameters['amplitude'].value / 1e2
            model.parameters['amplitude'].max = model.parameters['amplitude'].value * 1e2
            
            models.append(model)

    models_dict[model_name] = {
            "models" : models,
            "dof" : len(models.parameters.free_parameters)}
    return deepcopy(models_dict)

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

def get_nested_model_significance(fitted_null_model, fitted_alternative_model, return_dof=True):
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

def get_dfmodels_sigmatrix_dof(results:dict, bkg_method='ring', fit_method='stacked'):
    for key in results[bkg_method].keys():
        if (key == 'results') or  (key == 'results_3d'): res_key = key
    
    models_list = list(results[bkg_method][res_key].keys())

    dfmodels_0 = pd.DataFrame(index=pd.Index(models_list, name='null_model'), columns=pd.MultiIndex.from_product([["wilk","delta_dof"],models_list], names=['stat','alt_model']))
    dfmodels_wilk_dof = dfmodels_0.copy()

    for null_model, alt_model in product(models_list, models_list):
        fitted_null = results[bkg_method][res_key][null_model][fit_method]
        fitted_alt = results[bkg_method][res_key][alt_model][fit_method]
        fit_fail = (("fit_success" in fitted_null.keys()) and not fitted_null["fit_success"]) or (("fit_success" in fitted_alt.keys()) and not fitted_alt["fit_success"])
        if fit_fail:
            dfmodels_wilk_dof.loc[null_model, ("wilk", alt_model)] = -1
            continue
        
        null_components = null_model.split(' - ')
        alt_components = alt_model.split(' - ')
        # print(null_components, fitted_null['models'])
        is_nested_list = []
        for enum_null, alt in zip(enumerate(null_components), alt_components):
            i_comp, null = enum_null
            if null == 'No source': is_nested = True
            elif alt == 'No source': is_nested = False
            else:
                is_nested = is_nested_model(fitted_null['models'][f"{null} {i_comp+1}"],fitted_alt['models'][f"{alt} {i_comp+1}"])
            is_nested_list.append(is_nested)

        if all(is_nested_list):
            wilk_sig, delta_dof = get_nested_model_significance(fitted_null['fit_result'], fitted_alt['fit_result'])
            dfmodels_wilk_dof.loc[null_model,  ("wilk", alt_model)] = wilk_sig
            dfmodels_wilk_dof.loc[null_model,  ("delta_dof", alt_model)] = delta_dof
    return dfmodels_wilk_dof

def get_dfbest_models(dfmodels_sigmatrix:pd.DataFrame, results:dict, bkg_method='ring', fit_method='stacked', rel_L_tsh=1, sig_method='wilk'):
    for key in results[bkg_method].keys():
        if (key == 'results') or  (key == 'results_3d'): res_key = str(key)
    
    models_list = list(results[bkg_method][res_key].keys())
    dfmodels = pd.DataFrame(index=pd.Index(models_list,name='model'))
    
    models_to_remove = []
    for tested_model in models_list:
        fitted_model = results[bkg_method][res_key][tested_model][fit_method]
        if (("fit_success" in fitted_model.keys()) and not fitted_model["fit_success"]):
            dfmodels.loc[tested_model, "AIC"] = 1e3
            dfmodels.loc[tested_model, "AIC"] = 1e3
            models_to_remove.append(tested_model)
            continue
        fitted_model_res = fitted_model['fit_result']
        dfmodels.loc[tested_model, "AIC"] = 2*len(fitted_model_res.parameters.free_parameters) + fitted_model_res.total_stat
        # dfmodels['rel_L'] = np.exp((dfmodels.AIC.min() - dfmodels.AIC) * 0.5)
        dfmodels.loc[tested_model, 'wilk'] = dfmodels_sigmatrix.loc[models_list[0], tested_model]

    if sig_method == 'wilk': best_models = dfmodels[~dfmodels.index.isin(models_to_remove)].index.to_list()
    else:
        best_models = dfmodels[~dfmodels.index.isin(models_to_remove) & (dfmodels.rel_L > np.exp(-rel_L_tsh))].sort_values(by="AIC",ascending=True).index.to_list()
    
    print("-----------------------------------------")

    models_to_keep_as_alt = []
    print("----- Testing models as alternative -----")
    for imodel in range(len(best_models)):
        tested_model = best_models[imodel]
        if all(c == "No source" for c in tested_model.split(" - ")):
            models_to_remove.append(tested_model)
            continue
        print(f"\nTested model: {tested_model}")
        
        tested_model_as_alt = dfmodels_sigmatrix[tested_model].copy()

        better_null_models = tested_model_as_alt.loc[(tested_model_as_alt < 3) & (tested_model_as_alt >= 0) & (tested_model_as_alt != np.nan)]

        if len(better_null_models) > 0: better_null_str = f"Not more significant than {len(better_null_models)} null model{'s' if len(better_null_models)>1 else ''}"
        else: better_null_str = "More significant than all null models"
            
        if (len(better_null_models) > 0):
            models_to_remove.append(tested_model)
            print(f"REMOVED: {better_null_str}")
        else:
            models_to_keep_as_alt.append(tested_model)
            print(f"KEPT: {better_null_str}")
    
    # models_to_keep_not_overfitted = []
    
    # print("\n----- Testing remaining models as alternative for overfitting -----")
    # for imodel in range(len(models_to_keep_as_alt)):
    #     tested_model = models_to_keep_as_alt[imodel]
    #     print(f"\nTested model: {tested_model}")
        
    #     tested_model_as_alt = dfmodels_sigmatrix[tested_model].copy()
    #     nulls = tested_model_as_alt[tested_model_as_alt.notna()].index
    #     overfitting =  nulls[1:].isin(models_to_remove).any()
    #     if overfitting:
    #         models_to_remove.append(tested_model)
    #         print(f"REMOVED: at least 1 null model removed (overfitting)")
    #     else:
    #         models_to_keep_not_overfitted.append(tested_model)
    #         print(f"KEPT")


    models_to_keep = []
    print("\n----- Testing remaining models as null -----")
    for imodel in range(len(models_to_keep_as_alt)):
        tested_model = models_to_keep_as_alt[imodel]
        print(f"\nTested model: {tested_model}")
        
        tested_model_as_null = dfmodels_sigmatrix.loc[tested_model].copy()     
        better_alt_models = tested_model_as_null.loc[(tested_model_as_null > 3) & (tested_model_as_null != np.nan)]

        len_better_alt_models = len(better_alt_models)
        if len_better_alt_models > 0:
            better_alt_str = f"{len_better_alt_models} more significant alternative models"
            # for better_alt_model_name in better_alt_models.to_frame().index.to_numpy():
            #     if better_alt_model_name in models_to_remove: len_better_alt_models -= 1
            if len_better_alt_models == 0: better_alt_str += " (overfitting)"
        else:
            better_alt_str = "No more significant alternative model"
        
        if (len_better_alt_models > 0):
            models_to_remove.append(tested_model)
            print(f"REMOVED: {better_alt_str}")
        else:
            models_to_keep.append(tested_model)
            print(f"KEPT: {better_alt_str}")

    
    dfbest_models = dfmodels.reset_index()
    dfbest_models = dfbest_models[dfbest_models.model.isin(models_to_keep)]
    dfbest_models = dfbest_models[~dfbest_models.model.isin(models_to_remove)][["model", "wilk", "AIC"]]
    dfbest_models['relative_L'] = np.exp((dfbest_models.AIC.min() - dfbest_models.AIC) * 0.5)
    
    return dfmodels, dfbest_models.sort_values(by="AIC")

def plot_dfmodels_sigmatrix(dfmodels_wilk:pd.DataFrame, fontsize=15, rotation=45, figsize = (20,20), fig_ax=(None,None), shrink=0.7, title="", labels=["",""], annot=True, square=False):
    xlabel, ylabel = labels
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
    ax.set_title(label="Significance matrix\n" if title == "" else title, fontsize=fontsize+2)
    plt.tight_layout()
    plt.show()

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
        plot_kwargs.pop("lw_sigma", None)
        plot_kwargs.pop("alpha_sigma", None)
        ref_source_model.spectral_model.plot(ax=ax, **plot_kwargs)
        plot_kwargs.pop("label", None)
        edgecolor = plot_kwargs["color"]
        plot_kwargs.pop("color", None)
        ref_source_model.spectral_model.plot_error(ax=ax, alpha=0.1, edgecolor=edgecolor, facecolor=edgecolor, hatch='..', **plot_kwargs)
    elif plot_type=='spatial':
        center = SkyCoord(ra=ref_source_model.spatial_model.lon_0.value * u.deg,dec=ref_source_model.spatial_model.lat_0.value * u.deg, frame="icrs")
        if "Point" in ref_source_model_dict['spatial']['type']:
            ax.scatter(
                center.ra,
                center.dec,
                transform=ax.get_transform("icrs"),
                color=plot_kwargs["color_sigma"],
                edgecolor="white",
                marker="o",
                s=120,
                lw=1.5,
                label = plot_kwargs["label"]
            )
        else:
            if "Gaussian" in ref_source_model_dict['spatial']['type']: extension = ref_source_model.spatial_model.sigma.value
            elif "Disk" in ref_source_model_dict['spatial']['type']: extension = ref_source_model.spatial_model.r_0.value
            r = SphericalCircle(center, extension * u.deg,
                                    edgecolor=plot_kwargs["color_sigma"], facecolor='none',
                                    lw = plot_kwargs["lw_sigma"],
                                    ls = plot_kwargs["ls_sigma"],
                                    alpha = plot_kwargs["alpha_sigma"],
                                    transform=ax.get_transform('icrs'), label=plot_kwargs["label"])
            ax.add_patch(r)
        

def plot_spatial_model_from_dict(bkg_method, key, results, ref_models, ref_source_name, ref_models_to_plot, results_to_plot=['all'], bkg_methods_to_plot=['all'], fit_methods_to_plot=['all'], width=2 * u.deg, estimator='excess', figsize=(5,5),bbox_to_anchor=(1,1),fontsize=15, colors = ['blue', 'darkorange', 'purple', 'green'], smooth = 0.02, vmin_vmax=(-4.5,8.65),peak_dist_min=0.2):
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
            'title': f'Flux map ({bkg_method}, {estimator})'
        }
    }

    labelcolor_legend = "w"
    facecolor_legend = "indigo"
    alpha_legend = 0.5

    colors_contours = ["w", "crimson"]
    alpha_contours = 1
    lw_contours = [1, 1.5]
    
    skymaps_dict = results[bkg_method][f'skymaps_{estimator}']
    skymap = skymaps_dict[key].cutout(position=skymaps_dict[key]._geom.center_skydir, width=width)
    skymap_sig = skymaps_dict["significance_all"].cutout(position=skymaps_dict["significance_all"]._geom.center_skydir, width=width)
    
    cbar_label, title = (skymaps_args[key]["cbar_label"], skymaps_args[key]["title"])

    fig,ax=plt.subplots(figsize=figsize,subplot_kw={"projection": skymap.geom.wcs})
    vmin=np.nanmax(skymap.data)
    vmax=np.nanmin(skymap.data)
    skymap.smooth(smooth * u.deg).plot(add_cbar=False, cmap='magma', stretch="linear", ax=ax)

    CS = ax.contour(skymap_sig.smooth(smooth * u.deg).data[0], levels=[3,5], linewidths=lw_contours, colors=colors_contours, alpha=alpha_contours)
    ax.clabel(CS, [3,5,], inline=3, fontsize=12, colors=colors_contours)
    im = ax.images[-1]
    plt.colorbar(im,ax=ax, shrink=1, label=cbar_label)
    
    for ref_model_name in ref_models_to_plot:
        plot_ref(ax, ref_models, ref_source_name, ref_model_name, plot_type = "spatial")

    both_bkg_methods = bkg_methods_to_plot == ['all']
    both_fit_methods = fit_methods_to_plot == ['all']

    at_least_two_bkg_methods = len(bkg_methods_to_plot) > 1

    bkg_methods = ['ring', 'FoV'] if both_bkg_methods else bkg_methods_to_plot
    fit_methods = ['stacked', 'joint'] if fit_methods_to_plot == ['all'] else fit_methods_to_plot
    res_key = 'results' if 'results' in results[bkg_methods[0]].keys() else f'results_3d'
    tested_models = list(results[bkg_methods[0]][res_key].keys()) if results_to_plot == ['all'] else results_to_plot
    no_source_in_fov_model = list(results[bkg_methods[0]][res_key].keys())[0]
    i=0
    lw_result = 3
    for ibkg_method,bkg_method in enumerate(bkg_methods):
        for tested_model in tested_models:
            tested_model_str = ""
            for fit_method in fit_methods:
                fitted_null_model = results[bkg_method][res_key][no_source_in_fov_model][fit_method]["fit_result"]
                label_methods = " ("* (both_bkg_methods or both_fit_methods or at_least_two_bkg_methods) + bkg_method * (both_bkg_methods or at_least_two_bkg_methods) +","*(both_bkg_methods and both_fit_methods)+ fit_method * both_fit_methods +")"* (both_bkg_methods or both_fit_methods or at_least_two_bkg_methods)

                if (len(results[bkg_method][res_key][tested_model][fit_method]) == 0): continue
                results_model = results[bkg_method][res_key][tested_model][fit_method]['models'].copy()
                
                j=0
                for model in results_model[:-1]:
                    model_name = model.name
                    tested_model_str += (("" if j == 0 else " - ") + model_name[:-2])
                    center = SkyCoord(ra=model.spatial_model.lon_0.value * u.deg,dec=model.spatial_model.lat_0.value * u.deg, frame="icrs")
                    label=f"{model_name[:-2]}{label_methods}"
                    i_color = j if len(bkg_methods) == 1 else j + 2*ibkg_method
                    if j==0:
                        fitted_alternative_model = results[bkg_method][res_key][tested_model][fit_method]['fit_result']
                        wilk_sig = get_nested_model_significance(fitted_null_model, fitted_alternative_model, False)
                        label_wilk= f"$\sigma$={wilk_sig:.2f}"
                    
                    if "Point" in model_name:
                        r = SphericalCircle(center, 0.04 * u.deg,
                                            edgecolor='black', facecolor=colors[i_color],
                                            ls = "-",
                                            lw = 1,
                                            transform=ax.get_transform('icrs'), label=label)
                        ax.add_patch(r)
                    elif "Gauss" in model_name:
                        sigma = model.spatial_model.sigma.value
                        if "1D" in model_name:
                            label += f": $\sigma$={sigma:.2f}°"
                        
                            r = SphericalCircle(center, sigma * u.deg,
                                                edgecolor=colors[i_color], facecolor='none',
                                                ls = "-",
                                                lw = lw_result,
                                                transform=ax.get_transform('icrs'), label=label)
                            ax.add_patch(r)
                        else:
                            sky_region = model.spatial_model.to_region(x_sigma=1.)
                            pixel_region = sky_region.to_pixel(skymap.geom.wcs)
                            label += ": $\sigma_{eff}$"+f"={sigma:.2f}°"
                            pixel_region.plot(ax=ax,
                                            color = colors[i_color],
                                            lw=lw_result,
                                            ls = "-",
                                            label=label)
                    elif ("Disk" in model_name) or ("Ellipse" in model_name):
                        sky_region = model.spatial_model.to_region()
                        pixel_region = sky_region.to_pixel(skymap.geom.wcs)
                        if "Disk" in model_name:
                            r_0 = model.spatial_model.r_0.value
                            label += f": $r_0$={r_0:.2f}°"
                        else:
                            width_ellipse = sky_region.width.value/2
                            height_ellipse = sky_region.height.value/2
                            label += f": width,heigth={width_ellipse:.2f}°,{height_ellipse:.2f}°"

                        pixel_region.plot(ax=ax,
                                        color = colors[i_color],
                                        lw=lw_result,
                                        ls = "-",
                                        label=label)
                    j+=1
                i+=1

    ax.set_title(label=title+f"\n{tested_model_str}  ({label_wilk})",fontsize=fontsize)
    ax.legend(loc='upper left',fontsize=fontsize-3, labelcolor=labelcolor_legend, facecolor=facecolor_legend, framealpha=alpha_legend)
    plt.tight_layout()
    plt.show()

def plot_spectra_from_models_dict(ax, results, ref_models, ref_source_name, ref_models_to_plot, analysis_dim=3,energy_bounds = [3, 100], results_to_plot=['all'], bkg_methods_to_plot=['all'], fit_methods_to_plot=['all'], colors = ['blue', 'darkorange', 'purple', 'green'], plot_flux_points=True):

    if len(ref_models_to_plot)>0:
        for ref_model_name in ref_models_to_plot:
            plot_ref(ax, ref_models, ref_source_name, ref_model_name, plot_type = "spectral")
    
    bkg_methods = ['ring', 'FoV'] if bkg_methods_to_plot == ['all'] else bkg_methods_to_plot
    fit_methods = ['stacked', 'joint'] if fit_methods_to_plot == ['all'] else fit_methods_to_plot
    res_key = 'results' if 'results' in results[bkg_methods[0]].keys() else f'results_{analysis_dim}d'
    tested_models = list(results[bkg_methods[0]][res_key].keys()) if results_to_plot == ['all'] else results_to_plot
    i=0
    for ibkg_method,bkg_method in enumerate(bkg_methods):
        for tested_model in tested_models:
            for fit_method in fit_methods:
                if (len(results[bkg_method][res_key][tested_model][fit_method]) == 0):
                    print(f"Fit failed for {tested_model}")
                    continue
                results_model = results[bkg_method][res_key][tested_model][fit_method]['models'].copy()
                
                if analysis_dim == 3:
                    n_bkg = 2 if fit_method == 'combined' else 1
                    for i_model, model in enumerate(results_model[:-n_bkg]):
                        i_color = i_model if len(bkg_methods) == 1 else i_model + 2*ibkg_method
                        model_name = model.name
                        plot_kwargs = {
                            "energy_bounds": energy_bounds * u.TeV,
                            "sed_type": "e2dnde",
                            "ls" : "-",
                            "color": colors[i_color],
                            "yunits": u.Unit("TeV cm-2 s-1"),
                        }
                        
                        model_name_label = model_name[:-2]
                        label = f"{model_name_label} ({bkg_method}, {fit_method})"
                        model.spectral_model.plot(ax=ax, label=label,**plot_kwargs.copy())
                        model.spectral_model.plot_error(ax=ax, alpha=0.1, facecolor=colors[i_color], **plot_kwargs.copy())
                        if (model_name in results[bkg_method][res_key][tested_model][fit_method].keys()):
                            if (results[bkg_method][res_key][tested_model][fit_method][model_name]["flux_points_success"]):
                                flux_points = results[bkg_method][res_key][tested_model][fit_method][model_name]["flux_points"]
                                flux_points.plot(ax=ax, sed_type="e2dnde", color=colors[i_color], marker='o' if 'tail' in model_name else "x", markersize=5 if 'tail' in model_name else 7)
                else:
                    model_name = tested_model
                    i_color = i
                    plot_kwargs = {
                        "energy_bounds": energy_bounds * u.TeV,
                        "sed_type": "e2dnde",
                        "ls" : "-",
                        "color": colors[i_color],
                        "yunits": u.Unit("TeV cm-2 s-1"),
                    }
                    
                    label = f"{model_name} ({bkg_method}, {fit_method})"
                    results_model[0].spectral_model.plot(ax=ax, label=label,**plot_kwargs.copy())
                    results_model[0].spectral_model.plot_error(ax=ax, alpha=0.3, facecolor=colors[i_color], **plot_kwargs.copy())
                    if results[bkg_method][res_key][tested_model][fit_method][model_name]["flux_points_success"]:
                        flux_points = results[bkg_method][res_key][tested_model][fit_method][model_name]["flux_points"]
                        flux_points.plot(ax=ax, sed_type="e2dnde", color=colors[i_color], marker='o' if 'tail' in model_name else "x", markersize=5 if 'tail' in model_name else 7)
                i+=1

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


# def get_dfmodels_sigmatrix_dof(results:dict, bkg_method='ring', fit_method='stacked'):
#     for key in results[bkg_method].keys():
#         if (key == 'results') or  (key == 'results_3d'): res_key = key
    
#     models_list = list(results[bkg_method][res_key].keys())

#     dfmodels_0 = pd.DataFrame(index=pd.Index(models_list, name='null_model'), columns=pd.MultiIndex.from_product([["wilk","delta_dof"],models_list], names=['stat','alt_model']))
#     dfmodels_wilk_dof = dfmodels_0.copy()

#     for null_model, alt_model in product(models_list, models_list):
#         fitted_null = results[bkg_method][res_key][null_model][fit_method]
#         fitted_alt = results[bkg_method][res_key][alt_model][fit_method]
#         fit_fail = (("fit_success" in fitted_null.keys()) and not fitted_null["fit_success"]) or (("fit_success" in fitted_alt.keys()) and not fitted_alt["fit_success"])
#         if fit_fail:
#             dfmodels_wilk_dof.loc[null_model, ("wilk", alt_model)] = -1
#             continue
        
#         null_2, null_1 = null_model.split(' - ')
#         alt_2, alt_1 = alt_model.split(' - ')

#         if null_1 == 'No source': is_nested_1 = True
#         elif alt_1 == 'No source': is_nested_1 = False
#         else:
#             is_nested_1 = is_nested_model(fitted_null['models'][null_1 + ' 2'],fitted_alt['models'][alt_1 + ' 2'])
        
#         if null_2 == 'No source': is_nested_2 = True
#         elif alt_2 == 'No source': is_nested_2 = False
#         else:
#             is_nested_2 = is_nested_model(fitted_null['models'][null_2 + ' 1'],fitted_alt['models'][alt_2 + ' 1'])

#         if is_nested_1 and is_nested_2:
#             wilk_sig, delta_dof = get_nested_model_significance(fitted_null['fit_result'], fitted_alt['fit_result'])
#             dfmodels_wilk_dof.loc[null_model,  ("wilk", alt_model)] = wilk_sig
#             dfmodels_wilk_dof.loc[null_model,  ("delta_dof", alt_model)] = delta_dof
#     return dfmodels_wilk_dof