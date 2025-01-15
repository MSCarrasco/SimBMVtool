import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord,EarthLocation, angular_separation, position_angle,Angle
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

from copy import deepcopy
import pandas as pd
from pathlib import Path
import os, pathlib
import pickle as pk
import seaborn as sns
from itertools import product

# %matplotlib inline

from IPython.display import display
from gammapy.data import FixedPointingInfo, Observations, Observation, PointingMode
from gammapy.datasets import MapDataset
from gammapy.irf import load_irf_dict_from_file, Background2D, Background3D, FoVAlignment
from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, WcsGeom, WcsNDMap
from regions import CircleAnnulusSkyRegion, CircleSkyRegion
from gammapy.estimators import ExcessMapEstimator

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
