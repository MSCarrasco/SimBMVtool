# Installation steps

## SimBMVtool

`git clone https://github.com/MSCarrasco/SimBMVtool.git <path_to_SimBMVtool_clone>`

`cd <path_to_SimBMVtool_clone>`

`git clone https://github.com/gammapy/gammapy-extra.git`

Change the path to gammapy catalog and data accordingly in the config files if you store it elsewhere


## Working environment with gammapy and BAccMod

You have multiple options:

1. Create a dedicated environment with the one provided in the SimBMVtool folder with `conda env create -f simbmvtool-environment.yml`
2. If you already have a working environment with gammapy 1.1 and BAccMod 0.3.0 everything should work. If you have issues with some packages, check the versions in the environment file.
3. The official gammapy release doesn't include a fov rotation parameter useful to evaluate background IRF whith a ALTAZ FoV alignment. If you want more acurate evaluation you will have to install custom branches and follow the steps described here after:

`git clone https://github.com/MSCarrasco/gammapy.git <path_to_gammapy_clone>`

`cd <path_to_gammapy_clone>`

`conda env create -f environment-dev.yml -n <your_env_name>`

`conda activate <your_env_name>`

`python -m pip install .`

`git clone https://github.com/MSCarrasco/BAccMod.git <path_to_BAccMod_clone>`

`cd <path_to_BAccMod_clone>`

`python setup.py install`

Now you should be able to use the notebooks and follow the tutorials