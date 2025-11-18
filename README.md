# Installation steps

## SimBMVtool

`git clone https://github.com/MSCarrasco/SimBMVtool.git <path_to_SimBMVtool_clone>`

`cd <path_to_SimBMVtool_clone>`

`gammapy download datasets`

Change the path to gammapy catalog and data accordingly in the config files if you store it elsewhere

## Working environment with gammapy and BAccMod

SimBMVtool requires gammapy 2.0 and BAccMod 0.4.0

You have multiple options:

1. Create a dedicated environment with the one provided in the SimBMVtool folder with `conda env create -f simbmvtool-environment.yml`
2. If you already have a working environment with gammapy and BAccMod everything should work. If you have issues with some packages, check the versions in the environment file. If you just miss BAccMod, use `pip install BAccMod==0.4.0`

Now you should be able to use the notebooks and follow the tutorials
