# Installation steps

## SimBMVtool

`git clone https://github.com/MSCarrasco/SimBMVtool.git <path_to_SimBMVtool_clone>`

`cd <path_to_SimBMVtool_clone>`

`conda env create -f simbmvtool-environment.yml -n <env_name>`

`conda activate <env_name>`

If it is the first time:

`gammapy download datasets`

`conda env config vars set GAMMAPY_DATA=$PWD/gammapy-datasets/2.0`

Else:

`conda env config vars set GAMMAPY_DATA=<path_to_gammapy_datasets>`

Now you should be able to use the notebooks and follow the tutorials
