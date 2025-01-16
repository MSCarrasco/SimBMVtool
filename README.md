# Installation steps

## Working environment with gammapy and BAccMod
We are using a custom gammapy branch for reasons related to the simulation

`git clone https://github.com/MSCarrasco/gammapy.git <path_to_gammapy_clone>`

`cd <path_to_gammapy_clone>`

`conda env create -f environment-dev.yml -n <your_env_name>`

`conda activate <your_env_name>`

`python -m pip install .`

`git clone https://github.com/MSCarrasco/BAccMod.git <path_to_BAccMod_clone>`

`cd <path_to_BAccMod_clone>`

`python setup.py install`

## SimBMVtool

`git clone https://github.com/MSCarrasco/SimBMVtool.git <path_to_SimBMVtool_clone>`

`cd <path_to_SimBMVtool_clone>`

`gammapy download datasets`

Change the path to gammapy catalog and data accordingly in the config files if you store it elsewhere

Now you can use the notebooks and follow the tutorials
