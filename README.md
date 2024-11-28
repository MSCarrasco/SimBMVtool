# Installation steps

`git clone https://github.com/MSCarrasco/gammapy.git <path_to_gammapy_clone>`

`cd <path_to_gammapy_clone>`

`git checkout altaz_bkg_irf_rotation`

`conda env create -f environment-dev.yml -n <your_env_name>`

`conda activate <your_env_name>`

`python -m pip install .`

`git clone https://github.com/MSCarrasco/acceptance_modelisation.git <path_to_accmodel_clone>`

`cd <path_to_accmodel_clone>`

`git checkout SimBMVtool_compatible`

`python setup.py install`

`pip install seaborn`
