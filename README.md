# Installation steps

`git clone git@github.com:gabemery/gammapy.git <path_to_gammapy_clone>`

`cd <path_to_gammapy_clone>`

`git checkout altaz_bkg_irf_rotation`

`conda env create -f environment-dev.yml -n <your_env_name>`

`conda activate <your_env_name>`

`python -m pip install .`

`git clone https://github.com/MSCarrasco/acceptance_modelisation/tree/correct_alignment_transform_obs_to_camera_frame <path_to_accmodel_clone>`

`cd <path_to_accmodel_clone>`

`python setup.py install`

`pip install seaborn`
