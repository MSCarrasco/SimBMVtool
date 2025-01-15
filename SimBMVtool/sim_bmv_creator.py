import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from bmv_creator import BMVCreator

class SimBMVCreator(BMVCreator):

    def __init__(self, config_path:str, external_model=False):
        super().__init__(config_path=config_path, external_data=False, real_data=False, external_model=external_model)

    