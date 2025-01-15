import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from bmv_creator import BMVCreator

class ExternalBMVCreator(BMVCreator):

    def __init__(self, config_path:str, real_data=True, external_model=False):
        super().__init__(config_path=config_path, external_data=True, real_data=real_data, external_model=external_model)

    