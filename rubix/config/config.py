from rubix.utils import read_yaml
import os 


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubix_config.yml")



class Config:
    
    @staticmethod
    def load() -> dict:
        return read_yaml(CONFIG_PATH)
    
