from .utils import read_yaml
from .transformer import TransformerFactoryBase


class TransformerPipeline: 
    def __init__(self, cfgpath: str):

        self.config = read_yaml(cfgpath)
        self.pipeline = []

    def _add_step(self, current_name: str):


    def _build(self): 

        # find the starting point and look into corrections
        start_func = None
        start_name = None
        for key, node in self.config["Transformers"].items():

            if "name" not in node:
                raise ValueError(
                    "Error, each node of a pipeline must have a config node"
                )

            if node["active"] is False:
                continue

            if node["depends_on"] is None and start is None:
                start_func = globals()[node["name"]]
                start_name = node["name"]
            elif node["depends_on"] is None and start is not None:
                raise ValueError("There can only be one starting point.")
            else:
                continue
        
        self.pipeline = [start_func,]

        current_name = start_name

        for key, node in self.config["Transformers"].items():
            if current_name == node["depends_on"]:
                self.pipeline.append(globals()[node["name"]])
                current_name = node["name"]




    def run(self):
        for key, step in self.structure:
            pass
class TransformerPipelineDAG(TransformerPipeline):


    def build(self):

        # find the starting point
        start = None

        for key, node in self.config["Transformers"]:

            if "name" not in node:
                raise ValueError(
                    "Error, each node of a pipeline must have a config node"
                )

            if node.active is False:
                continue

            if node["depends_on"] is None and start is None:
                start = (key, globals()[node["name"]])
            elif node["depends_on"] is None and start is not None:
                raise ValueError("There can only be one starting point.")
            else:
                continue

        self.pipeline.append(start)

        current_name = self.start[0]

        def update_dag(self): 

