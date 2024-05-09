from rubix.pipeline import linear_pipeline as pipeline
from rubix.pipeline import transformer as transformer
from rubix.utils import get_config
from rubix.logger import get_logger
from .data import get_rubix_data
from pathlib import Path
from .rotation import get_galaxy_rotation
from .telescope import get_spaxel_assignment, get_split_data
from typing import Union
import time


class RubixPipeline:

    def __init__(self, user_config: Union[dict, str]):
        self.user_config = get_config(user_config)
        self.pipeline_config = _get_pipeline_config(
            self.user_config["pipeline"]["name"]
        )
        self.logger = get_logger(self.user_config["logger"])

        self.data = self._prepare_data()
        self.func = None

    def _prepare_data(self):
        # Get the data
        self.logger.info("Getting rubix data...")
        coords, velocities, metallicity, mass, age = get_rubix_data(self.user_config)
        self.logger.info(f"Data loaded with {len(coords)} particles.")

        # Setup the data dictionary
        # TODO: This is a temporary solution, we need to figure out a better way to handle the data
        data = {
            "coords": coords,
            "velocities": velocities,
            "metallicity": metallicity,
            "mass": mass,
            "age": age,
        }
        return data

    def _get_pipeline_functions(self):
        self.logger.info("Setting up the pipeline...")
        self.logger.debug("Pipeline Configuration: %s", self.pipeline_config)
        rotate_galaxy = get_galaxy_rotation(self.user_config)
        spaxel_assignment = get_spaxel_assignment(self.user_config)
        split_data = get_split_data(self.user_config)
        functions = [rotate_galaxy, spaxel_assignment, split_data]
        return functions

    def run(self):
        # Create the pipeline
        time_start = time.time()
        functions = self._get_pipeline_functions()
        self._pipeline = pipeline.LinearTransformerPipeline(
            self.pipeline_config, functions
        )

        # Assembling the pipeline
        self.logger.info("Assembling the pipeline...")
        self._pipeline.assemble()

        # Compiling the expressions
        self.logger.info("Compiling the expressions...")
        self.func = self._pipeline.compile_expression()

        # Running the pipeline
        self.logger.info("Running the pipeline on the input data...")
        output = self.func(self.data)

        time_end = time.time()

        self.logger.info(
            "Pipeline run completed in %.2f seconds.", time_end - time_start
        )
        return output

    def gradient(self):
        raise NotImplementedError("Gradient calculation is not implemented yet")


def _get_pipeline_config(name: str):
    config_path = str(Path(__file__).parent / "pipelines.yml")
    pipelines_config = get_config(config_path)

    # Get the pipeline configuration
    if name not in pipelines_config:
        raise ValueError(f"Pipeline {name} not found in the configuration")
    config = pipelines_config[name]
    return config
