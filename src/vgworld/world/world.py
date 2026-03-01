import json
from enum import Enum
from pathlib import Path

import tomllib

from vgworld.world.misc.enums import WorldParameterName, WorldMatrixName, WorldNoiseName, WorldGenerationStage
from virigir_math_utilities import Matrix2D
from virigir_math_utilities.noise.core import NoiseGenerator
from virigir_math_utilities.noise.generators.noise2d import NoiseGenerator2D
from vgworld.world.generation import elevation, latitude

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "world_configs.toml"
DEFAULT_NOISE_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "config.json"


# TODO: cuando implementemos las pipelines, pedir un README.md del world

# TODO: necesitamos método para regenerar, volver a cargar noises y params

class VGWorld:
    parameters: dict[WorldParameterName, float | int]
    noise: dict[WorldNoiseName, NoiseGenerator]
    matrix: dict[WorldMatrixName, Matrix2D]

    def __init__(self, config_name: str = "default_parameters"):
        self.regenerate_world(config_name)
        # TODO: pruebas
        self.run_generation_pipeline_for_region(0, self.parameters[WorldParameterName.world_size_x], 0,
                                                self.parameters[WorldParameterName.world_size_y])

    def regenerate_world(self, config_name: str = "default_parameters"):
        self.load_parameters_from_toml(config_name)
        self.initialize_noise(config_name)
        self.initialize_matrix()

    def load_parameters_from_toml(self, config_name: str) -> None:
        self.parameters = {}
        with open(DEFAULT_CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
        self.parameters = {
            WorldParameterName(k): v for k, v in raw[config_name].items()
        }

    def initialize_matrix(self):
        """Allocate world-size matrices for all WorldMatrixName entries."""
        self.matrix = {}
        for matrix_name in WorldMatrixName:
            self.matrix[matrix_name] = Matrix2D((self.parameters[WorldParameterName.world_size_x],
                                                 self.parameters[WorldParameterName.world_size_y]))

    def initialize_noise(self, config_name: str):
        self.noise = {}
        with open(DEFAULT_NOISE_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for name, noise_data in raw.get("noise", {}).items():
            try:
                key = WorldNoiseName[name]
                self.noise[key] = NoiseGenerator2D.from_dict(noise_data)
            except KeyError:
                pass  # noise key not in WorldNoiseName enum, skip

    # Generation
    def run_generation_pipeline_for_region(self, init_x: int, final_x: int, init_y: int, final_y: int):
        # TODO: comprobar que la región es válida (dentro de los límites del mundo, init < final, etc)
        for stage in WorldGenerationStage:
            self.run_generation_stage_for_region(stage, init_x, final_x, init_y, final_y)

    def run_generation_stage_for_region(self, stage: WorldGenerationStage, init_x: int, final_x: int, init_y: int,
                                        final_y: int):
        if stage is WorldGenerationStage.latitude:
            self.run_latitude_stage_for_region(init_x, final_x, init_y, final_y)
        elif stage is WorldGenerationStage.elevation:
            self.run_elevation_stage_for_region(init_x, final_x, init_y, final_y)
        elif stage is WorldGenerationStage.river:
            self.run_river_stage_for_region(init_x, final_x, init_y, final_y)
        elif stage is WorldGenerationStage.temperature:
            self.run_temperature_stage_for_region(init_x, final_x, init_y, final_y)

    """Pasamos las matrices vacías a los métodos que rellenan valores y las recuperamos ya rellenas"""

    def run_latitude_stage_for_region(self, init_x: int, final_x: int, init_y: int, final_y: int):
        latitude.fill_latitude(self.matrix[WorldMatrixName.latitude],
                               self.parameters[WorldParameterName.equator_latitude],
                               self.parameters[WorldParameterName.world_size_x],
                               self.parameters[WorldParameterName.world_size_y],
                               init_x, final_x, init_y, final_y)

    def run_elevation_stage_for_region(self, init_x: int, final_x: int, init_y: int, final_y: int):
        elevation.run_elevation_generation_pipeline(self.noise, self.matrix, self.parameters, init_x, final_x, init_y, final_y)
        """
        elevation.fill_continental_elevation(self.matrix[WorldMatrixName.continental_elevation],
                                             self.noise[WorldNoiseName.base_elevation],
                                             self.noise[WorldNoiseName.peaks_and_valleys],
                                             self.noise[WorldNoiseName.continentality],
                                             self.parameters[WorldParameterName.continental_scale],
                                             self.parameters[WorldParameterName.peaks_and_valleys_scale],
                                             self.parameters[WorldParameterName.sea_scale],
                                             self.parameters[WorldParameterName.min_continental_height],
                                             init_x, final_x, init_y, final_y)

        elevation.fill_is_volcanic_land(self.matrix[WorldMatrixName.is_volcanic_land],
                                        self.noise[WorldNoiseName.volcanic_noise],
                                        self.parameters[WorldParameterName.island_threshold],
                                        init_x, final_x, init_y, final_y)

        elevation.fill_elevation(self.matrix[WorldMatrixName.elevation],
                                 self.matrix[WorldMatrixName.is_volcanic_land],
                                 self.matrix[WorldMatrixName.continental_elevation],
                                 init_x, final_x, init_y, final_y)

        elevation.fill_is_continent(self.matrix[WorldMatrixName.is_continent], self.matrix[WorldMatrixName.elevation],
                                    self.parameters[WorldParameterName.sea_elevation_threshold], init_x, final_x,
                                    init_y, final_y)
        """

    def run_river_stage_for_region(self, init_x: int, final_x: int, init_y: int, final_y: int):
        pass

    def run_temperature_stage_for_region(self, init_x: int, final_x: int, init_y: int, final_y: int):
        pass

    # ------------------------------------------------------------------
    # Movement / Pathfinding
    # ------------------------------------------------------------------

    def is_walkable(self, x: int, y: int) -> bool:
        """
        Return whether the world cell at (x, y) can be walked on.

        This is the single point where world rules about passability live.
        The tilemap stores *what* is at each cell; this method decides
        *what that means* for movement (e.g. deep water blocks, forests
        slow but don't block, etc.).

        TODO: implement actual walkability logic based on world matrices
              and tile semantics once the biome / tile system is defined.

        Args:
            x: Grid x coordinate.
            y: Grid y coordinate.

        Returns:
            True if the cell is passable, False otherwise.
        """
        raise NotImplementedError(
            "is_walkable() must be implemented once tile semantics are defined."
        )

