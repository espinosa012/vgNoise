from enum import Enum
from pathlib import Path

import tomllib

from virigir_math_utilities import Matrix2D
from generation import elevation
from virigir_math_utilities.noise.core import NoiseGenerator
from virigir_math_utilities.noise.generators.noise2d import NoiseGenerator2D

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "world_configs.toml"

# TODO: cuando implementemos las pipelines, pedir un README.md del world

class WorldNoiseName(Enum):
    base_elevation = "Continentality"
    continentality = "Continentality"
    peaks_and_valleys = "PeaksAndValleys"
    volcanic_noise = "VolcanicNoise"

class WorldParameterName(Enum):
    global_seed = "GlobalSeed"
    world_size_x = "WorldSizeX"
    world_size_y = "WorldSizeY"
    equator_latitude = "EquatorLatitude"
    min_continental_height = "MinContinentalHeight"
    peaks_and_valleys_scale = "PeaksAndValleysScale"
    continental_scale = "ContinentalScale"
    sea_scale = "SeaScale"
    sea_elevation_threshold = "SeaElevationThreshold"
    island_scale = "IslandScale"
    volcanic_island_scale = "VolcanicIslandScale"
    island_threshold = "IslandThreshold"
    out_to_sea_factor = "OutToSeaFactor"


class WorldMatrixName(Enum):
    # todo: faltan más. quizás deberíamos definirlas en el json, teniendo un enum para mantenerlas a todas
    latitude = "Latitude"

    elevation = "Elevation"
    continental_elevation = "ContinentalElevation"
    is_volcanic_land = "IsVolcanicLand"
    is_continent = "IsContinent"

    river = "River"
    river_birth_positions = "RiverBirthPositions"
    river_flow = "RiverFlow"

    temperature = "Temperature"


class WorldGenerationStage(Enum):
    # TODO: quizás podríamos definir esto también en el json, para poder agregar etapas sin tocar código.
    latitude = 0
    elevation = 1
    river = 2
    temperature = 3


class VGWorld:
    parameters: dict[WorldParameterName, float | int]
    noise: dict[WorldNoiseName, NoiseGenerator]
    matrix: dict[WorldMatrixName, Matrix2D]

    def __init__(self, config_name: str = "default_parameters"):
        self.load_parameters_from_toml(config_name)
        self.initialize_noise(config_name)
        self.initialize_matrix()

    def load_parameters_from_toml(self, config_name: str) -> None:
        with open(DEFAULT_CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
        self.parameters = {
            WorldParameterName(k): v for k, v in raw[config_name].items()
        }

    def initialize_matrix(self):
        for matrix_name in WorldMatrixName:
            self.matrix[matrix_name] = Matrix2D((self.parameters[WorldParameterName.world_size_x],
                                                 self.parameters[WorldParameterName.world_size_y]))

    def initialize_noise(self, config_name: str):
        with open(DEFAULT_CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
        self.noise = {
            WorldNoiseName(name): NoiseGenerator2D.from_dict(noise_data)
            for name, noise_data in raw.get(config_name, {}).get("noise", {}).items()
        }

    # Generation
    def run_generation_pipeline_for_region(self, init_x: int, final_x: int, init_y: int, final_y: int):
        for stage in WorldGenerationStage:
            self.run_generation_stage_for_region(stage, init_x, init_y, final_x, final_y)

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
        pass

    def run_elevation_stage_for_region(self, init_x: int, final_x: int, init_y: int, final_y: int):
        elevation.fill_continental_elevation(self.matrix[WorldMatrixName.continental_elevation],
                                             self.noise[WorldNoiseName.base_elevation],
                                             self.noise[WorldNoiseName.peaks_and_valleys],
                                             self.noise[WorldNoiseName.continentality],
                                             self.parameters[WorldParameterName.continental_scale],
                                             self.parameters[WorldParameterName.peaks_and_valleys_scale],
                                             self.parameters[WorldParameterName.sea_scale],
                                             self.parameters[WorldParameterName.min_continental_height],
                                             init_x, final_x, init_y, final_y)

    def run_river_stage_for_region(self, init_x: int, final_x: int, init_y: int, final_y: int):
        pass

    def run_temperature_stage_for_region(self, init_x: int, final_x: int, init_y: int, final_y: int):
        pass
