from enum import Enum
from pathlib import Path

import tomllib

from virigir_math_utilities import Matrix2D

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "world_configs.toml"


class WorldParameterName(Enum):
    GlobalSeed = "GlobalSeed"
    WorldSizeX = "WorldSizeX"
    WorldSizeY = "WorldSizeY"
    EquatorLatitude = "EquatorLatitude"
    MinContinentalHeight = "MinContinentalHeight"
    PeaksAndValleysScale = "PeaksAndValleysScale"
    ContinentalScale = "ContinentalScale"
    SeaScale = "SeaScale"
    SeaElevationThreshold = "SeaElevationThreshold"
    IslandScale = "IslandScale"
    VolcanicIslandScale = "VolcanicIslandScale"
    IslandThreshold = "IslandThreshold"
    OutToSeaFactor = "OutToSeaFactor"


class WorldMatrixName(Enum):
    # todo: faltan más
    Latitude = "Latitude"
    Elevation = "Elevation"
    ContinentalElevation = "ContinentalElevation"
    IsVolcanicLand = "IsVolcanicLand"
    IsContinent = "IsContinent"
    River = "River"
    RiverBirthPositions = "RiverBirthPositions"
    RiverFlow = "RiverFlow"
    Temperature = "Temperature"

class WorldGenerationStage(Enum):
    Latitude = 0
    Elevation = 1
    River = 2
    Temperature = 3




class VGWorld:

    parameters: dict[WorldParameterName, float | int]
    matrix: list[Matrix2D]

    def __init__(self, config_name: str = "default"):
        self.matrix = []
        self.load_parameters_from_toml(config_name)

    def load_parameters_from_toml(self, config_name: str) -> None:
        with open(DEFAULT_CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
        self.parameters = {
            WorldParameterName(k): v for k, v in raw[config_name].items()
        }

    def run_generation_pipeline_for_region(self, init_x: int, final_x:int, init_y: int, final_y: int):
        for stage in WorldGenerationStage:
            self.run_generation_stage_for_region(stage, init_x, init_y, final_x, final_y)


    def run_generation_stage_for_region(self, stage: WorldGenerationStage, init_x: int, final_x:int, init_y: int, final_y: int):
        if stage is WorldGenerationStage.Latitude:
            self.run_latitude_stage_for_region(init_x, final_x, init_y, final_y)
        elif stage is WorldGenerationStage.Elevation:
            self.run_elevation_stage_for_region(init_x, final_x, init_y, final_y)
        elif stage is WorldGenerationStage.River:
            self.run_river_stage_for_region(init_x, final_x, init_y, final_y)
        elif stage is WorldGenerationStage.Temperature:
            self.run_temperature_stage_for_region(init_x, final_x, init_y, final_y)

    def run_latitude_stage_for_region(self, init_x: int, final_x:int, init_y: int, final_y: int):
        pass

    def run_elevation_stage_for_region(self, init_x: int, final_x:int, init_y: int, final_y: int):
        pass

    def run_river_stage_for_region(self, init_x: int, final_x:int, init_y: int, final_y: int):
        pass

    def run_temperature_stage_for_region(self, init_x: int, final_x:int, init_y: int, final_y: int):
        pass
