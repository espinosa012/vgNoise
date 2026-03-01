from enum import Enum


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
