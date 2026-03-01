from vgworld.world.misc.enums import WorldNoiseName, WorldMatrixName, WorldParameterName
from virigir_math_utilities import Matrix2D
from virigir_math_utilities.noise.core import NoiseGenerator


def run_elevation_generation_pipeline(noise: dict[WorldNoiseName, NoiseGenerator],
                                      matrix: dict[WorldMatrixName, Matrix2D],
                                      parameters: dict[WorldParameterName, float | int],
                                      init_x: int, final_x: int, init_y: int, final_y: int) -> None:
    # noise_matrix
    cont: Matrix2D = Matrix2D.from_noise(noise[WorldNoiseName.continentality],
                                         init_x, final_x, init_y, final_y)
    pv: Matrix2D = Matrix2D.from_noise(noise[WorldNoiseName.peaks_and_valleys],
                                       init_x, final_x, init_y, final_y)
    matrix[WorldMatrixName.elevation] = (cont * parameters[WorldParameterName.sea_scale]) * (
                pv * parameters[WorldParameterName.peaks_and_valleys_scale])


# Deprecated
def fill_continental_elevation(matrix_to_fill: Matrix2D,
                               base_elevation_noise: NoiseGenerator,
                               peaks_and_valleys_noise: NoiseGenerator, continentality_noise: NoiseGenerator,
                               continental_scale: float, peaks_and_valleys_scale: float, sea_scale: float,
                               min_continental_height: float, init_x: int, final_x: int, init_y: int,
                               final_y: int) -> None:
    # generamos las matrices intermedias
    pv: Matrix2D = Matrix2D.from_noise(peaks_and_valleys_noise, init_x, final_x, init_y,
                                       final_y) * peaks_and_valleys_scale
    sea: Matrix2D = Matrix2D.from_noise(continentality_noise, init_x, final_x, init_y, final_y) * sea_scale
    land: Matrix2D = pv - sea
    base_elevation: Matrix2D = Matrix2D.from_noise(base_elevation_noise, init_x, final_x, init_y, final_y)
    # estos son los valores generados
    generated_matrix: Matrix2D = (base_elevation * land) * continental_scale
    generated_matrix = generated_matrix.clamp_values(min_continental_height, 1.0)
    # rellenamos la región indicada de la matriz con los valores generados
    matrix_to_fill.set_submatrix(init_x, final_x, init_y, final_y, generated_matrix)


def fill_is_volcanic_land(matrix_to_fill: Matrix2D, volcanic_noise: NoiseGenerator, island_threshold_level: float,
                          init_x: int, final_x: int, init_y: int, final_y: int) -> None:
    generated_matrix: Matrix2D = Matrix2D.from_noise(volcanic_noise, init_x, final_x, init_y, final_y)
    # devolvemos una matrix binaria
    generated_matrix.binarize(island_threshold_level)
    matrix_to_fill.set_submatrix(init_x, final_x, init_y, final_y, generated_matrix)


def fill_elevation(matrix_to_fill: Matrix2D, is_volcanic_land: Matrix2D, continental_elevation: Matrix2D,
                   init_x: int, final_x: int, init_y: int, final_y: int) -> None:
    # TODO: no estamos considerando el suelo volcánico para la elevación.
    generated_matrix: Matrix2D = continental_elevation.get_submatrix(init_x, final_x, init_y, final_y)
    matrix_to_fill.set_submatrix(init_x, final_x, init_y, final_y, generated_matrix)


def fill_is_continent(matrix_to_fill: Matrix2D, elevation: Matrix2D, sea_elevation_threshold: float, init_x: int,
                      final_x: int, init_y: int, final_y: int) -> None:
    generated_matrix: Matrix2D = elevation.get_submatrix(init_x, final_x, init_y, final_y)
    generated_matrix.binarize(sea_elevation_threshold)
    matrix_to_fill.set_submatrix(init_x, final_x, init_y, final_y, generated_matrix)
