from virigir_math_utilities import Matrix2D
from virigir_math_utilities.noise.core import NoiseGenerator


def fill_continental_elevation(matrix_to_fill: Matrix2D, base_elevation_noise: NoiseGenerator,
                               peaks_and_valleys_noise: NoiseGenerator, continentality_matrix: NoiseGenerator,
                               continental_scale: float, peaks_and_valleys_scale: float, sea_scale: float,
                               min_continental_height: float, init_x: int, final_x: int, init_y: int, final_y: int) -> None:
    # generamos las matrices intermedias
    pv_matrix: Matrix2D = peaks_and_valleys_scale * (
        peaks_and_valleys_noise.get_submatrix(init_x, final_x, init_y, final_y))
    sea_matrix: Matrix2D = sea_scale * (continentality_matrix.get_submatrix(init_x, final_x, init_y, final_y))
    land_matrix: Matrix2D = pv_matrix - sea_matrix

    # estos son los valores generados
    generated_matrix: Matrix2D = (base_elevation_noise.get_submatrix(init_x, final_x, init_y, final_y).matmul(
        land_matrix) * continental_scale).clamp_values(min_continental_height, 1.0)

    # rellenamos la región indicada de la matriz con los valores generados
    matrix_to_fill.set_submatrix(init_x, final_x, init_y, final_y, generated_matrix)

