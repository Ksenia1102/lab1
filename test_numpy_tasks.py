import numpy as np
import pytest

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series,
    binarize,
    chess,
    draw_ellipse,
    draw_rectangle,
    matrix_statistics,
    one_hot,
    sum_prod,
    unique_columns,
    unique_rows,
)


# sum_prod 

def test_sum_prod_basic():
    # Проверяем умножение матриц на векторы с последующим суммированием
    matrices = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    vectors = np.array([[[1], [1]], [[1], [1]]])
    # A1x1 = [1,1]^T, A2x2 = [2,2]^T, сумма = [3,3]^T
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    np.testing.assert_array_equal(result, [[3], [3]])


def test_sum_prod_single_matrix():
    # Проверяем работу функции с единственной матрицей
    matrices = np.array([[[1, 2], [3, 4]]])
    vectors = np.array([[[1], [1]]])
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    np.testing.assert_array_equal(result, [[3], [7]])


#  binarize 

def test_binarize_basic():
    # Проверяем замену элементов матрицы по заданному порогу
    matrix = np.array([[0.1, 0.6], [0.5, 0.9]])
    result = binarize(BinarizeInput(matrix, threshold=0.5))
    np.testing.assert_array_equal(result, [[0, 1], [0, 1]])


def test_binarize_does_not_mutate_original():
    # Проверяем, что функция бинаризации не изменяет исходную матрицу
    matrix = np.array([[0.1, 0.6]])
    original = matrix.copy()
    binarize(BinarizeInput(matrix, threshold=0.5))
    np.testing.assert_array_equal(matrix, original)


def test_binarize_boundary_value_not_included():
    # Проверяем строгость неравенства на границе порога
    # строго больше threshold, на границе должно быть 0
    matrix = np.array([[0.5]])
    result = binarize(BinarizeInput(matrix, threshold=0.5))
    np.testing.assert_array_equal(result, [[0]])


#  unique_rows / unique_columns 

def test_unique_rows_basic():
    # Проверяем поиск уникальных элементов для каждой строки матрицы
    matrix = np.array([[1, 2, 2], [3, 3, 3]])
    assert unique_rows(MatrixInput(matrix)) == [[1, 2], [3]]


def test_unique_columns_basic():
    # Проверяем поиск уникальных элементов для каждого столбца матрицы
    matrix = np.array([[1, 2], [1, 3]])
    assert unique_columns(MatrixInput(matrix)) == [[1], [2, 3]]


def test_unique_rows_all_distinct():
    # Проверяем матрицу, у которой все элементы в строках уже уникальны
    matrix = np.array([[1, 2, 3]])
    assert unique_rows(MatrixInput(matrix)) == [[1, 2, 3]]


#  matrix_statistics 

def test_matrix_statistics_shapes():
    # Проверяем правильность форм возвращаемых массивов статистик случайной матрицы
    stats = matrix_statistics(RandomMatrixInput(rows=5, columns=4, mean=0.0, std=1.0, seed=1))
    assert stats.matrix.shape == (5, 4)
    assert stats.row_means.shape == (5,)
    assert stats.column_means.shape == (4,)
    assert stats.row_variances.shape == (5,)
    assert stats.column_variances.shape == (4,)


def test_matrix_statistics_reproducible_with_seed():
    # Проверяем воспроизводимость генерации случайной матрицы при использовании seed
    stats1 = matrix_statistics(RandomMatrixInput(rows=3, columns=3, seed=42))
    stats2 = matrix_statistics(RandomMatrixInput(rows=3, columns=3, seed=42))
    np.testing.assert_array_equal(stats1.matrix, stats2.matrix)


#  chess 

def test_chess_example():
    # Проверяем шахматное заполнение матрицы по размеру и значению
    result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    expected = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])
    np.testing.assert_array_equal(result, expected)


def test_chess_single_cell():
    # Проверяем построение шахматной матрицы минимального размера 1x1
    result = chess(ChessInput(rows=1, columns=1, first=5, second=9))
    np.testing.assert_array_equal(result, [[5]])


#  draw_rectangle 

def test_draw_rectangle_shape_and_background():
    # Проверяем размер создаваемого изображения и цвет фона для прямоугольника
    image = draw_rectangle(RectangleInput(
        width=2, height=2, image_height=6, image_width=6,
        shape_color=(255, 0, 0), background_color=(0, 0, 0),
    ))
    assert image.shape == (6, 6, 3)
    np.testing.assert_array_equal(image[0, 0], [0, 0, 0])


def test_draw_rectangle_center_pixel_is_shape_color():
    # Проверяем, что центральный пиксель изображения закрашен цветом прямоугольника
    image = draw_rectangle(RectangleInput(
        width=2, height=2, image_height=6, image_width=6,
        shape_color=(255, 0, 0), background_color=(0, 0, 0),
    ))
    np.testing.assert_array_equal(image[3, 3], [255, 0, 0])


#  draw_ellipse 

def test_draw_ellipse_shape():
    # Проверяем итоговую форму массива RGB-изображения с эллипсом
    image = draw_ellipse(EllipseInput(
        semi_axis_x=2, semi_axis_y=2, image_height=8, image_width=8,
        shape_color=(0, 255, 0), background_color=(0, 0, 0),
    ))
    assert image.shape == (8, 8, 3)


def test_draw_ellipse_center_is_shape_color():
    # Проверяем, что центр эллипса закрашен цветом фигуры
    image = draw_ellipse(EllipseInput(
        semi_axis_x=3, semi_axis_y=3, image_height=10, image_width=10,
        shape_color=(0, 255, 0), background_color=(0, 0, 0),
    ))
    np.testing.assert_array_equal(image[5, 5], [0, 255, 0])


def test_draw_ellipse_corner_is_background():
    # Проверяем, что угловые пиксели за пределами эллипса остаются цветом фона
    image = draw_ellipse(EllipseInput(
        semi_axis_x=2, semi_axis_y=2, image_height=10, image_width=10,
        shape_color=(0, 255, 0), background_color=(1, 2, 3),
    ))
    np.testing.assert_array_equal(image[0, 0], [1, 2, 3])


#  analyze_time_series 

def test_analyze_time_series_basic():
    # Проверяем расчет статистик, локальных экстремумов и скользящего среднего временного ряда
    values = [1, 3, 2, 5, 1]
    stats = analyze_time_series(TimeSeriesInput(values=values, window=2))
    assert stats.mean == pytest.approx(np.mean(values))
    assert stats.variance == pytest.approx(np.var(values))
    assert stats.std == pytest.approx(np.std(values))
    np.testing.assert_array_equal(stats.local_maxima_indices, [1, 3])
    np.testing.assert_array_equal(stats.local_minima_indices, [2])
    expected_moving_average = [(1 + 3) / 2, (3 + 2) / 2, (2 + 5) / 2, (5 + 1) / 2]
    np.testing.assert_allclose(stats.moving_average, expected_moving_average)


def test_analyze_time_series_no_extrema():
    # Проверяем монотонный временной ряд, не содержащий локальных экстремумов
    values = [1, 2, 3, 4]
    stats = analyze_time_series(TimeSeriesInput(values=values, window=2))
    assert len(stats.local_maxima_indices) == 0
    assert len(stats.local_minima_indices) == 0


def test_analyze_time_series_moving_average_full_window():
    # Проверяем скользящее среднее с размером окна, равным длине всего ряда
    values = [1, 2, 3, 4]
    stats = analyze_time_series(TimeSeriesInput(values=values, window=4))
    np.testing.assert_allclose(stats.moving_average, [2.5])


#  one_hot

def test_one_hot_example():
    # Проверяем построение кодирования для вектора из примера
    result = one_hot(OneHotInput(labels=[0, 2, 3, 0]))
    expected = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])
    np.testing.assert_array_equal(result, expected)


def test_one_hot_explicit_class_count():
    # Проверяем формирование представления с явно заданным количеством классов
    result = one_hot(OneHotInput(labels=[0, 1], class_count=5))
    assert result.shape == (2, 5)


def test_one_hot_single_label():
    # Проверяем кодирование для вектора из одного элемента
    result = one_hot(OneHotInput(labels=[0]))
    np.testing.assert_array_equal(result, [[1]])
