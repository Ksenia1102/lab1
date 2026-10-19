"""Заготовки задач на NumPy."""

import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


# Задание 1
def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    return np.einsum("pij,pjk->ik", matrices, vectors)  # считает сумму всех A_i @ x_i сразу


# Задание 2
def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    matrix = np.asarray(matrix)
    return np.where(matrix > threshold, 1, 0)


# Задание 3 - строки
def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


# Задание 3 - столбцы
def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(col).tolist() for col in matrix.T]


# Задание 4
def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    row_variances = matrix.var(axis=1)
    column_variances = matrix.var(axis=0)
    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances,
    )


# Задание 5
def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    parity = np.indices((rows, columns)).sum(axis=0) % 2  # собираем чётность индексов через np.indices
    return np.where(parity == 0, first, second)


# Задание 6 - прямоугольник
def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color

    center_y, center_x = image_height // 2, image_width // 2
    half_h, half_w = height // 2, width // 2
    top, left = center_y - half_h, center_x - half_w
    bottom, right = top + height, left + width

    image[top:bottom, left:right] = shape_color
    return image


# Задание 6 - эллипс
def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color

    y0, x0 = image_height // 2, image_width // 2
    yy, xx = np.mgrid[0:image_height, 0:image_width]
    mask = ((xx - x0) ** 2) / (semi_axis_x ** 2) + ((yy - y0) ** 2) / (semi_axis_y ** 2) <= 1

    image[mask] = shape_color
    return image


# Задание 7
def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)

    mean = float(values.mean())
    variance = float(values.var())
    std = float(values.std())

    local_maxima = []
    local_minima = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            local_maxima.append(i)
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_minima.append(i)

    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=np.array(local_maxima, dtype=int),
        local_minima_indices=np.array(local_minima, dtype=int),
        moving_average=moving_average,
    )


# Задание 8
def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels)

    if class_count is None:
        class_count = int(labels.max()) + 1

    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result


if __name__ == "__main__":
    print("Ручная проверка\n")

    # Задание 1
    matrices = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    vectors = np.array([[[1], [1]], [[1], [1]]])
    res_1 = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    print(f"1. Сумма произведений матриц на вектор:\n{res_1}\n")

    # Задание 2
    mat_2 = np.array([[0.1, 0.6], [0.5, 0.9]])
    res_2 = binarize(BinarizeInput(mat_2, threshold=0.5))
    print(f"2. Бинаризация матрицы:\n{res_2}\n")

    # Задание 3
    mat_3 = np.array([[1, 2, 2], [3, 3, 3]])
    res_3_rows = unique_rows(MatrixInput(mat_3))
    res_3_cols = unique_columns(MatrixInput(mat_3))
    print(f"3. Уникальные элементы строк: {res_3_rows}")
    print(f"3. Уникальные элементы столбцов: {res_3_rows}")

    # Задание 4
    stats_4 = matrix_statistics(RandomMatrixInput(rows=3, columns=3, mean=0.0, std=1.0, seed=42))
    print(f"4. Случайная матрица:\n{stats_4.matrix}")
    print(f"4. row_means: {stats_4.row_means}\n")

    # Задание 5
    res_5 = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    print(f"5. Шахматная матрица:\n{res_5}\n")

    # Задание 6
    rect_6 = draw_rectangle(RectangleInput(
        width=2, height=2, image_height=6, image_width=6,
        shape_color=(255, 0, 0), background_color=(0, 0, 0),
    ))
    ellipse_6 = draw_ellipse(EllipseInput(
        semi_axis_x=2, semi_axis_y=2, image_height=8, image_width=8,
        shape_color=(0, 255, 0), background_color=(0, 0, 0),
    ))
    print(f"6. Форма изображения прямоугольника: {rect_6.shape}")
    print(f"6. Форма изображения эллипса: {ellipse_6.shape}\n")

    # Задание 7
    ts_res_7 = analyze_time_series(TimeSeriesInput(values=[1, 3, 2, 5, 1], window=2))
    print(f"7. Среднее: {ts_res_7.mean}")
    print(f"7. Локальные максимумы:: {ts_res_7.local_maxima_indices}\n")

    # Задание 8
    res_8 = one_hot(OneHotInput(labels=[0, 2, 3, 0]))
    print(f"8. one_hot результат:\n{res_8}")