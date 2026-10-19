"""Построение гистограмм для задачи 4 (matrix_statistics)."""

import matplotlib.pyplot as plt

from grader_contracts.numpy_tasks import MatrixStatistics


def plot_matrix_statistics(stats: MatrixStatistics) -> None:
    # Строит гистограммы средних и дисперсий по строкам и столбцам
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(stats.row_means, bins="auto")
    axes[0, 0].set_title("Средние по строкам")

    axes[0, 1].hist(stats.column_means, bins="auto")
    axes[0, 1].set_title("Средние по столбцам")

    axes[1, 0].hist(stats.row_variances, bins="auto")
    axes[1, 0].set_title("Дисперсии по строкам")

    axes[1, 1].hist(stats.column_variances, bins="auto")
    axes[1, 1].set_title("Дисперсии по столбцам")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    from grader_contracts.numpy_tasks import RandomMatrixInput
    from numpy_tasks import matrix_statistics

    stats = matrix_statistics(RandomMatrixInput(rows=50, columns=50, mean=0.0, std=1.0, seed=42))
    plot_matrix_statistics(stats)
