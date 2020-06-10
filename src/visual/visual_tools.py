import numpy as np
import matplotlib.pyplot as plt

from src.data.sk_data import Iris


def plot_scatter(x, y):
    if x.shape[1] != 2:
        raise Exception("x shape is not lawful")
    label_count = np.unique(y).size
    colors = ['r', 'b', 'g', 'k', 'm', 'y', 'c'][:label_count]
    markers = ['.', 'o', 'v', '^', '<', '>', '1'][:label_count]
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(x[y == l, 0],
                    x[y == l, 1],
                    c=c, label=l, marker=m)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()


def plot_line(*args):
    """
    :param args:
    :return:
    """
    color = ['r', 'g', 'b', 'y', 'k', 'm', 'c']
    for i in range(0, len(args), 2):
        plt.plot(args[i], args[i + 1], color[i // 2], label='type' + str(i // 2))
    plt.title('The Lasers in Three Conditions')
    plt.xlabel('row')
    plt.ylabel('column')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x = Iris.features[:, :2]
    y = Iris.label
    plot_scatter(x, y)

    # line
    x1 = np.arange(0, 10, 0.1)
    y1 = np.sin(x1)
    x2 = np.arange(0, 12, 0.2)
    y2 = np.cos(x2)
    x3 = np.arange(-10, 10, 0.1)
    y3 = 2 * np.cos(x3) + 1
    plot_line(x1, y1, x2, y2, x3, y3)
