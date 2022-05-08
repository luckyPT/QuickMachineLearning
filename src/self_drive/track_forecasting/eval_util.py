import math


def fde(label_points, pre_points):
    return math.sqrt((label_points[-1][0] - pre_points[-1][0]) ** 2 + (label_points[-1][1] - pre_points[-1][1]) ** 2)


def ade(label_points, pre_points):
    pre_count = len(pre_points)
    diff_sum = 0
    for i in range(pre_count):
        diff_sum += math.sqrt(
            (label_points[i][0] - pre_points[i][0]) ** 2 + (label_points[i][1] - pre_points[i][1]) ** 2)
    return diff_sum / pre_count
