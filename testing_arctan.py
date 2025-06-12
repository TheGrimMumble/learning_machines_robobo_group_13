import numpy as np


def dist_from_origin_reward(y, x):
    distance = np.sqrt(y**2 + x**2)
    angle = np.arctan2(y, x)
    ideal_angle = np.pi / 4  # y = x
    angle_diff = np.abs(np.arctan2(
        np.sin(angle - ideal_angle),
        np.cos(angle - ideal_angle)
        ))
    direction_score = np.cos(angle_diff)  # 1 = perfect, -1 = opposite
    
    return distance, direction_score


testing = [
    (1,1),
    (-1,-1),
    (30, 30),
    (40, 20),
    (20, 40),
    (-10, -15),
    (5, -30),
    (-10, 100),
    (-1, 1)
]

for vec in testing:
    y, x = vec
    distance, angle = dist_from_origin_reward(y, x)
    print(vec, round(distance, 3), round(angle, 3))
