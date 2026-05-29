def find_farthest_pair(points):
    max_dist_sq = -1
    best = (0, 1)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist_sq = dx * dx + dy * dy
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                best = (i, j)
    return best


def find_farthest_triple(points):
    max_dist_sum = -1
    best = (0, 1, 2)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                dx_ij = points[i][0] - points[j][0]
                dy_ij = points[i][1] - points[j][1]
                dx_ik = points[i][0] - points[k][0]
                dy_ik = points[i][1] - points[k][1]
                dx_jk = points[j][0] - points[k][0]
                dy_jk = points[j][1] - points[k][1]
                dist_sum = (
                    (dx_ij**2 + dy_ij**2) ** 0.5
                    + (dx_ik**2 + dy_ik**2) ** 0.5
                    + (dx_jk**2 + dy_jk**2) ** 0.5
                )
                if dist_sum > max_dist_sum:
                    max_dist_sum = dist_sum
                    best = (i, j, k)
    return best
