import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize


def skeleton(img: np.ndarray, thresh_low=127, thresh_high=255, thresh_type=cv.THRESH_BINARY) -> np.ndarray:
    ret, thr = cv.threshold(img, thresh_low, thresh_high, thresh_type)
    thr[thr > 0] = 1

    skel = skeletonize(thr).astype(np.uint8)

    S = np.zeros([skel.shape[0]+2, skel.shape[0]+2], dtype=np.uint8)
    S[1:-1, 1:-1] = skel

    return S


def corner_points(skeleton, max_points):
    # fix corner points (move them on to the skeleton's line)

    S = skeleton.copy()

    k = 0.09
    min_distance = 3
    c = cv.goodFeaturesToTrack(S, max_points, k, min_distance)
    c = np.int0(c)

    result = []

    for i in range(c.shape[0]):
        result.append([c[i, 0, 1], c[i, 0, 0]])
    return result


def define_graph(img: np.ndarray, max_points=50):
    S = skeleton(img)
    cpoints = corner_points(S, max_points)

    check = S.copy()
    neighbour_matrix = np.zeros([max_points, max_points])

    # coordinate to node id
    c2id = {}
    # Node ID
    r = 0
    # points between two points
    line = []
    while np.sum(check) > 0:
        print(f"Sum is: {np.sum(check)}")
        # choosing node point
        p = cpoints[0][0]
        q = cpoints[0][1]

        area = check[p - 1:p + 2, q - 1:q + 2]
        if np.sum(area) == 1:
            del cpoints[0]
            check[p, q] = 0
            continue

        if _c2id([p, q]) not in c2id:
            c2id[_c2id([p, q])] = r
            neighbour_matrix[r, r] = 1
            r += 1
        # selecting area
        while True:
            # Copy to prevent backtracking
            check_sub = check.copy()
            check_sub[p, q] = 0
            # refresh area
            area = check_sub[p-1:p+2, q-1:q+2]
            # query for available paths
            paths = np.where(area == 1)
            step_direction = [paths[0][0]-1 if not (paths[0][0] == 1 and paths[1][0] == 1) else paths[0][1]-1,
                              paths[1][0]-1 if not (paths[0][0] == 1 and paths[1][0] == 1) else paths[1][1]-1]
            # adding point to the line
            line.append([p, q])
            # updating the points location
            p += step_direction[0]
            q += step_direction[1]
            # if it is a point from the possible node points
            if [p, q] in cpoints:
                # generate and add node id
                first_id = c2id[_c2id(line[0])]
                second_id = None
                if _c2id([p, q]) not in c2id:
                    c2id[_c2id([p, q])] = r
                    neighbour_matrix[r, r] = 1
                    second_id = r
                    r += 1
                # updating neighbouring matrix
                neighbour_matrix[first_id, second_id] = line.__len__()
                neighbour_matrix[second_id, first_id] = line.__len__()
                # removing line from the original work area of the skeleton
                for l in line[1:]:
                    check[l[0], l[1]] = 0
                # resetting lines
                line = []
                break
    return neighbour_matrix


def _c2id(coordinate: list):
    return f"{coordinate[0]},{coordinate[1]}"
