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

    neighbours = []
    for i in range(c.shape[0]):
        min = 4000
        point = None
        match = False
        it = 0
        if S[c[i, 0, 1], c[i, 0, 0]] == 0 or S[c[i, 0, 1], c[i, 0, 0]] == 1:
            neighbours = []
            for j in range(2):
                it = j
                for k in range((j + 1) * 2 + 1):
                    for l in range((j + 1) * 2 + 1):
                        if S[c[i, 0, 1] + k - 1, c[i, 0, 0] + l - 1] == 1:
                            neighbours.append((c[i, 0, 1] + k - 1, c[i, 0, 0] + l - 1))

            for p in neighbours:
                #print(S[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2])
                count = np.sum(S[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2])
                #print(count)
                if 0 < count < min:
                    min = count
                    point = p

        if point is not None:
            # print(point)
            c[i, 0, 1] = point[0]
            c[i, 0, 0] = point[1]
            result.append([point[0], point[1]])
    return result


def define_graph(img: np.ndarray, max_points=100):
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
    while cpoints.__len__() > 0:
        print(f"Num of Cpoints: {cpoints.__len__()}")
        # choosing node point
        p = cpoints[0][0]
        q = cpoints[0][1]

        area = check[p - 1:p + 2, q - 1:q + 2]
        if np.sum(area) <= 1:
            del cpoints[0]
            check[p, q] = 0
            continue

        if _c2id([p, q]) not in c2id:
            c2id[_c2id([p, q])] = r
            neighbour_matrix[r, r] = 1
            r += 1
        # selecting area
        # Copy to prevent backtracking
        check_sub = check.copy()
        while True:
            check_sub[p, q] = 0
            # refresh area
            area = check_sub[p-1:p+2, q-1:q+2]
            # query for available paths
            paths = np.where(area == 1)

            # if it is a branching point
            if paths[0].shape[0] > 1 and [p, q] not in cpoints:
                cpoints.append([p, q])
                first_id = c2id[_c2id(line[0])]
                second_id = None
                if _c2id([p, q]) not in c2id:
                    c2id[_c2id([p, q])] = r
                    neighbour_matrix[r, r] = 1
                    second_id = r
                    r += 1
                else:
                    second_id = c2id[_c2id([p, q])]
                # updating neighbouring matrix
                neighbour_matrix[first_id, second_id] = line.__len__()
                neighbour_matrix[second_id, first_id] = line.__len__()
                # removing line from the original work area of the skeleton
                for l in line[1:]:
                    check[l[0], l[1]] = 0
                # resetting lines
                line = []
                break

            # if the point is the end of a line and there is no marked node point
            if paths[0].shape[0] == 0 and [p, q] not in cpoints:
                first_id = c2id[_c2id(line[0])]
                second_id = None
                if _c2id([p, q]) not in c2id:
                    c2id[_c2id([p, q])] = r
                    neighbour_matrix[r, r] = 1
                    second_id = r
                    r += 1
                else:
                    second_id = c2id[_c2id([p, q])]
                # updating neighbouring matrix
                neighbour_matrix[first_id, second_id] = line.__len__()
                neighbour_matrix[second_id, first_id] = line.__len__()
                # removing line from the original work area of the skeleton
                for l in line[1:]:
                    check[l[0], l[1]] = 0
                if line.__len__() == 1:
                    check[p, q] = 0
                # resetting lines
                line = []
                break

            step_direction = [paths[0][0]-1,
                              paths[1][0]-1]
            # adding point to the line
            line.append([p, q])
            # updating the points location
            p += step_direction[0]
            q += step_direction[1]

            print("Paths", paths)
            print("p, q", [p, q])
            print("cpoints", cpoints)

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
                else:
                    second_id = c2id[_c2id([p, q])]
                # updating neighbouring matrix
                neighbour_matrix[first_id, second_id] = line.__len__()
                neighbour_matrix[second_id, first_id] = line.__len__()
                # removing line from the original work area of the skeleton
                for l in line[1:]:
                    check[l[0], l[1]] = 0
                # if there is 2 points next to each other then we are merging them
                if line.__len__() == 1:
                    area = check[p-1:p+2, q-1:q+2]
                    area2 = check[line[0][0]-1:line[0][0]+2, line[0][1]-1:line[0][1]+2]
                    # removing the one with smaller degree
                    if np.sum(area) > np.sum(area2):
                        cpoints.remove([line[0][0], line[0][1]])
                        check[line[0][0], line[0][1]] = 0
                    else:
                        cpoints.remove([p, q])
                        check[p, q] = 0

                # resetting lines
                line = []
                break
    return neighbour_matrix


def _c2id(coordinate: list):
    return f"{coordinate[0]},{coordinate[1]}"
