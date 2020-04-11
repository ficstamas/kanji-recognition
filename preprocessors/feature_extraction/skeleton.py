import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage.interpolation import rotate



def skeleton(img: np.ndarray, thresh_low=127, thresh_high=255, thresh_type=cv.THRESH_BINARY) -> np.ndarray:
    ret, thr = cv.threshold(img, thresh_low, thresh_high, thresh_type)
    thr[thr > 0] = 1

    skel = skeletonize(thr).astype(np.uint8)

    S = np.zeros([skel.shape[0]+2, skel.shape[0]+2], dtype=np.uint8)
    S[1:-1, 1:-1] = skel

    return S



def kernel_gen(kernel):

    kernel_list = []
    '''
    for i in range(4):
        kernel_list.append(kernel)
        kernel = np.rot90(kernel)
    '''

    shifting = []
    for i in range(8):
        shifting.append(100+i)

    index = 0
    prev_index = 0
    mod = False
    for j in range(3):
        for k in range(3):
            if j == 1 and k == 1:
                continue
            if index == 3:
                index = 7
                prev_index = 3
                mod = True
            elif index == 4:
                index = 3
                prev_index = 4
                mod = True
            elif index == 5:
                index = 6
                prev_index = 5
                mod = True
            elif index == 6:
                index = 5
                prev_index = 6
                mod = True
            elif index == 7:
                index = 4
                prev_index = 7
                mod = True
            shifting.insert(index,kernel[j][k])
            shifting.remove(100+index)
            if mod:
                index = prev_index
                mod = False
            index += 1

    for i in range(8):
        kernel_ = kernel.copy()
        index = -1
        kernel_list.append(kernel_)
        for j in range(3):
            for k in range(3):
                if j == 1 and k == 1:
                    continue
                else:
                    index += 1
                    if index == 3:
                        index = 7
                        prev_index = 3
                        mod = True
                    elif index == 4:
                        index = 3
                        prev_index = 4
                        mod = True
                    elif index == 5:
                        index = 6
                        prev_index = 5
                        mod = True
                    elif index == 6:
                        index = 5
                        prev_index = 6
                        mod = True
                    elif index == 7:
                        index = 4
                        prev_index = 7
                        mod = True
                    kernel_[j][k] = shifting[index%8]
                    if mod:
                        index = prev_index
                        mod = False
        asd = shifting[0]
        for p in list(range(len(shifting))):
            shifting[p] = shifting[(p+1)%8]
        shifting[-1] = asd


    kernel1 = np.array([
        [0, 0, 0],
        [-8, 16, -8],
        [0, 0, 0]
    ])
    #kernel_list.append(kernel1)
    #kernel_list.append(np.rot90(kernel1))

    return kernel_list

def barmi(skeleton):
    S:np.ndarray
    k = 7
    l = 3
    kernel = np.array([
        [-1, -1, -1],
        [-2, 6, 2],
        [-1, -1, -1]
    ])
    kernel_cur = np.zeros((k,k))


    kernels = kernel_gen(kernel)

    S = np.zeros((skeleton.shape[0]+k-1,skeleton.shape[1]+k-1))
    conv_endpoints = np.zeros((skeleton.shape[0] + k - 1, skeleton.shape[1] + k - 1))
    conv_curve = np.zeros((skeleton.shape[0] + k - 1, skeleton.shape[1] + k - 1))
    conv_debug = np.zeros((skeleton.shape[0] + k - 1, skeleton.shape[1] + k - 1))


    _l = int((l-1)/2)
    _k = int((k-1)/2)
    w = np.zeros((k,k))

    conv_debug[_k:S.shape[0] - _k, _k:S.shape[1] - _k] = skeleton

    for p in range(k):
        for q in range(k):
            if p == _k and q == _k:
                w[p,q] = 0
                continue
            if p == _k:
                w[p,q] = 0
                continue
            if q == _k:
                w[p, q] = 0
                continue
            v = np.abs([p - _k, q - _k])
            alpha = np.arctan(v[0] / v[1])
            w[p,q] = np.abs(np.sin(2 * alpha))*(1/np.sqrt(np.power(v[0],2)+np.power(v[1],2)))



    S[_k:S.shape[0]-_k,_k:S.shape[1]-_k] = skeleton
    for i in range(_k,S.shape[0]-_k):
        for j in range(_k,S.shape[1]-_k):
            area = S[i-_l:i+_l+1,j-_l:j+_l+1]
            conv_endpoints[i,j] = np.average([np.sum(area*ker) for ker in kernels])
            area = S[i-_k:i+_k+1,j-_k:j+_k+1]
            conv_curve[i,j] += np.sum(area*w)


    conv_debug = conv_debug*conv_curve
    breakpoint()


def corner_points(skeleton, max_points):
    # fix corner points (move them on to the skeleton's line)

    S = skeleton.copy()
    '''
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
    '''
    result = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i][j] >= 1:
                result.append([i,j])

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
