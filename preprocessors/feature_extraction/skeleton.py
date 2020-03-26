import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize


def skeleton(img: np.ndarray, thresh_low=127, thresh_high=255, thresh_type=cv.THRESH_BINARY) -> np.ndarray:
    ret, thr = cv.threshold(img, thresh_low, thresh_high, thresh_type)
    thr[thr > 0] = 1

    skel = skeletonize(thr).astype(np.uint8)

    return skel


def corner_points(skeleton):
    S = np.zeros([skeleton.shape[0]+2, skeleton.shape[0]+2], dtype=np.uint8)
    S[1:-1, 1:-1] = skeleton

    N = np.zeros(S.shape, dtype=np.uint8)

    # for i in range(S.shape[0]-1):
    #     for j in range(S.shape[1]-1):
    #         area = S[i:i+3, j:j+3]
    #         if area[1, 1] == 0:
    #             continue
    #
    #         sm = np.sum(area)
    #         if sm >= 4 or sm == 2:
    #             l = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    #             view = l
    #             good = True
    #             for _ in range(4):
    #                 if np.where(area + view > 1)[0].shape[0] >= 4 or np.where(area + view.T > 1)[0].shape[0] >= 4:
    #                     good = False
    #                     break
    #                 view = np.rot90(view)
    #             if not good:
    #                 continue
    #
    #             N[i+1, j+1] = sm-1

    max_points = 80
    k = 0.09
    min_distance = 3
    c = cv.goodFeaturesToTrack(S, max_points, k, min_distance)
    c = np.int0(c)

    for i in range(c.shape[0]):
        N[c[i, 0, 1], c[i, 0, 0]] = 1
    return N

