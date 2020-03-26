from preprocessors.feature_extraction.skeleton import skeleton, corner_points
from utils.images import load_images
import cv2 as cv

kanjis = load_images(minimum_count=5, random_seed=0, category_limit=1)
x_train, y_train, _, _ = kanjis.train_test_split(None)

img = x_train[0]

skel = skeleton(img, thresh_low=56)
N = corner_points(skel)

skel[skel > 0] = 255
N[N > 0] = 255

cv.imshow("original", img)
cv.imshow("N", cv.resize(N, (N.shape[0]*4, N.shape[1]*4), interpolation=cv.INTER_AREA))
cv.imshow("skeleton", cv.resize(skel, (skel.shape[0]*8, skel.shape[1]*8), interpolation=cv.INTER_AREA))
cv.waitKey(0)
cv.destroyAllWindows()
