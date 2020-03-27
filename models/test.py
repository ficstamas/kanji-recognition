from preprocessors.feature_extraction.skeleton import define_graph, corner_points, skeleton
from utils.images import load_images
import cv2 as cv

kanjis = load_images(minimum_count=10, random_seed=0, category_limit=1)
x_train, y_train, _, _ = kanjis.train_test_split(None)

img = x_train[2]

# S = skeleton(img)
# C = corner_points(S,50)

asd = define_graph(img)
print(asd)

#cv.imshow("original", img)

# cv.imshow("N", cv.resize(N, (N.shape[0]*4, N.shape[1]*4), interpolation=cv.INTER_AREA))
# cv.imshow("skeleton", cv.resize(skel, (skel.shape[0]*8, skel.shape[1]*8), interpolation=cv.INTER_AREA))
cv.waitKey(0)
cv.destroyAllWindows()

