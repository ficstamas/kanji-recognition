from utils.images import load_images
import logging
from preprocessors.distribution import ccw_distribution
from scipy.stats import kstest
import numpy as np
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

kanjis = load_images(minimum_count=5, random_seed=0, category_limit=None)
x, y, _, _ = kanjis.train_test_split(None)

class_dists = ccw_distribution(x, y)

p_values = []

for f in range(class_dists.shape[1]):
    _, p = kstest(class_dists[:, f], 'norm')
    p_values.append(p)

p_values = np.array(p_values)
print(p_values[p_values >= (0.05/class_dists.shape[1])].shape)