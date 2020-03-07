import numpy as np

hellinger = lambda x, y: np.sqrt(np.sum(np.power(np.sqrt(x)-np.sqrt(y), 2)))/np.sqrt(2)
