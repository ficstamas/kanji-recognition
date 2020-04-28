from itertools import islice
import numpy as np
import glob


def data_generator(files, batch_size, n_classes):
    while 1:
        lines = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                _ = f.readline()  # ignore the header
                while True:
                    temp = len(lines)
                    lines += list(islice(f, batch_size-temp))
                    if len(lines) != batch_size:
                        break
                    idxs = []
                    vals = []
                    ##
                    y_idxs = []
                    y_vals = []
                    y_batch = np.zeros([batch_size, n_classes], dtype=float)
                    count = 0
                    for line in lines:
                        itms = line.strip().split(' ')
                        ##
                        y_idxs = [int(itm) for itm in itms[0].split(',')]
                        for i in range(len(y_idxs)):
                            y_batch[count, y_idxs[i]] = 1.0/len(y_idxs)
                        idxs += [(count, int(itm.split(':')[0])) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield idxs, vals, y_batch


if __name__ == '__main__':

    train_files = glob.glob("data/Amazon/amazon_train.txt")

    training_data_generator = data_generator(train_files, 128, 670091)
    idxs_batch, vals_batch, labels_batch = next(training_data_generator)
    print("")
