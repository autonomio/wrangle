from .array_random_shuffle import array_random_shuffle


def array_to_kfold(x, y, folds=10, shuffled=True):

    if shuffled is True:
        x, y = array_random_shuffle(x, y)

    out_x = []
    out_y = []

    x_len = len(x)
    step = int(x_len / folds)

    lo = 0
    hi = step

    for i in range(folds):
        out_x.append(x[lo:hi])
        out_y.append(y[lo:hi])

        lo += step
        hi += step

    return out_x, out_y
