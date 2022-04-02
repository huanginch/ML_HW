import numpy as np
import matplotlib.pyplot as plt


def rand_samples(m, b, n_points, rand_param):
    x_coors, y_coors, labels = np.array([]), np.array([]), np.array([]) # create empty array
    c = 1 if m >= 0 else -1 # coef of r (right: positive, left: negative)

    # number of positive and negtive samples
    pos_num = int(n_points / 2)
    neg_num = n_points - pos_num

    # randomly generate points
    for state, n_points in [['pos', pos_num], ['neg', neg_num]]:
        x = np.random.randint(0, rand_param, n_points)
        r = np.random.randint(1, rand_param, n_points) # distance between point and line

        if state == 'pos':
            y = m * x + b - (r * c)
            labels = np.append(labels, np.ones(n_points, dtype=int))
        else:
            y = m * x + b + (r * c)
            labels = np.append(labels, -1*np.ones(n_points, dtype=int))

        x_coors = np.append(x_coors, x)    # save x coordinates
        y_coors = np.append(y_coors, y)    # save y coordinates

    return x_coors, y_coors, labels

if __name__ == '__main__':
    # y = mx + b
    m, b = 2, 1

    # other parameters
    n_points = 30
    rand_param = 30
    pos_num = int(n_points / 2)

    # plot function curve
    x = np.arange(rand_param + 1)   # x = [0, 1,..., rand_param]
    y = m * x + b
    plt.plot(x, y)

    # randomly generate points
    x_coors, y_coors, labels = rand_samples(m, b, n_points, rand_param)

    # plot random points. Blue: positive, red: negative
    plt.plot(x_coors[:pos_num], y_coors[:pos_num], 'o', color='blue')   # positive
    plt.plot(x_coors[pos_num:], y_coors[pos_num:], 'o', color='red')    # negative
    plt.show()

