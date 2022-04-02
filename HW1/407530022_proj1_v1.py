import numpy as np
import matplotlib.pyplot as plt

#my function to generate random number datasets
def rand_num(m, b, n_num, max_num):
    x_coors, y_coors, labels = np.array([]), np.array([]), np.array([]) # create empty 1D array
    c = 1 if m >= 0 else -1 # coef of r (right: positive, left: negative)

    # number of positive and negtive samples
    pos_num = int(n_num / 2)
    neg_num = n_num - pos_num

    # randomly generate points
    for state, n_num in [['pos', pos_num], ['neg', neg_num]]:
        x = np.random.randint(0, max_num, n_num)
        r = np.random.randint(1, max_num, n_num) # distance between point and line

        if state == 'pos':
            y = m * x + b - (r * c)
            labels = np.append(labels, np.ones(n_num, dtype=int))
        else:
            y = m * x + b + (r * c)
            labels = np.append(labels, -1*np.ones(n_num, dtype=int))

        x_coors = np.append(x_coors, x)    # save x coordinates
        y_coors = np.append(y_coors, y)    # save y coordinates

    return x_coors, y_coors, labels

#main
if __name__ == '__main__':
    m, b =-2, 3 # parameter for the linear model
    n_num = 30 # generate 30 points
    max_num = 45 # max number
    pos_num = int(n_num/2) # the number of positive numbers

    # plot function curve
    x = np.arange(max_num + 1)   # x = [0, 1,..., max_num]
    y = m * x + b
    plt.plot(x, y)

    # randomly generate points
    x_coors, y_coors, labels = rand_num(m, b, n_num, max_num)

    # plot random points -> blue: positive, red: negative
    plt.plot(x_coors[:pos_num], y_coors[:pos_num], 'o', color='blue')   # positive
    plt.plot(x_coors[pos_num:], y_coors[pos_num:], 'o', color='red')    # negative
    plt.show()