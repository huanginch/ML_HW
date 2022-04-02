import numpy as np
import matplotlib.pyplot as plt

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

def rand_num_mislabel(m, b, n_num, max_num):
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
            if n_num < 50:
                # mislabel 50 points
                labels = np.append(labels, -1*np.ones(n_num, dtype=int))
            else:
                labels = np.append(labels, np.ones(n_num, dtype=int))
        else:
            y = m * x + b + (r * c)
            if n_num < 50:
                # mislabel 50 points
                labels = np.append(labels, np.ones(n_num, dtype=int))
            else:
                labels = np.append(labels, -1*np.ones(n_num, dtype=int))

        x_coors = np.append(x_coors, x)    # save x coordinates
        y_coors = np.append(y_coors, y)    # save y coordinates

    return x_coors, y_coors, labels

#Pocket Algorithm
def Pocket(dataset, labels, times):
    error = 0.0
    w = np.zeros(3) #init w
    for i in range(0, times):
        for x, y in zip(dataset, labels):
            if int(np.sign(w.T.dot(x))) != y: 
                wt = w + y * x
                error0 = error_rate(dataset, labels, w) #error of w
                error1 = error_rate(dataset, labels, wt) # error of wt
                if error1 < error0:
                    w = wt
                    error = error1
                    break
    return w, error

def error_rate(dataset, labels, w):
    error = 0.0
    for x, y in zip(dataset, labels):
        if int(np.sign(w.T.dot(x))) != y:
            error = error +1.0
    return error/len(dataset)


if __name__ == '__main__':
    m, b =-2, 3 # parameter for the linear model
    n_num = 2000 # generate 2000 points
    max_num = 3000 # max number
    pos_num = int(n_num/2) # the number of positive numbers

    # randomly generate points with correct label
    x_coors, y_coors, labels = rand_num(m, b, n_num, max_num)
    dataset = np.vstack((np.ones(n_num, dtype=int), x_coors, y_coors)).T

    #Pocket with correct label
    print("Processing Pocket with correct label...\n")
    times = 10 # iteration times
    w_correct, errorC = Pocket(dataset, labels, times)
    print("Error rate: %f\n" %errorC)

    # randomly generate points with wrong label
    x_coors, y_coors, labels = rand_num_mislabel(m, b, n_num, max_num)
    dataset = np.vstack((np.ones(n_num, dtype=int), x_coors, y_coors)).T

    #Pocket with mislabel label
    print("Processing Pocket with wrong label...\n")
    times = 10 # iteration times
    w_mislabel, errorM = Pocket(dataset, labels, times)
    print("Error rate: %f\n" %errorM)