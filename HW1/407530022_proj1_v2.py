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

#PLA  x: (1, x, y) in dataset, y: labels, linear model: w0 + w1*x1 + w2*x2 = 0, w: [w0, w1, w2]
def PLA(dataset, labels):
    w = np.zeros(3) #init w
    count = 0
    while check_error(w, dataset, labels) is not None: #if there is mistske
        x, y = check_error(w, dataset, labels)
        w = w + y * x #correct the mistake
        count = count + 1
    print("The iteration times: %d\n" %count)
    return w

def check_error(w, dataset, labels):
    result = None
    for x, y in zip(dataset, labels):
        if int(np.sign(w.T.dot(x))) != y:
            result =  x, y   
    return result

if __name__ == '__main__':
    m, b =-2, 3 # parameter for the linear model
    n_num = 30 # generate 30 points
    max_num = 45 # max number
    pos_num = int(n_num/2) # the number of positive numbers

    # randomly generate points
    x_coors, y_coors, labels = rand_num(m, b, n_num, max_num)
    dataset = np.vstack((np.ones(n_num, dtype=int), x_coors, y_coors)).T

    # PLA
    w = PLA(dataset, labels)

    #PLA curve
    l = np.linspace(-max_num,max_num)
    a,b = -w[1]/w[2], -w[0]/w[2]
    plt.plot(l, a*l + b, 'b-')

    # plot random points -> blue: positive, red: negative
    plt.plot(x_coors[:pos_num], y_coors[:pos_num], 'o', color='blue')   # positive
    plt.plot(x_coors[pos_num:], y_coors[pos_num:], 'o', color='red')    # negative
    plt.show()