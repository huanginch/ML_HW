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

    # randomly generate points
    x_coors, y_coors, labels = rand_num(m, b, n_num, max_num)
    dataset = np.vstack((np.ones(n_num, dtype=int), x_coors, y_coors)).T

    # PLA
    print("Processing PLA...\n")
    w_pla = PLA(dataset, labels)
    print("PLA end\n")

    #Pocket
    print("Processing Pocket...\n")
    times = 50 # iteration times
    w_pocket, errorPt = Pocket(dataset, labels, times)
    print("Pocket error rate: %f\n" %errorPt)

    #PLA curve
    x = np.arange(max_num + 1) 
    a,b = -w_pla[1]/w_pla[2], -w_pla[0]/w_pla[2]
    plt.plot(x, a*x + b, 'g-')

    #Pocket curve
    x = np.arange(max_num + 1) 
    c,d = -w_pocket[1]/w_pocket[2], -w_pocket[0]/w_pocket[2]
    plt.plot(x, c*x + d, 'k-')

    # plot random points -> blue: positive, red: negative
    plt.plot(x_coors[:pos_num], y_coors[:pos_num], 'o', color='blue')   # positive
    plt.plot(x_coors[pos_num:], y_coors[pos_num:], 'o', color='red')    # negative
    plt.show()