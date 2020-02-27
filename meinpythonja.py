import numpy as np

def someFunction(a=2):
    print("The argument is:", a)
    b = np.ones((120000,12000))
    #b = np.random.random_sample((120000,12000))
    print("Array has been initialized.")
    c = np.sum(np.exp(b))/(b.shape[0]*b.shape[1])
    print("The average of the exponential is:", c)
    return c

print('ja')