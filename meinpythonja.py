import numpy as np

def someFunction(a=2):
    print("a is", a)
    print("Step 1")
    print("Step 2")
    b = np.ones((120000,120000))
    print("Step 3")
    c = 0
    print(c)
    np.exp(b)
    print("Step 4")
    return 0

print('ja')