import numpy as np
from numpy import matmul
from numpy import identity
from numpy import array
from numpy import dot

a = np.random.rand(4,2)
n = 4

print(a)

n = 2
p = 3
alpha = 4
X = array([[1,2,3], [4,5,6]])
Y = array([[1],[2]])
XT = np.transpose(X)

## X'*X + alpha*I
temp1 = XT@X + alpha*identity(p)

## (X'*X + alpha*I)*X'
temp2 = temp1@XT

## (X'*X + alpha*I)*X'*Y
w = temp2@Y

w = np.linalg.inv(XT@X + alpha*identity(p))@XT@Y
print(w)
print(X.shape[1])




