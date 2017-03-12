import numpy as np
from refactoredRF import ranking_loss, rel_ranking_loss

perm = np.random.permutation(24)
X = perm.reshape((8, 3))
print(X)
Y = np.array([0,0,1,1,0,0,1,1])
perm = np.random.permutation(8)
Y = Y[perm]
print(Y)

index = 1
threshold = 12
x_index = X[:, index]
dec = (x_index < threshold)

# print(dec)

# ranking_loss(Y, dec, index, X)
ret = rel_ranking_loss(Y, dec, index, X)
print(ret)
