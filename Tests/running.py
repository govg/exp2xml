#   This is necessary to import our modules
import sys
sys.path.append("../Python/")

from rf_w_dimucbl import RandomForest
import numpy as np

model = RandomForest(numTrees=20, maxDepth=10, splitCriterion='exp')
Xtr = np.load("../Data/mnist/Xtrain.npy")
print(Xtr.shape)
Xte = np.load("../Data/mnist/Xtest.npy")
Ytr = np.load("../Data/mnist/Ytrain.npy")
print(Ytr.shape)
Yte = np.load("../Data/mnist/Ytest.npy")

model.fit(Xtr, Ytr)
model.learnWeights(Xtr, Ytr)

acc = (model.predict(Xte) == Yte).mean()

print(acc)
