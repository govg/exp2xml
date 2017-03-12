from refactoredRF import RandomForest
import numpy as np

model = RandomForest(numTrees=20, maxDepth=10, splitCriterion='exp')
Xtr = np.load("Xtrain.npy")
Xte = np.load("Xtest.npy")
Ytr = np.load("Ytrain.npy")
Yte = np.load("Ytest.npy")

model.fit(Xtr, Ytr)

acc = (model.predict(Xte) == Yte).mean()

print(acc)