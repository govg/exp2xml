import refactoredRF as rflib
import numpy as np

X_train = np.load("../Data/Xtrain.npy")
X_test = np.load("../Data/Xtest.npy")
Y_train = np.load("../Data/Ytrain.npy")
Y_test = np.load("../Data/Ytest.npy")

clf = rflib.RandomForest(maxDepth = 8, numTrees = 32, splitCriterion='exp')
data = np.ndarray(100,np.float)

for i in xrange(100):
    clf.fit(X_train, Y_train)
    data[i] = (Y_test == clf.predict(X_test)).mean()

np.save("data",data)

    

