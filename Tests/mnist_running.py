import numpy as np
from rf_dimucb import RandomForest
import mnist_input as test

train_data, train_labels, test_data, test_labels, validation_data, validation_labels = test.get_data()

shape = train_data.shape
train_data = np.reshape(train_data, (shape[0], shape[1]*shape[2]))
shape = test_data.shape
test_data = np.reshape(test_data, (shape[0], shape[1]*shape[2]))
Xtr = []
Ytr = []
Xte = []
Yte = []
for i, data in enumerate(train_data):
    if train_labels[i][4] == 1:
        Xtr.append(data)
        Ytr.append(0)
    elif train_labels[i][7] == 1:
        Xtr.append(data)
        Ytr.append(1)

for i, data in enumerate(test_data):
    if test_labels[i][4] == 1:
        Xte.append(data)
        Yte.append(0)
    elif test_labels[i][7] == 1:
        Xte.append(data)
        Yte.append(1)

Xtr = np.array(Xtr)
Ytr = np.array(Ytr)
Xte = np.array(Xte)
Yte = np.array(Yte)

print Xtr.shape, Ytr.shape, Xte.shape, Yte.shape


model = RandomForest(numTrees=20, maxDepth=8, splitCriterion='exp')
model.fit(Xtr, Ytr)
acc = (model.predict(Xte) == Yte).mean()

print(acc)
