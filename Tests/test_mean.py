import os, sys, datetime, getpass, time
import numpy as np
sys.path.append("../Python")

#   Import the required RF libs. In the future, conver these
#   to single lib, with options. Maybe inherit?

from rf_w_dimucbl import RandomForest as RF_W_U_L

#   Assume that all the folders inside the Data folder have 
#   requisite .npy training and testing files
datasets = os.listdir("../Data/")
clfs = []
logfile = "../Tests/testlogs_weighted.txt"

curtime = str(datetime.datetime.now())
log = open(logfile, 'a')
log.write("\nTests started: " +  curtime)
log.write("\nStarted by: " + getpass.getuser())
log.write("\nDatasets: " + str(datasets))

#   It is this ugly so that parameters can be assigned in a "neat" manner
#   and classifiers added or removed without interfering with others

clfs.append(RF_W_U_L(numTrees=50, maxDepth=20))


#   Loop over all the datasets, and then over all the classifiers
for curdataset in datasets:

    #   Load all data
    curpath = "../Data/" + curdataset + "/"

    Xtr = np.load(curpath + "Xtrain.npy")
    Ytr = np.load(curpath + "Ytrain.npy")
    Xts = np.load(curpath + "Xtest.npy")
    Yts = np.load(curpath + "Ytest.npy")

    print "Current data is: ", curdataset
    log.write("\n\nDATASET: " + curdataset)

    for clf in clfs:
        
        starttime = time.time()
        clf.fit(Xtr, Ytr)
        clf.learnWeights(Xtr, Ytr)
        tottime = time.time() - starttime

        
        acc = (Yts == clf.predict(Xts)).mean()

        #   Print to stdout
        print "Current classifier is: ", clf.clfname()
        print "Current params are: ", clf.clfparams()
        print "Accuracy is: ", acc
        print "Time taken: ", tottime

        #   Write to logfile
        log.write("\nCLF: " + clf.clfname())
        log.write("\n" + clf.clfparams()[0])
        log.write("\n" + clf.clfparams()[1])
        log.write("\nACC: " + str(acc))
        log.write("\nTIME: " + str(tottime))

curtime = str(datetime.datetime.now())
log.write("\nTests ended: " + curtime)
log.write("\n\n")
log.close()
