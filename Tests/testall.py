import os, sys, datetime, getpass
sys.path.append("../Python")

#   Import the required RF libs. In the future, conver these
#   to single lib, with options. Maybe inherit?
import rf_dimucb_level.RandomForest as RF_UCB_L
import rf_dimucb.RandomForest as RF_UCB
import rf_ts_level.RandomForest as RF_TS
import refactoredRF.RandomForest as RF_BASE

#   Assume that all the folders inside the Data folder have 
#   requisite .npy training and testing files
datasets = os.listdir("../Data/")
clfs = []
logfile = "../Tests/testlogs.txt"

curtime = str(datetime.datetime.now())
log = open(logfile, 'a')
log.write("\nTests started: " +  curtime)
log.write("\nStarted by: " + getpass.getuser())

#   It is this ugly so that parameters can be assigned in a "neat" manner
#   and classifiers added or removed without interfering with others
clfs.append(RF_UCB_L())
clfs.append(RF_UCB())
clfs.append(RF_TS())
clfs.append(RF_BASE())

#   Loop over all the datasets, and then over all the classifiers
for curdataset in datasets:

    curpath = "../Data/" + curdataset + "/"

    Xtr = np.load(curpath + "Xtrain.npy")
    Ytr = np.load(curpath + "Ytrain.npy")
    Xts = np.load(curpath + "Xtest.npy")
    Yts = np.load(curpath + "Ytest.npy")

    #   Load all data
    for clf in clfs:
        
        starttime = time.time()
        clf.fit(Xtr, Ytr)
        tottime = time.time() - starttime

        acc = (Yts == clf.predict(Xts)).mean()

        #   Print to stdout
        print("Current classifier is : ", clf.clfname())
        print("Current params are : ", clf.clfparams())
        print("Accuracy is: ", acc)
        print("Time taken: ", tottime)

        #   Write to logfile
        log.write("\n" + clf.clfname())
        log.write("\n" + clf.clfparams())
        log.write("\n" + acc)

log.close()
