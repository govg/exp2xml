## @package Random Forest
#
#This contains all the files for the Random forest package . 
# __all__=['ontheflyTrainer','ontheflyTester']
""" 
# RandomForestBatch
"""
import numpy as np
from collections import deque


class Node:
	def __init__(self, sc, classwts, classes, maxdepth):
		self.leafNode = False
		self.x=-1;
		self.leftidx = -1
		self.rightidx = -1
		self.leftchild = -1
		self.rightchild = -1
		self.level = -1
		self.t=0;
		self.leafDist= np.array([])
		self.classwts = classwts
		self.classes = classes
		self.maxdepth = maxdepth
		if (sc == 'infogain'):
			self.splitCriterion = informationGain
		if (sc == 'gini'):
			self.splitCriterion = giniGain

	
	def fit(self,X,Y):
		"""" Function for training the node given the relevant data. 

		:param X: Training data at the node. 
		:param Y: Training class labels at the node
		:param model: The splitNode model containing feature for comparison, threshold, other weaklearner params
		:param sc: infogain or gini
		:returns: The model of the weaklearner 
		# """    
		# if (sc=='infogain'):
		# 	splitCriterion = informationGain
		# if (sc=='gini'):
		# 	splitCriterion = giniGain
		
		u = self.classes
		D = X.shape[1]
		highestInfoGain =-100  # initialization
		# node = Node()
		candidates = {'var':0,'thresh':0}
		# for each variable, how many random threshold checks have to be made. 
		numChecks= 10
		# number of variables to be evaluated at each node. set to sqrt(totalDim of features)
		numVars = np.round(np.sqrt(D)).astype(np.int16) 
		isLeafNode = self.trivialNodeCheck(Y) or self.level>=self.maxdepth
		if isLeafNode:
			self.leafNode = True
			hc = np.histogram(Y,np.append(u,np.Inf))[0]
			hc = (hc.astype(np.float)+1.0)/self.classwts
			self.leafDist = hc/hc.sum()
			# print "Leaf node\n"
			return
			
		for _ in xrange(numVars):
			var = np.random.random_integers(D)-1  ### -1 because it generates between 1 and D and we need between 0 and D-1
			tempX = X[:,var]
			tmin = np.min(tempX)
			tmax = np.max(tempX)
			for _ in xrange(numChecks):
				t = np.random.random()*(np.float(tmax)-tmin)+tmin
				dec = (tempX<t)
				InfoGain = self.splitCriterion(Y,dec,u)
				if (InfoGain>highestInfoGain):
					highestInfoGain = InfoGain
					candidates['var'] = var
					candidates['thresh'] = t
					

		self.x = candidates['var']
		self.t = candidates['thresh']
		self.leafNode = False
		return

	
	def trivialNodeCheck(self,y):
		"""This function checks if a given node is at a trivial state and needs no splitting. This facilitates the passage of histograms down 
		to the leave when a trivial node occurs. There are several levels of checking. Number of datapoints at the node, distribution of classes. 
		
		:param numel: number of vectors reaching a node
		:param y: class labels at those vectors
		:returns: a boolean variable determining a node is trivial or not.
		"""
		
		if (y.shape[0]<2):
			return True
			
		yhist = np.bincount(y.astype(np.int32)).astype(np.float)
		yhist /= yhist.sum()
		
		if (yhist.max()>0.999):
			return True
		
		return False


			
	
	def predict(self,X):
		"""" Function for testing data at the node.
	 
		:param model: The splitNode model containing feature for comparison, threshold, other weaklearner params
		:param X: Data with which the node is to be tested
		:returns: A binary array yhat (0 = data goes left, 1 = data goes right)
		"""
		yhat = (X[:,self.x] < self.t)
		return yhat    




############################################################################
############################################################################        
############################################################################
#       Tree class definitions
############################################################################
############################################################################
############################################################################

class tree:
	def __init__(self,depth, splitCriterion, Weighting):
		self.classes = []
		self.Nodes = []
		self.depth = depth
		self.splitCriterion = splitCriterion
		self.weighting = Weighting

	
	def fit(self,X,Y):
		"""This function trains a tree initialised by model to a given depth d

		:param X: Training features
		:param Y: Training labels
		:param model: the tree model which is to be trained
		:param splitCriterion: gini or infogain
		:param classWeights: boolean variable for using inverse class freq or none.
		:returns: trained tree model
		"""    
		u = np.unique(Y)
		N = X.shape[0]
		if self.weighting==False:
			classwts = np.ones_like(classweights(Y))
		else:
			classwts = classweights(Y)
		dataix = np.arange(N)
		queue = deque()
		tempnode = Node(self.splitCriterion, classwts, u, self.depth)
		tempnode.leftidx = 0
		tempnode.rightidx = dataix.size-1
		tempnode.level = 0
		self.Nodes.append(tempnode)
		queue.append(0)
		while(queue):
			currentnode = queue.popleft()
			leftidx = self.Nodes[currentnode].leftidx
			rightidx = self.Nodes[currentnode].rightidx
			currentlevel = self.Nodes[currentnode].level
			reld = dataix[leftidx:rightidx]  
			self.Nodes[currentnode].fit(X[reld],Y[reld])
			if self.Nodes[currentnode].leafNode == False:
				yhat = self.Nodes[currentnode].predict(X[reld])
				yhat = yhat.reshape((yhat.size,))
				i1 = yhat.nonzero()[0]
				i2 = (yhat==0).nonzero()[0]
				dataix[self.Nodes[currentnode].leftidx:self.Nodes[currentnode].rightidx] = np.concatenate((reld[i1],reld[i2])) 
				#append left child
				nodeleft = Node(self.splitCriterion, classwts, u, self.depth)
				nodeleft.leftidx = self.Nodes[currentnode].leftidx
				nodeleft.rightidx = self.Nodes[currentnode].leftidx+i1.shape[0]
				nodeleft.level = currentlevel+1
				self.Nodes.append(nodeleft)
				self.Nodes[currentnode].leftchild = len(self.Nodes)-1
				queue.append(len(self.Nodes)-1) 
				#append right child
				noderight = Node(self.splitCriterion, classwts, u, self.depth)
				noderight.leftidx = self.Nodes[currentnode].rightidx-i2.shape[0]
				noderight.rightidx = self.Nodes[currentnode].rightidx
				noderight.level = currentlevel+1
				self.Nodes.append(noderight)
				self.Nodes[currentnode].rightchild = len(self.Nodes)-1 
				queue.append(len(self.Nodes)-1)
				del i1, i2     
				
		self.classes = u
		return


	
	def predict(self,X):
		""" This function is for testing a tree of the forest 

		
		:param model: Tree model generated by the trainer code
		:param X: Data with which the tree is to be tested
		:returns: probabilities of each of the classes
		"""
		queue = deque([{'leftidx':0,'rightidx':X.shape[0],'nodeidx':0}])
		dataix = np.arange(X.shape[0])
		ysoft = np.ndarray((X.shape[0],self.classes.shape[0]))
		while(queue):
			currentelem = queue.popleft()
			leftidx = currentelem['leftidx']
			rightidx = currentelem['rightidx']
			nodeidx = currentelem['nodeidx']
			reld = dataix[leftidx:rightidx] 
			if self.Nodes[nodeidx].leafNode:
				ff = dataix[leftidx:rightidx]
				ld = self.Nodes[nodeidx].leafDist
				ysoft[ff,:] =  np.tile(ld, (ff.size,1))
			else:
				yhat = self.Nodes[nodeidx].predict(X[reld])
				yhat = yhat.reshape((yhat.size,))
				i1 = yhat.nonzero()[0]
				i2 = (yhat==0).nonzero()[0]
				dataix[leftidx:rightidx] = np.concatenate((reld[i1],reld[i2])) 
				queue.append({'leftidx':leftidx,'rightidx':leftidx+i1.shape[0],'nodeidx':self.Nodes[nodeidx].leftchild})
				queue.append({'leftidx':rightidx-i2.shape[0],'rightidx':rightidx,'nodeidx':self.Nodes[nodeidx].rightchild})
				del i1, i2
		
		return ysoft

############################################################################
############################################################################        
############################################################################
#       Forest class definition
############################################################################
############################################################################
############################################################################

class RandomForest:
	""" This is the class for the Random forest implementation using batch features (all features given at once). This
	doesn't require any config or feature extractor modules. It uses batchTrainer.py and batchTester.py for training and testing respectively.
	"""
	
	def __init__(self,numTrees=20,maxDepth=100,splitCriterion='infogain',weighting=True):
		"""The constructor function for the RandomForestBatch class which takes the arrays to which the RF is fit. 

		:params numTrees: Number of trees in the model
		:params maxDepth: The maximum depth to which each tree will be trained. 
		:params splitCriterion: The criterion for splitting - 'gini' or 'infogain'
		:params classweights: To counteract the imbalanced training data. Boolean variable
		"""

		self.numTrees = numTrees
		self.maxDepth = maxDepth
		self.splitCriterion = splitCriterion
		self.weighting= weighting
		self.treeModels = [tree(maxDepth, self.splitCriterion, self.weighting) for _ in xrange(numTrees)]
			
	def fit(self,X,Y):
		"""To train with the data provided as ndarrays
	
		:param X: Training features
		:param Y: Labels 
		:returns: Trained model. 
		"""
		for i in xrange(len(self.treeModels)):
			self.treeModels[i].fit(X,Y)
		return

	def predict(self,X):
		"""To test the forest model with the data provided as ndarray
	
		:param X: Training features
		:returns: Predicted class labels. 
		"""
		if len(X.shape)==1:
			X= X.reshape((1,X.size))

		cl = self.treeModels[0].classes
		numCl = cl.size
		overProb = np.zeros((X.shape[0],numCl))
		for i in xrange(self.numTrees):
			probs = self.treeModels[i].predict(X)
			overProb+=probs
			
		winningClass = np.argmax(overProb, axis=1)
		return cl[winningClass]   

        def clfname(self):
                return "RF ORIG"

        def clfparams(self):
                a = "T:" + str(self.numTrees)
                b = "D:" + str(self.maxDepth)
                
                return a,b


############################################################################
############################################################################        
############################################################################
#       Misc functions not needed as class members.
############################################################################
############################################################################
############################################################################


def informationGain(y,d,u):
	"""Function to calculate the information gain

	:param y: labels of data at the current node
	:param d: binary array of the length(y) specifying data split
	:param u: the list of classes to look for. 
	:returns: the information gain given by the current split criterion.
	"""
	yl = y[d]
	yr = y[~d]
	H = entropy(y,u)
	HL = entropy(yl,u)
	HR = entropy(yr,u)
	
	return H - ((yl.size*HL)/y.size)-((yr.size*HR)/y.size)

def entropy(y,u):
	""""Function to calculate the entropy of the split

	:param y: labels after split
	:param u: unique of all labels (is not required, might be removed from future versions)
	:returns: entropy calculated by sum(p*log(p))
	"""
	#overall entropy of y and u
	cdist = np.bincount(y.astype(np.int32)).astype(np.float)
	cdist  += 1
	cdist /=cdist.sum()
	cdist *= np.log(cdist)
	return -cdist.sum()

def giniGain(y,d,u):
	"""Function to calculate the Gini impurity based information gain

	:param y: labels of data at the current node
	:param d: binary array of the length(y) specifying data split
	:param u: the list of classes to look for. 
	:returns: the information gain given by the current split criterion.
	"""
	yl = y[d]
	yr = y[~d]
	H = gini(y,u)
	HL = gini(yl,u)
	HR = gini(yr,u)
	
	return H - ((yl.size*HL)/y.size)-((yr.size*HR)/y.size)

def gini(y,u):
	"""Function to calculate the Gini impurity

	:param y: labels of data at the current node
	:param u: the list of classes to look for. 
	:returns: the gini impurity index
	"""

	cdist = np.bincount(y.astype(np.int32))
	cdist= cdist.astype(np.float)
	cdist  += 1
	cdist /=cdist.sum()
	return 1-(cdist**2).sum()

def classweights(y_train):
	"""Gives the relative frequencies of labels
	:param y_train: Input array for which weights have to be found
	:returns: normalized histogram count
	"""
	hh = np.bincount(y_train.astype(np.int8))
	return hh.astype(np.float)/hh.sum()
