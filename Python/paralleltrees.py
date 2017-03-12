import numpy 
from collections import deque
import pp
import random
import copy
#import mf
class Node:
	def __init__(self):
		self.leafNode = True
		self.x=-1;
		self.leftidx = -1
		self.rightidx = -1
		self.leftchild = -1
		self.rightchild = -1
		self.level = -1
		self.t=0;
		self.leafDist= numpy.array([])

	# @profile
	def fit(self,X,Y,sc,classwts,classes,maxdepth):
		"""" Function for training the node given the relevant data. 

		:param X: Training data at the node. 
		:param Y: Training class labels at the node
		:param model: The splitNode model containing feature for comparison, threshold, other weaklearner params
		:param sc: infogain or gini
		:returns: The model of the weaklearner 
		"""    
		if (sc=='infogain'):
			splitCriterion = mf.informationGain
		if (sc=='gini'):
			splitCriterion = mf.giniGain
		# print "Training\n"
		u = classes
		D = X.shape[1]
		highestInfoGain =-100  # initialization
		# node = Node()
		candidates = {'var':0,'thresh':0}
		# for each variable, how many random threshold checks have to be made. 
		numChecks= 10
		# number of variables to be evaluated at each node. set to sqrt(totalDim of features)
		numVars = numpy.round(numpy.sqrt(D)).astype(numpy.int16) 
		isLeafNode = self.trivialNodeCheck(Y) or self.level>=maxdepth
		if isLeafNode:
			self.leafNode = True
			hc = numpy.histogram(Y,numpy.append(u,numpy.Inf))[0]
			hc = (hc.astype(numpy.float)+1.0)/classwts
			self.leafDist = hc/hc.sum()
			# print "Leaf\n"
			return self
			
		for _ in xrange(numVars):
			var = numpy.random.random_integers(D)-1  ### -1 because it generates between 1 and D and we need between 0 and D-1
			tempX = X[:,var]
			tmin = numpy.min(tempX)
			tmax = numpy.max(tempX)
			for _ in xrange(numChecks):
				t = numpy.random.random()*(numpy.float(tmax)-tmin)+tmin
				dec = (tempX<t)
				InfoGain = splitCriterion(Y,dec,u)
				# InfoGain = random.randint(1,10)
				if (InfoGain>highestInfoGain):
					highestInfoGain = InfoGain
					candidates['var'] = var
					candidates['thresh'] = t
					

		self.x = candidates['var']
		self.t = candidates['thresh']
		# self.leftidx = nodein.leftidx
		# self.rightidx = nodein.rightidx
		# self.level = nodein.level
		self.leafNode = False
		return self

	# @profile
	def trivialNodeCheck(self,y):
		"""This function checks if a given node is at a trivial state and needs no splitting. This facilitates the passage of histograms down 
		to the leave when a trivial node occurs. There are several levels of checking. Number of datapoints at the node, distribution of classes. 
		
		:param numel: number of vectors reaching a node
		:param y: class labels at those vectors
		:returns: a boolean variable determining a node is trivial or not.
		"""
		
		if (y.shape[0]<2):
			return True
			
		yhist = numpy.bincount(y.astype(numpy.int32)).astype(numpy.float)
		yhist /= yhist.sum()
		
		if (yhist.max()>0.999):
			return True
		   
		return False


			
	# @profile
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
	def __init__(self,depth):
		self.classes = []
		self.Nodes = []
		self.depth = depth


	
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
		return self




	# @profile
	def predict(self,X):
		""" This function is for testing a tree of the forest 

		
		:param model: Tree model generated by the trainer code
		:param X: Data with which the tree is to be tested
		:returns: probabilities of each of the classes
		"""
		queue = deque([{'leftidx':0,'rightidx':X.shape[0],'nodeidx':0}])
		dataix = numpy.arange(X.shape[0])
		ysoft = numpy.ndarray((X.shape[0],self.classes.shape[0]))
		while(queue):
			currentelem = queue.popleft()
			leftidx = currentelem['leftidx']
			rightidx = currentelem['rightidx']
			nodeidx = currentelem['nodeidx']
			reld = dataix[leftidx:rightidx] 
			if self.Nodes[nodeidx].leafNode:
				ff = dataix[leftidx:rightidx]
				ld = self.Nodes[nodeidx].leafDist
				ysoft[ff,:] =  numpy.tile(ld, (ff.size,1))
			else:
				yhat = self.Nodes[nodeidx].predict(X[reld])
				yhat = yhat.reshape((yhat.size,))
				i1 = yhat.nonzero()[0]
				i2 = (yhat==0).nonzero()[0]
				dataix[leftidx:rightidx] = numpy.concatenate((reld[i1],reld[i2])) 
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
	
	def __init__(self,numTrees=20,maxDepth=100,splitCriterion='infogain',weighting=False):
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
		self.treeModels = [tree(maxDepth) for _ in xrange(numTrees)]

	# class forest:
	# 	"""forest class: Use this by just calling forest(numoftrees,maximumdepthofeachtree)

	# 	:param numTrees: number of trees required in the forest
	# 	:param maxDepth: The maximum depth to which each of the trees in the forest will be trained 
	# 	:returns: Initialized forest data structure.
	# 	""" 
	# 	def __init__(self,numTrees,maxDepth):
	# 		self.treeModels = [tree(maxDepth) for _ in xrange(numTrees)]
			
	def fit(self,X,Y):
		"""To train with the data provided as ndarrays
	
		:param X: Training features
		:param Y: Labels 
		:returns: Trained model. 
		"""
		# for i in xrange(len(self.treeModels)):
		# 	self.treeModels[i].fit(X,Y, self.splitCriterion, self.weighting)
		

		server = pp.Server()
		jobqueue = deque()

		for i in xrange(len(self.treeModels)):
			#jobqueue.append(server.submit(self.treeModels[i].fit, (X[reld],Y[reld],self.splitCriterion,self.weighting), modules = ('paralleltrees','copy','numpy','random')))
			jobqueue.append(server.submit(self.treeModels[i].fit, (X,Y,self.splitCriterion,self.weighting), modules = ('paralleltrees','copy','numpy','random')))

		print "This should be parallel now, check CPU usage in task manager, should be >25%"
		for i in xrange(len(self.treeModels)):
			self.treeModels[i] = copy.deepcopy(jobqueue[0]())
			jobqueue.popleft()

		return
	
	# # @profile
	# def train(X,Y,numTrees=20,depth=5,splitCriterion='infogain',weighting=True): ## data configuration done in config.py and featureExtractorModule.py 
	# 	"""Trains a random forest model with 'numTrees' trees and each tree of depth='depth'
		
	# 	:params numTrees: Number of trees in the model [default=20]
	# 	:params maxDepth: The maximum depth to which each tree will be trained. [default=5]
	# 	:params splitCriterion: The criterion for splitting - 'gini' or 'infogain'
	# 	:params classweights: To counteract the imbalanced training data. Boolean variable
	# 	"""
	# 	f_instance = forest(numTrees,depth)
	# 	for i in xrange(len(f_instance.treeModels)):
	# 		f_instance.treeModels[i] = treeTrain(X,Y,f_instance.treeModels[i],splitCriterion,weighting)
	# 	return f_instance

	def predict(self,X):
		"""To test the forest model with the data provided as ndarray
	
		:param X: Training features
		:returns: Predicted class labels. 
		"""
		if len(X.shape)==1:
			X= X.reshape((1,X.size))

		cl = self.treeModels[0].classes
		numCl = cl.size
		overProb = numpy.zeros((X.shape[0],numCl))
		for i in xrange(self.numTrees):
			probs = self.treeModels[i].predict(X)
			overProb+=probs
			
		winningClass = numpy.argmax(overProb, axis=1)
		return cl[winningClass]

	#	return self.test(self.forestModel,X)    



	# # @profile
	# def test(forest,X):
	# 	"""Tests a random forest model with model generated by ontheflyTrainer.py
		
	# 	:param forest: forest object generated by the trainer module
	# 	:param X: Data with which the forest is to be tested
	# 	:returns: Predicted class labels of the test data
	# 	"""
	# 	numTrees = len(forest.treeModels)
	# 	cl = forest.treeModels[0].classes
	# 	numCl = cl.size
	# 	overProb = numpy.zeros((X.shape[0],numCl))
	# 	for i in xrange(numTrees):
	# 		probs = treeTest(forest.treeModels[i],X)
	# 		overProb+=probs
			
	# 	winningClass = numpy.argmax(overProb, axis=1)
	# 	return cl[winningClass]

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
