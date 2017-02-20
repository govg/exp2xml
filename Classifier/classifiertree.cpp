/*
Author	:	ic070897
Title	:	Core functions to the ClassifierTree class

This contains the definitions of all the class variables and the functions for
the class of the Tree datatype.
*/

#include "classifiertree.hpp"
#include "typedef.hpp"
#include <queue>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>

#define DATA_MIN 40



//	The definitions of the functions used in the ClassifierTree class.
ClassifierTree::ClassifierTree(int maxdepth)
{
	Node n;
	maxDepth = maxdepth;
	nodes.push_back(n);	
}
void ClassifierTree::createBag(const MatrixReal& totalMat, const VectorInteger& totalLabels, MatrixReal& featMat, VectorInteger& labels, double bagSize)
{
	std::vector<int> bag;
	std::map<int,int> tags;
	int numClasses(0);


	for(int i = 0; i < totalLabels.size(); ++i)
		tags[totalLabels[i]]++;

	int curClass(-1);


	for(int i = 0; i < numClasses; i++)
	{

		while(!tags[++curClass]);

		for(int j = 0; j < totalLabels.size(); ++j)
		{
			if(totalLabels[j] == curClass)
				bag.push_back(j);
		}
		
		std::random_shuffle(bag.begin(),bag.end());

		for(int j = 0; j < floor(bag.size() * bagSize); j++)
		{
			featMat<<totalMat.row(bag[j]);
			labels<<totalLabels(bag[j]);
		}

		bag.clear();
	}
}
/*
Function 	:	trainTree
Args		:	MatrixReal featMat, the feature matrix (of dimensions n x d), VectorInteger labels, the label set (of dimension)
Returns 	:	nothing

This is the entire training function for the class. It accesses the protected variables in the class, and trains
the nodes according to the data provided. It calls trainNode on each of the nodes, and checks for triviality using isTrivial
This also accounts for dominance of any one class in the given training set.

*/
void ClassifierTree::trainTree(const MatrixReal& totalMat, const VectorInteger& totalLabels,int numClasses,VectorReal classWts, bool bagging = false, double bagSize = 1.00 )
{

//	This part is to facilitate the bagging:

	MatrixReal featMat;
	VectorInteger labels;

	if(bagging)
		createBag(totalMat,totalLabels,featMat,labels,bagSize);
	else
	{
		featMat = totalMat;
		labels = totalLabels;

	}	
//	We work with a queue of nodes, initially containing only the root node.
//	We process the queue until it becomes empty.
	int size = labels.rows();
	int dims = featMat.cols();
	std::queue<int> toTrain;
	std::vector<bool> indices(size);
	std::vector<int> nodeix;
	for(int i = 0; i < size; ++i)
		nodeix.push_back(i);

//	std::cout<<"Training tree, dimensions set\n";
	int cur;

//	The relevant indices for the root node is the entire set of training data
	nodes[0].start = 0;
	nodes[0].end = size-1;
	nodes[0].depth = 0;
	
	toTrain.push(0);
//	While the queue isn't empty, continue processing.
	while(!toTrain.empty())
	{

		cur = toTrain.front();
		toTrain.pop();
		if(isTrivial(cur,numClasses,labels,nodeix))
		{
			nodes[cur].isLeaf = true;
			nodes[cur].hist.resize(numClasses);
			nodes[cur].hist = VectorReal::Zero(numClasses);
			for(int i = nodes[cur].start; i <= nodes[cur].end; ++i)
			{
				nodes[cur].hist[labels(nodeix[i])] += 1.0;
			}
		}	
		else
		{
			trainNode(cur, featMat, labels,indices,nodeix);
			VectorReal relFeat;
			relFeat = featMat.col(nodes[cur].x);

//			We just set the indices depending on whether the features are greater or lesser.
//			Conventions followed : greater goes to the right.
			for(int k = nodes[cur].start; k <= nodes[cur].end; ++k)
				indices[k] = relFeat(nodeix[k]) < nodes[cur].threshold;
			int part = partition(nodes[cur].start,nodes[cur].end,indices,nodeix);

			Node right, left;
//			Increase the depth of the children
			right.depth = left.depth = nodes[cur].depth + 1;

//			Correctly assign the partitions
			left.start = nodes[cur].start;
			left.end = part -1;
	
//			Push back into the relevant places and also link the parent and the child
			nodes.push_back(left);
			nodes[cur].leftChild = nodes.size()-1;
			toTrain.push(nodes[cur].leftChild);

//			Ditto with the right node. 
			right.start = part;
			right.end = nodes[cur].end;
			nodes.push_back(right);
			nodes[cur].rightChild = nodes.size()-1;
			toTrain.push(nodes[cur].rightChild);
		}

	}
}
/*
Function 	:	testTree
Args		:	VectorReal feat (the feature vector) of the particular datapoint we wish to classify. 
Returns 	:	VectorReal, Returns the normalized probability distribution of the classes into which the given data point
				can fall into. 
*/
VectorReal ClassifierTree::testTree(const VectorReal& feat)
{
	int cur;
	cur = 0;

	while(!nodes[cur].isLeaf)
	{
//		std::cout<<"Right now at node no : "<<cur<<"\n";
//		std::cout<<"Node variables : x,t "<<nodes[cur].x<<"\t"<<nodes[cur].threshold<<"\n";
//		std::cout<<"The feature value is "<<feat(nodes[cur].x)<<"\n";

		if(nodes[cur].threshold < feat(nodes[cur].x))
			cur = nodes[cur].rightChild;
		else
			cur = nodes[cur].leftChild;	
	}
//	std::cout<<"Right now at node no : "<<cur<<"\n";
//	for(int i = 0; i < nodes[cur].hist.size(); ++i)
//		std::cout<<nodes[cur].hist(i)<<"\t";

	return nodes[cur].hist/nodes[cur].hist.sum();
}
/*
Function 	:	trainNode 
Args		:	int cur (the id of the node to be trained), MatrixReal features, VectorInteger labels (the total feature vector and labels)
Returns 	:	nothing

This trains each node given by id, and pushes it's children on to the queue. This function selects k different features,
then generates d different thresholds for each of the features, and finds the partitioning which maximises information gain.
*/
void ClassifierTree::trainNode(int cur, const MatrixReal& features, const VectorInteger& labels,std::vector<bool>& indices,const std::vector<int>& nodeix)
{
	double infoGain(100.00);

	int numVars, numChecks,featNum,dims;

	dims = features.cols();
	VectorReal relFeat;
	relFeat.resize(features.rows());

	dims = features.cols();

	numVars = (int)((double)sqrt((double)dims));

	numChecks = 4;

	for(int i = 0; i <= numVars; ++i)
	{
		featNum = rand()%dims;
		relFeat = features.col(featNum);

		double tmax, tmin, curInfo,threshold;

		tmax = relFeat.maxCoeff();
		tmin = relFeat.minCoeff();
	
		for(int j = 0; j < numChecks; ++j)
		{
			threshold  = (rand()%100)/100.0 * (tmax-tmin) + tmin;
			
			for(int k = nodes[cur].start; k <= nodes[cur].end; ++k)
				indices[k] = (relFeat(nodeix[k]) < threshold);

			curInfo = informationGain(nodes[cur].start, nodes[cur].end, labels,indices,nodeix);

			if(curInfo < infoGain)
			{
				infoGain = curInfo;
				nodes[cur].x = featNum;
				nodes[cur].threshold = threshold;
			}
		}
	}

	return;
}
/*
Function 	:	treeHist
Args		:	nothing
Returns 	:	nothing

This is just a debugging function. It prints the histogram of the tree it is called on, node wise. In case of split nodes
it will simply print the features number and the threshold of the split.
*/
void ClassifierTree::treeHist()
{
	for(unsigned int i = 0; i < nodes.size(); ++i)
	{
		

		if(nodes[i].isLeaf)
		{
			std::cout<<"\nNode number "<<i<<"\n";			
			for(signed int j = 0; j < nodes[i].hist.size(); ++j)
				std::cout<<nodes[i].hist(j)<<"\t";
		}
		else
		{
			std::cout<<"\nNode number "<<i<<"\n";
			std::cout<<"X : "<<nodes[i].x<<" thres : "<<nodes[i].threshold<<"\n";
		}

		std::cout<<"\nStart : "<<nodes[i].start<<" End : "<<nodes[i].end<<"\n";
	}
}
/*
Function 	:	informationGain
Args		:	int i,j (start and end indices), VectorInteger labels(labels set of training data)
Returns 	:	double, The information gained in splitting the nodes contained in [i,j]
				We decide which one to send left and which to send right based on 
				the array indices[], which is a member variable of the ClassifierTree class
*/

double ClassifierTree::entr(const VectorInteger& y)
{

	VectorReal histogram;
	VectorInteger histInt;

//	histogram.resize((int)y.maxCoeff()+1);
//	histogram = VectorReal::Zero((int)y.maxCoeff()+1);

//	Hardcoded value on the number of classes here, change if num(classes)>10
	histogram.resize(10);

//	This line was removed because it was creating std::bad_alloc exceptions.
//	This is probably because of the multiple resizing of the array, which could
//	force the execution to run out of available memory
//	histogram.resize(y.maxCoeff()+1);

	histogram = VectorReal::Zero(histogram.size());
	histInt = VectorInteger::Zero(histogram.size());

	for(int i = 0; i < y.size(); ++i)
	{
		histInt(y(i)) += 1;
	}

	for(int i = 0; i < histogram.size(); ++i)
		histInt(i) += 1;

	double sum(0.0);

	sum = histInt.sum();

	double temp;

	for(int i = 0; i < histogram.size(); ++i)
	{
//		std::cout<<"REDUC\n";

		temp = log((double)histInt(i)/sum)*((double)histInt(i)/sum);
		histogram(i) = temp;
//		histogram(i) = temp/ (double)log(2.0);
	}

	return -histogram.sum();

}

double ClassifierTree::informationGain(int i, int j, const VectorInteger& labels,const std::vector<bool>& indices,const std::vector<int>& nodeix)
{
	
	VectorInteger l,r,t;
	int llabel, rlabel;
	int ctr(0);

	for(int k = i; k <= j; ++k)
		ctr += ((indices[k])?1:0);

	l.resize(ctr);
	r.resize(j+1-i-ctr);
	t.resize(j-i+1);

	l = VectorInteger::Zero(ctr);
	r = VectorInteger::Zero(j+1-i-ctr);
	t = VectorInteger::Zero(j-i+1);

	for(int k = i, m=0, n=0; k <= j; ++k)
	{
		if(indices[k])
			l[m++] = labels[nodeix[k]];
		else
			r[n++] = labels[nodeix[k]];
		
		t[k-i] = labels[nodeix[k]];
	}

	llabel = (l.sum()/l.size() > 0.5 ? 1 : 0);
	rlabel = (r.sum()/r.size() > 0.5 ? 1 : 0);

	double h,hl,hr;
	if(llabel)
		hl = (1-l.sum()/l.size()); 
	else
		hl = l.sum()/l.size();

	if(rlabel)
		hr = (1-r.sum()/r.size()); 
	else
		hr = r.sum()/r.size();
/*	

	h = entr(t);
	hl = entr(l);
	hr = entr(r);

	return h - (((l.size()*hl)+(r.size()*hr))/t.size());
*/
	return hr + hl;

	
}
/*
Function 	:	partition 
Args		:	int l,r (the left and the right indices which need to be partitioned)
Returns 	:	int The position of the first element of the second set of the partition

Note that the partition is made using the indices array, which stores a true or false 
denoting whether it stays on the left or goes to the right.
*/
int ClassifierTree::partition(int l, int r,std::vector<bool>& indices,std::vector<int>& nodeix)
{

	int temp;
	bool temp2;

	while ( l <= r)
	{
		if(!indices[l])
		{
			temp = nodeix[l];
			nodeix[l] = nodeix[r];
			nodeix[r] = temp;

			temp2 = indices[l];
			indices[l] = indices[r];
			indices[r] = temp2;

			r--;
		}
		else
			l++;
	}
	return r+1;
}
/*
Function 	:	isTrivial
Args		:	int i(the current node id), VectorInteger labels(the set of all the labels)
Returns 	:	bool, a boolean variable whether it's is a leaf or not
*/
bool ClassifierTree::isTrivial(int i,int nbclasses, const VectorInteger& labels,const std::vector<int>& nodeix)
{
//This part checks for triviality of the node
//The three ways a node can be trivial are  
//a) it has too little elements,
//b) it has the same class elements as before
//c) it has hit the max depth
	VectorReal hist = VectorReal::Zero(nbclasses);

  	for(int k = nodes[i].start; k <= nodes[i].end; ++k)
    	hist(labels(nodeix[k])) += 1.0;
	
	hist = hist/hist.sum();
  	if(hist.maxCoeff() > (.9))
    {
		nodes[i].isLeaf = true;
	}

  	if(nodes[i].end - nodes[i].start < DATA_MIN)
  	{
  		nodes[i].isLeaf = true;
  		return nodes[i].isLeaf;
  	}  

	if(nodes[i].depth == maxDepth)
  	{
  		nodes[i].isLeaf = true;
  		return nodes[i].isLeaf;
  	}   	
	return nodes[i].isLeaf;
}

void ClassifierTree::readTree(std::fstream& storage, int numClasses)
{
	int size, internals,i;
//	storage>>size>>internals>>maxDepth;

	storage.read((char*)&size, sizeof(size));
	storage.read((char*)&internals, sizeof(internals));
	storage.read((char*)&maxDepth, sizeof(maxDepth));
	
	std::cout<<"testing:"<<size<<" "<<internals<<" "<<maxDepth<<std::endl;
	nodes.resize(size);

//	std::cout<<"\n"<<size<<"\t"<<internals<<"\t"<<maxDepth<<"\t";
	
	for(int k = 0; k < internals; k++)
	{
		storage.read((char*)&i, sizeof(i));
//		std::cout<<i<<"\t";
		storage.read((char*)&nodes[i].depth, sizeof(nodes[i].depth));
		storage.read((char*)&nodes[i].x, sizeof(nodes[i].x));
		storage.read((char*)&nodes[i].threshold, sizeof(nodes[i].threshold));
		storage.read((char*)&nodes[i].start, sizeof(nodes[i].start));
		storage.read((char*)&nodes[i].end, sizeof(nodes[i].end));
		storage.read((char*)&nodes[i].leftChild, sizeof(nodes[i].leftChild));
		storage.read((char*)&nodes[i].rightChild, sizeof(nodes[i].rightChild));
		storage.read((char*)&nodes[i].isLeaf, sizeof(nodes[i].isLeaf));
//		std::cout<<k<<"kek\t";

	}
//	std::cout<<"\nNow for leaves\n";
	for(int k = 0; k < size - internals; k++)
	{

		storage.read((char*)&i, sizeof(i));
	//	std::cout<<i<<"\t";
		storage.read((char*)&nodes[i].depth, sizeof(nodes[i].depth));
		storage.read((char*)&nodes[i].x, sizeof(nodes[i].x));
		storage.read((char*)&nodes[i].threshold, sizeof(nodes[i].threshold));
		storage.read((char*)&nodes[i].start, sizeof(nodes[i].start));
		storage.read((char*)&nodes[i].end, sizeof(nodes[i].end));
		storage.read((char*)&nodes[i].leftChild, sizeof(nodes[i].leftChild));
		storage.read((char*)&nodes[i].rightChild, sizeof(nodes[i].rightChild));
		storage.read((char*)&nodes[i].isLeaf, sizeof(nodes[i].isLeaf));
		
		nodes[i].hist.resize(numClasses);
		
//		std::cout<<k<<"\t";
		for(int j = 0; j < nodes[i].hist.size(); j++)
			storage.read((char*)&nodes[i].hist[j], sizeof(nodes[i].hist[j]));
	}

	
}

void ClassifierTree::storeTree(std::fstream& storage)
{
	int internals(0);

	for(int i = 0; i < nodes.size(); i++)
		internals+= ((nodes[i].isLeaf)?0:1);
	int size;
	size = nodes.size();
	storage.write((char*)&size, sizeof(size));
	storage.write((char*)&internals, sizeof(internals));
	storage.write((char*)&maxDepth, sizeof(maxDepth));
	for(int i = 0; i < nodes.size(); i++)
	{
		if(!nodes[i].isLeaf)
		{

			storage.write((char*)&i, sizeof(i));
			storage.write((char*)&nodes[i].depth, sizeof(nodes[i].depth));
			storage.write((char*)&nodes[i].x, sizeof(nodes[i].x));
			storage.write((char*)&nodes[i].threshold, sizeof(nodes[i].threshold));
			storage.write((char*)&nodes[i].start, sizeof(nodes[i].start));
			storage.write((char*)&nodes[i].end, sizeof(nodes[i].end));
			storage.write((char*)&nodes[i].leftChild, sizeof(nodes[i].leftChild));
			storage.write((char*)&nodes[i].rightChild, sizeof(nodes[i].rightChild));
			storage.write((char*)&nodes[i].isLeaf, sizeof(nodes[i].isLeaf));

		}
	}

	for(int i = 0; i < nodes.size(); i++)
	{
		if(nodes[i].isLeaf)
		{

			storage.write((char*)&i, sizeof(i));
			storage.write((char*)&nodes[i].depth, sizeof(nodes[i].depth));
			storage.write((char*)&nodes[i].x, sizeof(nodes[i].x));
			storage.write((char*)&nodes[i].threshold, sizeof(nodes[i].threshold));
			storage.write((char*)&nodes[i].start, sizeof(nodes[i].start));
			storage.write((char*)&nodes[i].end, sizeof(nodes[i].end));
			storage.write((char*)&nodes[i].leftChild, sizeof(nodes[i].leftChild));
			storage.write((char*)&nodes[i].rightChild, sizeof(nodes[i].rightChild));
			storage.write((char*)&nodes[i].isLeaf, sizeof(nodes[i].isLeaf));

			for(int j = 0; j < nodes[i].hist.size(); j++)
				storage.write((char*)&nodes[i].hist[j], sizeof(nodes[i].hist[j]));

		}
	}
}
