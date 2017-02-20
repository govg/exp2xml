/*
Author	:	ic070897
Title	:	Tree data type declration

This header file contains the declaration of the tree data type. 
*/

#ifndef ClassifierTREE_H
#define ClassifierTREE_H

#include "nodes.hpp"
#include "typedef.hpp"

#include <queue>

class ClassifierTree
{
public:
	ClassifierTree(int maxdepth);
	void trainTree(const MatrixReal& featMat, const VectorInteger& labels,int nbclasses,VectorReal classWts,bool bagging,double bagSize);
	VectorReal testTree(const VectorReal&);
	void storeTree(std::fstream& storage);
	void readTree(std::fstream& storage, int numClasses);
	void treeHist(); //for debugging

protected:
	std::vector<Node> nodes;
	int maxDepth;

	bool isTrivial(int id,int nbclasses,const VectorInteger& labels,const std::vector<int>& nodeix);
	double informationGain(int start, int end, const VectorInteger& labels,const std::vector<bool>& indices,const std::vector<int>& nodeix);
	double entr(const VectorInteger& y);
	void trainNode(int id, const MatrixReal& featMat, const VectorInteger& labels,std::vector<bool>& indices,const std::vector<int>& nodeix);
	void createBag(const MatrixReal& totalMat, const VectorInteger& totalLabel, MatrixReal& featMat, VectorInteger& labels,double bagSize);
	int partition(int start,int end,std::vector<bool>& indices,std::vector<int>& nodeix);
};

#endif /* ClassifierTREE_H */
