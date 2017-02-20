/*
Author	:	ic070897
Title	:	Core functions to the ClassifierForest class

This contains all the functions declared in the forest class.
*/


#include "classifierforest.hpp"
#include "typedef.hpp"

#include <iostream>
#include <fstream>

ClassifierForest<double>::ClassifierForest()
{
	
}
ClassifierForest<double>::ClassifierForest(int nbtrees, int maxdepth, bool bagging = false, double proportion = 1.0)
{
	numTrees = nbtrees;
	nbClass_ = -1;
	ClassifierTree t(maxdepth);
	while(nbtrees--)
	{
		trees.push_back(t);
	}

	setBagging = bagging;

	bagProportion  = proportion; 
}

/*
Function 	:	fit
Args		:	featMat (the feature vectors) and labels (the features classes) 
Returns 	:	nothing

This function is a generic fit function called to train the tree on the given data points. It trains the tree
present in the ClassifierForest object. 
*/
bool ClassifierForest<double>::fit(const MatrixReal& featMat, const VectorInteger& labels)
{
	uniqueLabelsStats(labels);
	for(int i = 0; i < numTrees; ++i)
	{
//		std::cout<<"Training tree no "<<i<<"\n";
		VectorReal classwts(uniqueLabelsCount_.size());
		for (int j = 0; j < uniqueLabelsCount_.size(); ++j) {
			classwts[j] = uniqueLabelsCount_[j];
		}

		classwts/=classwts.sum();
		trees[i].trainTree(featMat,labels,nbClass_,classwts,setBagging, bagProportion);
	}

	return true;
}
/*
Function 	:	forestHist
Args		:	nothing
Returns 	:	nothing

This is just a debugging function. It calls the treeHist() function on all the trees it has, which in turn 
prints the histogram of the tree it is called on, node wise. In case of split nodes
it will simply print the features number and the threshold of the split.
*/
void ClassifierForest<double>::forestHists()
{
	for(unsigned int i = 0; i < trees.size(); ++i)
	{
		std::cout<<"\nTree number "<<i+1<<"\n";
		trees[i].treeHist();
	}
}
/*
Function 	:	predict
Args		:	feat (the feature vector) which we need to fit
Returns 	:	the predicted class for the given feature vector

This will test all the trees and give an aggregate vote of which class the given feature vector most likely
corresponds to.
*/
int ClassifierForest<double>::predict(const VectorReal& feat)
{
	VectorReal result;
	int probableClass;

	result = trees[0].testTree(feat);
	for(int i = 1; i < numTrees; ++i)
	{
		result += trees[i].testTree(feat);
	}

	result.maxCoeff(&probableClass);
	return (probableClass);
}

void ClassifierForest<double>::storeForest(std::fstream& storage)
{
//	storage<<nbClass_<<"\t"<<numTrees<<"\n";
	storage.write((char*)&nbClass_, sizeof(nbClass_));
	storage.write((char*)&numTrees, sizeof(numTrees));
	for(int i = 0; i  < numTrees; ++i)
		trees[i].storeTree(storage);
}
void ClassifierForest<double>::readForest(std::fstream& storage)
{

//	storage>>nbClass_>>numTrees;

	storage.read((char*)&nbClass_, sizeof(nbClass_));
	storage.read((char*)&numTrees, sizeof(numTrees));
	

	for(int i = 0; i < numTrees; ++i)
	{
		ClassifierTree a(1);
		a.readTree(storage, nbClass_);
		trees.push_back(a);
//		std::cout<<"\nRead Tree "<<i<<"\n";
	}

}


bool ClassifierForest<double>::readAndTrain(std::fstream& dataFile)
{

	assert(!dataFile.fail());


	std::vector<std::vector<double>> filedata;
	std::vector<int> filelabels;

	double t;
	int t2,dims;
	int ctr(0);
	std::vector<double> temp;


	dataFile.read((char*)&dims, sizeof(dims));
	while(!dataFile.eof())
	{
		dataFile.read((char*) &t2, sizeof(t2));

		filelabels.push_back(t2);


		for(int i = 0; i < dims; i++)
		{
			dataFile.read((char*)&t, sizeof(t));
			temp.push_back(t);

		}

		filedata.push_back(temp);

		temp.clear();
		
	}
	
	MatrixReal trainData;
	VectorInteger trainLabel;

	trainData.resize(filedata.size(),filedata[0].size());
	trainLabel.resize(filelabels.size());

	for(unsigned int i = 0; i < filedata.size(); ++i)
	{
		for(unsigned int j = 0; j < filedata[0].size(); ++j)
			trainData(i,j) = filedata[i][j];
	}

//	std::cout<<"Attempting to store labels\n";
	
	
	for(unsigned int i = 0; i < filelabels.size(); ++i)
	{
		trainLabel(i) = filelabels[i];
		
	}
	return fit(trainData, trainLabel);

	dataFile.close();
}