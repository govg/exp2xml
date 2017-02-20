/*
Author	:	ic070897
Title	:	Forest data type definition

This is the header file behind the forest class. 
*/
#ifndef FOREST_H
#define FOREST_H

#include "classifiertree.hpp"
#include "typedef.hpp"
#include "IClassifier.hpp"

template<typename T>
class ClassifierForest:public IClassifier<double>{
};

template<>
class ClassifierForest<double>:public IClassifier<double>{
public:
	ClassifierForest();
	ClassifierForest(int nbtrees, int maxdepth, bool bagging, double proportion);
	bool fit(const MatrixReal&, const VectorInteger&);
	int predict(const VectorReal&);
	void forestHists();
	void storeForest(std::fstream& storage);
	void readForest(std::fstream& storage);
	bool readAndTrain(std::fstream& trainData);
protected:
	std::vector<ClassifierTree> trees;
	int numTrees;
	bool setBagging;
	double bagProportion;
};

#endif /* FOREST_H */
