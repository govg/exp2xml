/*
 * IClassifier.h
 *
 *  Created on: Apr 9, 2014
 *      Author: Ic017914
 */

#ifndef ICLASSIFIER_H_
#define ICLASSIFIER_H_
#include <vector>
#include "Eigen/Dense"

template<typename datatype>
class IClassifier {
public:
	IClassifier(){nbClass_=0;}
	virtual bool fit(const Eigen::Matrix<datatype,Eigen::Dynamic,Eigen::Dynamic>& featMat,
			const VectorInteger& labels) = 0;
	virtual int predict(const Eigen::Matrix<datatype,Eigen::Dynamic,1>& feat) = 0;
	virtual ~IClassifier(){};
protected:
	//members
	int nbClass_;
	std::vector<int> uniqueClassLabels_;
	std::vector<int> uniqueLabelsCount_;

	//functions
	void uniqueLabelsStats(const VectorInteger& labels); //compute unique class labels and corresponding counts
};


template<typename datatype> inline void IClassifier<datatype>::uniqueLabelsStats(const VectorInteger& labels){
	bool labelexists = false;
	for (int i = 0; i < labels.size(); ++i) {
		int labeltemp = labels[i];
		labelexists = false;
		for (int j = 0; j < uniqueClassLabels_.size(); ++j) {
			if(uniqueClassLabels_[j]==labeltemp){
				labelexists = true;
				uniqueLabelsCount_[j]+=1; //increment count
				break;
			}
		}
		if(!labelexists)
		{
			uniqueClassLabels_.push_back(labeltemp); //push new label to unique class labels
			uniqueLabelsCount_.push_back(1); //count initialized with 1
		}
	}

	nbClass_ = uniqueClassLabels_.size();
}



#endif /* ICLASSIFIER_H_ */
