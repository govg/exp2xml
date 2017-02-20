
/*
Author	:	ic070897
Title	:	Driver Program

Change the default values in the object a itself.
*/

#include "classifierforest.hpp"
#include "typedef.hpp"


#include <iostream>
#include <fstream>
#include <ctime>

int main()
{
	

	ClassifierForest<double> a(20,10,false,1.0);

	std::fstream data,label;

	data.open("D:\\randomForestCpp\\Datasets\\gentraindata");
	label.open("D:\\randomForestCpp\\Datasets\\gentrainlabel");

	MatrixReal testdata;
	VectorInteger testlabel;

	double t;
	double s;
	std::vector<int> totallabel;
	std::vector<double> temp;
	std::vector<std::vector<double>> totaldata;

	temp.clear();
	totaldata.clear();
	totallabel.clear();

	int ctr(0);

	while(!data.eof())
	{
		for(int i = 0; i < 20; i ++)
		{
			data>>t;
			temp.push_back(t);

		}

		ctr++;

		if(ctr % 1000 == 0 )
			std::cout<<ctr/1000<<"\t";

		totaldata.push_back(temp);
		temp.clear();

	}


	data.close();

	ctr = 0;

	while(!label.eof())
	{
		label>>s;
		totallabel.push_back((int)s);

		ctr++;
			if(ctr % 1000 == 0 )
			std::cout<<ctr/1000<<"\t";
	}
	label.close();
	std::cout<<"Size of feature matrix is "<<totaldata.size()<<"\t"<<totaldata[0].size()<<"\n";
	testdata.resize(totaldata.size(),totaldata[0].size());
	std::cout<<"Size of label vector is "<<totallabel.size()<<"\n";
	testlabel.resize(totallabel.size());

	for(int i = 0; i < totaldata.size(); i++)
	{
		for(int j = 0; j < totaldata[0].size(); j++)
			testdata(i,j) = totaldata[i][j];

		testlabel(i) = totallabel[i];
	}

	a.fit(testdata,testlabel);
	std::cout<<"\nFit done\n";
	std::cout<<a.predict(testdata);
	std::cin>>ctr;


	return 0;

}

