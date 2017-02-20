#include <iostream>
#include <vector>
#include <fstream>

using namespace std;


int main()
{
	bool temp;
	std::vector<bool> arr;
	std::vector<std::vector<bool>> mat;

	int temp2;
	std::vector<int> arr2;

	int dims, num;

	fstream features,labels, output;

	features.open("data.bin", ios::in | ios::binary);
	labels.open("labels.bin", ios::in | ios::binary);
	output.open("trainingdata.bin", ios::out | ios::binary);

	features.read((char*)&dims, sizeof(dims));

	while(!features.eof())
	{
		for(int i = 0; i < dims; i++)
		{
			features.read((char*)&temp, sizeof(temp));
			arr.push_back(temp);
		}
		mat.push_back(arr);
		arr.clear();			
	}

	while(!labels.eof())
	{
		labels.read((char*)&temp2, sizeof(temp2));
		arr2.push_back(temp2);
	}

	output.write((char*)&dims, sizeof(dims));
	output.write((char*)&num, sizeof(num));


	for(int i = 0; i < num; i++)
	{
		for(int j = 0; j < dims; j++)
			output.write((char*)&mat[i][j], sizeof(mat[i][j]));

		output.write((char*)&arr2[i], sizeof(arr2[i]));
	}


	return 0;

}


