#include <iostream>
#include <bitset>
#include "typedef.hpp"
using namespace std;


struct coord
{
	int x,y,z;
}spoints[26];

int main()
{
	for(int i = 0; i < 3; i++)
		for(int j = 0;  j < 3; j++)
			for(int k = 0; k < 3; k++)
			{
				spoints[i*9 + j*3 + k].x = k-1;
				spoints[i*9 + j*3 + k].y = j-1;
				spoints[i*9 + j*3 + k].z = i-1;
			}


}

MatrixBinary get3dLBP(int*** datamat, int x, int y, int z)
{
	MatrixBinary totalmat;
	int index;

	totalmat.resize((x-2)*(y-2)*(z-2), 26);

	for(int i = 1; i < z-1; i++)
	{
		for(int j = 1; j < y-1; j++)
		{
			for(int k = 1; k < x-1; k++)
			{
				for(int l = 0; l < 26; l++)
					{
						index = (i-1)*(x-2)*(y-2) + (j-1)*(x-2) + k-1;
						totalmat(index,l) = (datamat[i][j][k] > datamat[i-spoints[l].x][j-spoints[l].y][k-spoints[l].z]);
					}
			}
		}
	}

	return totalmat;


}
