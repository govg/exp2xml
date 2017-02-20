/*
Author	:	ic070897
Title	:	Node data type definition

This is the header file behind the Node struct. 
*/

#ifndef NODES_H
#define NODES_H

#include "typedef.hpp"

#include <vector>
#include <cmath>
#include <iostream>

//	The definition of the basic structure, Node
struct Node
{
	int x,depth;
	double threshold;
	int start,end;
	VectorReal hist;
	int leftChild,rightChild;
	bool isLeaf;

	Node(){
		isLeaf = false;
		start = end = -1;
		leftChild = rightChild = -1;
		x = depth = -1;
	};
};

#endif /*NODES_H */
