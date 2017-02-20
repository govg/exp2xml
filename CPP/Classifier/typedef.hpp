/*
Author	:	ic070897
Title	:	Type definitons 
*/

#ifndef TYPEDEF_H
#define TYPEDEF_H

#include <Eigen/Dense>
#include <limits>
#include <string>
#include <iostream>

//Following conventions by Ic017914 

//Matrix shorthands

typedef Eigen::MatrixXd MatrixReal;
typedef Eigen::MatrixXi MatrixInteger;
typedef Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic> MatrixBinary;

//Vector shorthands

typedef Eigen::VectorXd VectorReal;
typedef Eigen::VectorXi VectorInteger;
typedef Eigen::Matrix<bool,Eigen::Dynamic,1> VectorBinary;

//2D array shorthands

typedef Eigen::ArrayXXd Array2DReal;
typedef Eigen::ArrayXXi Array2DInteger;
typedef Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> Array2DBinary;

#define RealMax std::numeric_limits<double>::max()
#define RealMin std::numeric_limits<double>::min()

#endif /*TYPEDEF_H*/