cmake_minimum_required(VERSION 2.8.8)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND CMAKE_POLICY)

PROJECT(randomForestCpp)

set(CMAKE_CXX_FLAGS "-I/usr/include/eigen3 -fpermissive")
#VC include directory (optional)
#SET(VC_DIR "C:/Program Files (x86)/Microsoft Visual Studio 10.0/VC/include")

#Eigen directory
#SET(Eigen_DIR "D:/Home/setup/eigen")

#include_directories(${VC_DIR} ${Eigen_DIR} )

FILE(GLOB clf_src_files Classifier/*.cpp)
FILE(GLOB clf_header_files Classifier/*.hpp)

#FILE(GLOB ercf_src_files ERCForest/*.cpp)
#FILE(GLOB ercf_header_files ERCForest/*.hpp)

#FILE(GLOB gbf_src_files GenBinaryForest/*.cpp)
#FILE(GLOB gbf_header_files GenBinaryForest/*.hpp)

ADD_EXECUTABLE(clf ${clf_src_files} ${clf_header_files})
#ADD_EXECUTABLE(ercf ${ercf_src_files} ${ercf_header_files})
#ADD_EXECUTABLE(gbf ${gbf_src_files} ${gbf_header_files})
