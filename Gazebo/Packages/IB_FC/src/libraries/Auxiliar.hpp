/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares.
March 1st, 2019
This code is used to declare some hand mande functions.
*/

/*********************************************************************************** c++ libraries */
#include <fstream>
#include <vector>

using namespace std;

char **createMatrix(int type, int typePtr, int n, int m);//to create a matrix with dynamic memory
void freeMatrix(char **M, int n);//to free the memory of a matrix from previous function
void appendToFile(string name, double *val, int n); //writes a new row to the specified file with the array val
void appendToFile(string name, float *val, int n); //writes a new row to the specified file with the array val
int needsAltitudeConsensus(int ct); //to verify if a controller needs altitude consensus
int isHomographyConsensus(int ct); //to check if the given controller uses the homography or the fundamental matrix
int bearingsNeeded(int ct); //to verify if the consensus works with bearings
int needsFilter(int ct); //to verify if we need to filter the relative positions obtained with descomposition
int epipoleConsensus(int ct); //to verify if it is a consensus that uses epipoles
int isValidConsensus(int ct);//verifies id the consensus is valid
int isValidDesiredFormation(int shape);//verifies if the given desired formation is valid
int find(int x, int *a, int n); //to return the index of a value in a given array
double average(vector<double> &a, int n); //computes average 
