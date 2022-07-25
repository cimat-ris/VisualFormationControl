#include <iostream>
#include <string>
#include <vector>

/*
	Function: writeFile
	description: Writes the vect given as param into a file with the specified name
	params: std:vector containing the info and file name
*/
void writeFile(std::vector<float> &vec, const std::string& name);
void readLaplacian(char *dir, int ** L , int n, int * neighbors, int* n_neig, int actual );

