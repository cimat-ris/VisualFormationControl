#include <iostream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

/*
	Function: writeFile
	description: Writes the vect given as param into a file with the specified name
	params: std:vector containing the info and file name
*/
void writeFile(vector<float> &vec, const string& name){
	ofstream myfile;
		myfile.open(name);
	for(int i=0;i<vec.size();i++)
		myfile << vec[i] << endl;
	myfile.close();
}
