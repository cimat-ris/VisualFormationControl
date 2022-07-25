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

void readLaplacian(char *dir, int ** L , int n, int * neighbors, int* n_neigh, int actual ){
    
    //  TODO: allocate linerly
//     int **L;
    L = new int *[n];
    for(int i = 0; i <n; i++)
        L[i] = new int[n];
    
	fstream inFile;
	int x=0,i=0,j=0;
	//open file
	inFile.open(dir);
	//read file
	while (inFile >> x) {
		L[i][j] = x;
		j++; if(j==n) {i++;j=0;}
	}
	
	//close file
	inFile.close();
    
    
	//search in the corresponding laplacian row
	for(int i=0;i<n;i++)
		if(L[actual][i]==-1){
			neighbors[*n_neigh]=i;
			(*n_neigh)++;
		}

    
//     return L;
}
