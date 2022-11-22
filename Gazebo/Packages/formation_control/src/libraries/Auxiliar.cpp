#include "Auxiliar.hpp"

/*
	Function: createMatrix
	description: creates a matrix with dynamic memory 
	params:
		type: sizeof(int) for example
		typePtr: sizeof(int*) for example
		n: number of rows
		m: numver of cols
	returns:
		allocated matrix, the given result needs to be casted to the desired type matrix
*/
char **createMatrix(int type, int typePtr, int n, int m){
	char **M = new char*[typePtr*n];
	for(int i=0;i<n;i++)
		M[i] = new char[type*m];

	return M;
}

/*
	Function: freeMatrix
	description: free the memory of the given matrix. The matrix was allocated
		with createMatrix function
	params:
		M: pointer to the matrix (casted to char **)
		n: number of rows
*/	
void freeMatrix(char **M, int n){
	for(int i=0;i<n;i++)
		delete [] M[i];
	delete [] M;
}

/*
	Function: appendToFile
	description: adds a new row with the given data to the specified file
	params:
		name: name of the file
		val: array with the values to write
		n: size of the array
*/
void appendToFile(string name, double *val, int n){  
	ofstream outfile;
	outfile.open(name, std::ios_base::app);	
	for(int i=0;i<n;i++)
		outfile << val[i] << " " ;
	outfile << endl;
}

/*
	Function: needsAltitudeConsensus
	description: Verifies if the given controller scheme needs the altitude consensus
	params: 
		ct: consensus code
	returns: 
		1: if needs it
		0: if does not
*/
int needsAltitudeConsensus(int ct){
	static const int  n_altitude_schemes=3; int altitude_schemes[n_altitude_schemes]={3,4,7}; //to verifiy if an altitude consensus is needed
	for(int i=0;i<n_altitude_schemes;i++)
		if(ct == altitude_schemes[i]){
			return 1;
		}
	return 0;
}

/*
	Function: isHomographyConsensus
	description: Verifies if the given controller scheme uses homography
	params: 
		ct: consensus code
	returns: 
		1: if needs it
		0: if does not
*/
int isHomographyConsensus(int ct){
	static const int  n_homography_schemes=4; int homography_schemes[n_homography_schemes]={1,3,5,7}; 
	for(int i=0;i<n_homography_schemes;i++)
		if(ct == homography_schemes[i]){
			return 1;
		}
	return 0;
}

/*
	Function: bearingsNeeded
	description: Verifies if the given controller scheme uses bearings
	params: 
		ct: consensus code
	returns: 
		1: if needs it
		0: if does not
*/
int bearingsNeeded(int ct){
	static const int n_bearings_schemes=6; int bearings_schemes[n_bearings_schemes]={1,2,5,6,8,9};

	for(int i=0;i<n_bearings_schemes;i++)
		if(ct == bearings_schemes[i])
			return 1;
	return 0;
}

/*
	Function: needsFilter
	description: Verifies if the given controller scheme uses filters for relative positions
	params: 
		ct: consensus code
	returns: 
		1: if needs it
		0: if does not
*/
int needsFilter(int ct){
	static const int n_filter_schemes=8; int filter_schemes[n_filter_schemes]={1,2,3,4,5,6,8,9};
	for(int i=0;i<n_filter_schemes;i++)
		if(ct == filter_schemes[i])
			return 1;
	return 0;
}

/*
	Function: epipoleConsensus
	description: Verifies if the given controller scheme uses the epipoles
	params: 
		ct: consensus code
	returns: 
		1: if needs it
		0: if does not
*/
int epipoleConsensus(int ct){
	static const int n_epipole_schemes=1; int epipole_schemes[n_epipole_schemes] = {9};
	for(int i=0;i<n_epipole_schemes;i++)
		if(ct == epipole_schemes[i])
			return 1;
	return 0;	
}

/*
	Function: isHomographyConsensus
	description: verifies if the given consensus code is valid
	params:
		ct: consensus code
	returns:
		1: the code is valid
		0: the code is not valid
*/
int isValidConsensus(int ct){
	if (ct >= 1 and ct <= 10)
		return 1;
	return 0;
}

/*
	Function: isValidDesiredFormation
	description: verifies if the given formation code is valid
	params: 
		shape: formation code
	returns:
		1: if is valid
		0: if is not
*/
int isValidDesiredFormation(int shape){
	if(shape >=0 and shape<=2)
		return 1;
	return 0;
}

/*
	function: find
	description: returns the index of a given element on an array
	params:
		x: value to find
		a: array where we will look for the x
		n: number of elements in the array a
	returns: 
		index where x is in the array a
*/
int find(int x, int *a, int n){
	for(int i=0;i<n;i++)
		if(a[i] == x)
			return i;
	return -1;
}

/*
	function: average
	description: computes the average 
	returns: the average
*/
double average(vector<double> &a, int n){
	double avg = 0;
	for(int i=0;i<n;i++)
		avg+=a[i];

	return avg/(double) n;
}

