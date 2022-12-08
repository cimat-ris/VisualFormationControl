#include "DirectedGraph.hpp"
#include <iostream>

using namespace std;

DirectedGraph::DirectedGraph(int V) : V_(V)
{ 
	adjList_ = new list<int>[V]; 
} 


int DirectedGraph::DFSRecursive(int v, std::vector<bool> &visited, int lmax, std::list<int> &path) 
{ 
	// Mark the current node as visited 
	visited[v] = true; 

	// Explore the adjacent vertices
	for (list<int>::const_iterator it = adjList_[v].begin(); it != adjList_[v].end(); ++it) 
		if (!visited[*it]) {
			if (adjList_[*it].size()<lmax-1) {
				path.push_back(*it);
				return *it;
			}
			int recursion = DFSRecursive(*it, visited, lmax, path); 
			if (recursion!=-1) {
				path.push_back(*it);
				return recursion;
			}
		}
	return -1;	
} 

// DFS traversal of the vertices reachable from v. 
// It uses recursive DFSUtil() 
int DirectedGraph::DFS(int v, int lmax) 
{ 
	// Mark all the vertices as not visited 
	std::vector<bool> visited(V_,false); 
	// The path to be found (if any)
	std::list<int> path;
	// Call the recursive function 
	int recursion = DFSRecursive(v, visited, lmax, path); 
	int vstart = v;
	if (path.size()>0) {
		for (std::list<int>::const_reverse_iterator it=path.rbegin();it!=path.rend();it++) {
			// Flip the edge vstart-*it
			adjList_[vstart].remove(*it);
			adjList_[*it].push_back(vstart);
			vstart = *it;
		}
	} 
	return recursion;
} 

// Apply the load balancing algorithm
void DirectedGraph::loadBalance() 
{

	// Initialize the graph edges in such a way that there is only one edge between two vertices
	// When there are two, simply remove the first one
	for (int v=0;v<V_;v++)
		for (list<int>::const_iterator it = adjList_[v].begin(); it != adjList_[v].end(); ++it) 
			adjList_[*it].remove(v);

			
	// Cycle over calls to DFS
	while (1) {
		std::pair<int,int> maxLoad = getMaxLoad();
		// Call for the path search: if none found, this should be over
		if (DFS(maxLoad.first,maxLoad.second)<0)
			return;
	}

}

ostream& operator<<(ostream& os, const DirectedGraph& g)
{
	for (int v=0;v<g.V_;v++) {
		os << "* " << v << " |";
		for (list<int>::const_iterator it = g.adjList_[v].begin(); it != g.adjList_[v].end(); ++it) 
    		 os << " " << *it;
    	os << endl;
	}
    return os;
}

int DirectedGraph::countConnections(int label, DirectedGraph g){
	int n = 0;
	for(int v=0;v<g.V_;v++){
		if(v == label){
			for (list<int>::const_iterator it = adjList_[label].begin(); it != adjList_[label].end(); ++it) 
				n++;
			break;
		}
	}

	return n;
//     return adjList_[label].size();
}

void DirectedGraph::getConnections(int label, DirectedGraph g, int *s){
	int n = 0;
	for(int v=0;v<g.V_;v++){
		if(v == label){
			for (list<int>::const_iterator it = adjList_[label].begin(); it != adjList_[label].end(); ++it){ 
				s[n] = *it; n++;}
			break;
		}
	}
}
