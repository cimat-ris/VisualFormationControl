#include <list>
#include <vector>
#include <iostream>

#define DIRECTEDGRAPH_H  

// Directed graph 
class DirectedGraph 
{ 
	int V_; // Number of vertices 

	// Pointer to an array containing 
	// adjacency lists 
	std::list<int> *adjList_; 

	// Recursive function used by DFS 
	int DFSRecursive(int v, std::vector<bool> &visited, int lmax, std::list<int> &path); 
public: 
	DirectedGraph(int V);                  // Constructor from number of nodes
	DirectedGraph(int V, int **adjMatrix); // Constructor from adjacency matrix

	// Add a directed edge (v,w) to graph 
	inline void addEdge(int v, int w) { 
		adjList_[v].push_back(w); // Add w to the adjacency list of v. 
	} 

	// DFS traversal of the graph
	int DFS(int v, int lmax); 

	// Load balance
	void loadBalance();

	// Get the maximal load within the graph
	inline std::pair<int,int> getMaxLoad() const {
		int lmax = 0;
		int vmax = 0;
		// Determine the maximum load and the vertex that has it
		for (int v=0;v<V_;v++) if (adjList_[v].size()>lmax) {
			lmax = adjList_[v].size();
			vmax = v;
		}
		return std::make_pair(vmax,lmax);
	}

	// For impression
	friend std::ostream& operator<<(std::ostream& os, const DirectedGraph& g);
	int countConnections(int label, DirectedGraph g);
	void getConnections(int label, DirectedGraph g, int *s);
}; 
