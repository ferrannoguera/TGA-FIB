#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

using namespace std;

//int total_points, total_values, K, max_iterations;
 
vector< vector<double> > PointValues; 
vector< vector<double> > KCentroids;
vector<int> ClusteringValues;



void printClusters() {
	for (int i = 0; i<KCentroids.size(); ++i) {
		cout << "Centroid " << i << ": ";
		for (int j = 0; j<KCentroids[i].size(); ++j) {
			cout << KCentroids[i][j] << " ";
		}
		cout << endl;
	}
	for (int i = 0; i<PointValues.size(); ++i) {
		cout << "Point " << i << ": ";
		for (int j = 0; j<PointValues[i].size(); ++j) {
			cout << PointValues[i][j] << " ";
		}
		cout << "is located on cluster: " << ClusteringValues[i] << endl;
	}
}

bool updatePointDistances(){
	double sum, min_dist;
	int min_k;
	bool change = false;
	for (int i = 0; i<PointValues.size(); ++i) {
		min_dist = 0.0;
		for (int j = 0; j<KCentroids.size(); ++j) {
			sum = 0.0;
			for (int k = 0; k<PointValues[i].size(); ++k) {
				sum += pow(KCentroids[j][k] -
					   PointValues[i][k], 2.0);
			}
			if (j == 0) {
				min_dist = sqrt(sum);
				min_k = j;
			}
			if (min_dist > sqrt(sum)) {
				min_dist = sqrt(sum);
				min_k = j;
			}
		}
		if (ClusteringValues[i] != min_k) {
			ClusteringValues[i] = min_k;
			change = true;
		}
		//cout << "min_dist: " << min_dist << " min_k: " << min_k << endl;
	}
	return change;
}

void updateCentroids(int total_values){
	vector<vector<double> > updatingK;/*(ClusteringValues.size(), 
							vector<double>(KCentroids[0].size(),0.0));*/
	updatingK.resize(KCentroids.size());
	for (int i = 0; i<ClusteringValues.size(); ++i) {
		vector<double> AddingK;
		for (int j = 0; j<PointValues[i].size(); ++j) {
			AddingK.push_back(PointValues[i][j]);
		}
		for (int j = 0; j<AddingK.size(); ++j) {
			updatingK[ClusteringValues[i]].push_back(AddingK[j]);
		}
	}
	/*cout << "PRINTING UPDATINGK: " << endl;
	for (int i = 0; i<updatingK.size(); ++i) {
		for (int j = 0; j<updatingK[i].size(); ++j) {
			cout << updatingK[i][j] << " ";
		}
		cout << endl;
	}
	cout << "STOP PRINTING K" << endl;*/
	vector<double> KUpdated(total_values,0);
	for (int i = 0; i<updatingK.size(); ++i) {
		//cout << "i: " << i << endl;
		vector<double> KUpdated(total_values,0);
		for (int j = 0; j<updatingK[i].size(); ++j) {
			//cout << "KUPDATED: " << KUpdated[j%total_values] << " j%total_values: " << j%total_values << " updatingK[i][j]: " << updatingK[i][j];
			KUpdated[j%total_values] += updatingK[i][j];
			//cout << "KUPDATED: " << KUpdated[j%total_values] << endl;
		}
		//cout << "FUCKLAVIDA" << endl;
		//cout << endl;
		if (updatingK[i].size() > 0) {
			for (int j = 0; j<KUpdated.size(); ++j) {
				//cout << "B4/KUpdated: " << KUpdated[j] << " ";
				//cout << "THE DIVIDNEDNEOS: " << updatingK[i].size() << " " << total_values << endl;
				KUpdated[j] /= (updatingK[i].size()/total_values);
				//cout << "KUpdated: " << KUpdated[j] << " ";
			}
			//cout << endl;
			KCentroids[i] = KUpdated;
		}
	}
	/*cout << "UPDATED CENTROIDS" << endl;
	for (int i = 0; i<KCentroids.size(); ++i) {
		for (int j = 0; j<KCentroids[i].size(); ++j) {
			cout << KCentroids[i][j] << " ";
		}
		cout << endl;
	}*/
}

void kmeans(int K, int total_points, int total_values, int max_iterations) {
	vector<int> prohibited_indexes;

	// choose K distinct values for the centers of the clusters
	// Sense cap mena de criteri, va posant punts als clusters fins que repeteix algun punt
	// pot inclus no posat tots els punts, as far as i see
	for(int i = 0; i < K; i++)
	{
		while(true)
		{
			int index_point = rand() % total_points;

			if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
					index_point) == prohibited_indexes.end())
			{
				prohibited_indexes.push_back(index_point);
				ClusteringValues[index_point] = i;
				break;
			}
		}
	}
	KCentroids = vector<vector<double> >(K, vector<double>(total_values));
	//printClusters();
	updateCentroids(total_values);
		
	/*for (int i = 0; i < K; ++i) {
		cout << "KCentroid: " << i << endl;
		for (int j = 0; j<KCentroids[i].size(); ++j) {
			cout << KCentroids[i][j] << " ";
		}
		cout << endl;
	}*/
	int counter = 0;
	cout << "Iteracio " << counter << " :" << endl;
	printClusters();
	cout << endl;
	while (updatePointDistances() and counter <= max_iterations) {
		++counter;
		updateCentroids(total_values);
		cout << "Iteracio " << counter << " :" << endl;
		printClusters();
		cout << endl;
	}
	cout << "RESULTAT FINAL" << endl;
	printClusters();
}


int main() {
	int total_points, total_values, K, max_iterations;
	cin >> total_points >> total_values >> K >> max_iterations;
	if(K > total_points)
			cout << "INPUT ERROR";
			
	ClusteringValues.resize(total_points);
	for(int i = 0; i < total_points; i++) {
		vector<double> values;

		for(int j = 0; j < total_values; j++)
		{
			double value;
			cin >> value;
			values.push_back(value);
		}
		PointValues.push_back(values);
	}
	/*for (int i = 0; i<PointValues.size(); ++i) {
		for (int j = 0; j<PointValues[i].size(); ++j) {
			cout << PointValues[i][j] << " ";
		}
		cout << endl;
	}*/
	kmeans(K,total_points, total_values, max_iterations);	
}
