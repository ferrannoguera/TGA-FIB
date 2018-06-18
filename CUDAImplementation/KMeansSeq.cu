#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <algorithm>
using namespace std;

vector< vector<double> > PointValues; 
vector< vector<double> > KCentroids;
vector< vector<double> > KPDist;
vector<int> ClusteringValues;



void printClusters();

float GetTime(void)        {
  struct timeval tim;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  tim=ru.ru_utime;
  return ((double)tim.tv_sec + (double)tim.tv_usec / 1000000.0)*1000.0;
}


//Updatea los nuevos valores de K segun los cambios que ha habido en
//la assignacion de puntos
void updateCentroids(int total_values){
	vector<vector<int> > updatingK;
	updatingK.resize(KCentroids.size());
	for (int i = 0; i<ClusteringValues.size(); ++i) {
		updatingK[ClusteringValues[i]].push_back(i);
	}
	
	for (int i = 0; i<KCentroids.size(); ++i) {
		for (int j = 0; j<total_values; ++j) {
			KCentroids[i][j] = 0;
		}
	}
	
	
	for (int i = 0; i<updatingK.size(); ++i) {
		for (int j = 0; j<updatingK[i].size(); ++j) {
			for (int k = 0; k < PointValues[updatingK[i][j]].size(); ++k) {
				KCentroids[i][k] += PointValues[updatingK[i][j]][k];
			}
		}
	}
	
	for (int i = 0; i<KCentroids.size(); ++i) {
		for (int j = 0; j<KCentroids[i].size(); ++j) {
			KCentroids[i][j] /= updatingK[i].size();
		}
	}
}

bool updatePointDistances();
bool updateNeighborhood();
void updateDist();
void CheckCudaError(char sms[], int line);

int main(int argc, char** argv) {

 
  cudaEvent_t E1, E2, E3, E4, E5;
  float TiempoTotal, TiempoUpdateCentroids, TiempoUpdatePointDistances;

  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);
  cudaEventCreate(&E4);
  cudaEventCreate(&E5);
  
  
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
	vector<int> prohibited_indexes;
	srand(1);
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
	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);
	KCentroids = vector<vector<double> >(K, vector<double>(total_values));
	KPDist = vector<vector<double> >(K, vector<double>(total_points));
	updateCentroids(total_values); 
	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);
	
	
	printClusters();
	
	int counter = 0;
	cudaEventRecord(E3, 0);
	cudaEventSynchronize(E3);
	updateDist();
	bool yeray = updateNeighborhood();
	cudaEventRecord(E4, 0);
	cudaEventSynchronize(E4);
	while (yeray and counter <= max_iterations) {
		++counter;
		updateCentroids(total_values);
		updateDist();
		yeray = updateNeighborhood();

	}
	cout << "LLAMADAS A UPDATECENTROIDS: " << counter << endl;
	cout << "LLAMADAS A UPDATEPOINTDISTANCES: " << counter+1 << endl;
	cudaEventRecord(E5, 0);
	cudaEventSynchronize(E5);


  cudaEventElapsedTime(&TiempoUpdateCentroids, E1, E2);
  cudaEventElapsedTime(&TiempoUpdatePointDistances, E3, E4);
  cudaEventElapsedTime(&TiempoTotal,  E1, E5);
  
	printf("Tiempo UpdateCentroids: %4.6f milseg\n", TiempoUpdateCentroids);
	printf("Tiempo UpdatePointDistances function: %4.6f milseg\n", 
		TiempoUpdatePointDistances);
	printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);

  cudaEventDestroy(E1); 
  cudaEventDestroy(E2); cudaEventDestroy(E3);
  cudaEventDestroy(E4); cudaEventDestroy(E5);
 

}

//Updatea la distancia de los puntos con las nuevas K's (si hay algun
//cambio retorna true, else false
bool updatePointDistances(){
	double sum, min_dist;
	int min_k;
	bool change = false;
	for (int i = 0; i<PointValues.size(); ++i) {
		min_dist =
		 0.0;
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
	}
	return change;
}


void updateDist(){
	for(int i = 0; i<KPDist.size(); i++){
		for(int j = 0; j<KPDist[0].size(); j++){
			double tmp = 0.0;
			for(int k = 0; k<PointValues.size(); k++){
				tmp += pow(KCentroids[i][k]-PointValues[j][i], 2.0);
			}
			KPDist[i][j] = sqrt(tmp);
		}
	}	
}

bool updateNeighborhood(){
	bool ret = false;
	for(int i = 0; i<KPDist[0].size(); i++){
		double mini = KPDist[0][i];
		int ind = 0;
		for (int j = 1; j<KPDist.size(); ++j){
			if(KPDist[j][i] < mini){
				mini = KPDist[j][i];
				ind = j;
			}
		}
		if(ClusteringValues[i] != ind){
			ret = true;
			ClusteringValues[i] = ind;
		}
	}	
	return ret;
}

void printClusters() {
	for (int i = 0; i<KCentroids.size(); ++i) {
		cout << "Centroid " << i << ": ";
		for (int j = 0; j<KCentroids[i].size(); ++j) {
			cout << KCentroids[i][j] << " ";
		}
		cout << endl;
	}
}

int error(float a, float b) {

  if (abs (a - b) / a > 0.000001) return 1;
  else  return 0;

}

void CheckCudaError(char sms[], int line) {
  cudaError_t error;
 
  error = cudaGetLastError();
  if (error) {
    printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
    exit(EXIT_FAILURE);
  }


}


