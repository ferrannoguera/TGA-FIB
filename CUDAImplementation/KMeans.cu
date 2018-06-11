#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
using namespace std;

vector< vector<double> > PointValues; 
vector< vector<double> > KCentroids;
vector<int> ClusteringValues;



void printClusters();
void updateCentroids(int total_values);
bool updatePointDistances();
void CheckCudaError(char sms[], int line);

int main(int argc, char** argv)

{
  //unsigned int N;
  //unsigned int numBytes;
  //unsigned int nBlocks, nThreads;
 
  cudaEvent_t E0, E1, E2, E3, E4, E5;
  float TiempoTotal, TiempoUpdateCentroids, TiempoUpdatePointDistances;

  //float *h_x, *h_y, *H_y;
  //float *d_x, *d_y;

  //N = 1024 * 1024 * 16;
  //nThreads = 1024;
  //nBlocks = N/nThreads;  // Solo funciona bien si N multiplo de nThreads
  //numBytes = N * sizeof(float);


  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);
  cudaEventCreate(&E4);
  cudaEventCreate(&E5);

  // Obtener Memoria en el host
  //h_x = (float*) malloc(numBytes); 
  //h_y = (float*) malloc(numBytes); 
  //H_y = (float*) malloc(numBytes);    // Solo se usa para comprobar el resultado

  // Obtiene Memoria [pinned] en el host
  //cudaMallocHost((float**)&h_x, numBytes); 
  //cudaMallocHost((float**)&h_y, numBytes); 
  //cudaMallocHost((float**)&H_y, numBytes);   // Solo se usa para comprobar el resultado
  
  cudaEventRecord(E0, 0);
  cudaEventSynchronize(E0);
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
	//kmeans(K,total_points, total_values, max_iterations);
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
	KCentroids = vector<vector<double> >(K, vector<double>(total_values));
	  // Ejecutar el kernel 
	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);
	updateCentroids(total_values); 
	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);
	//updateCentroids<<<1, 1>>>(total_values);
	//CheckCudaError((char *) "Invocar Kernel", __LINE__);
	
	int counter = 0;
	cudaEventRecord(E3, 0);
	cudaEventSynchronize(E3);
	bool yeray = updatePointDistances();
	cudaEventRecord(E4, 0);
	cudaEventSynchronize(E4);
	while (yeray and counter <= max_iterations) {
		++counter;
		updateCentroids(total_values);
		yeray = updatePointDistances();
	}
	cudaEventRecord(E5, 0);
	cudaEventSynchronize(E5);
 
 
  // Obtener Memoria en el device
  //cudaMalloc((float**)&d_x, numBytes); 
  //cudaMalloc((float**)&d_y, numBytes); 
  //CheckCudaError((char *) "Obtener Memoria en el device", __LINE__); 

  // Copiar datos desde el host en el device 
  //cudaMemcpy(d_x, h_x, numBytes, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_y, h_y, numBytes, cudaMemcpyHostToDevice);
  //CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);





  //cudaEventRecord(E2, 0);
  //cudaEventSynchronize(E2);

  // Obtener el resultado desde el host 
  // Guardamos el resultado en H_y para poder comprobar el resultado
  //cudaMemcpy(H_y, d_y, numBytes, cudaMemcpyDeviceToHost); 
  //CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Liberar Memoria del device 
  //cudaFree(d_x); cudaFree(d_y);

  //cudaDeviceSynchronize();

  //cudaEventRecord(E3, 0);
  //cudaEventSynchronize(E3);


  cudaEventElapsedTime(&TiempoUpdateCentroids, E1, E2);
  cudaEventElapsedTime(&TiempoUpdatePointDistances, E3, E4);
  cudaEventElapsedTime(&TiempoTotal,  E0, E5);
 
  //printf("nThreads: %d\n", nThreads);
  //printf("nBlocks: %d\n", nBlocks);
  printf("Tiempo UpdateCentroids function: %4.6f milseg\n", 
		TiempoUpdateCentroids);
  printf("Tiempo UpdatePointDistances function: %4.6f milseg\n", 
		TiempoUpdatePointDistances);
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);

  //cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

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
	}
	return change;
}

void updateCentroids(int total_values){
	vector<vector<double> > updatingK;
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
	vector<double> KUpdated(total_values,0);
	for (int i = 0; i<updatingK.size(); ++i) {
		vector<double> KUpdated(total_values,0);
		for (int j = 0; j<updatingK[i].size(); ++j) {
			KUpdated[j%total_values] += updatingK[i][j];
		}
		if (updatingK[i].size() > 0) {
			for (int j = 0; j<KUpdated.size(); ++j) {
				KUpdated[j] /= (updatingK[i].size()/total_values);
			}
			KCentroids[i] = KUpdated;
		}
	}
}


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


