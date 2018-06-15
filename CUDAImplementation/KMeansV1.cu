#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
using namespace std;

//vector< vector<double> > PointValues; 
//vector< vector<double> > KCentroids;
//vector<int> ClusteringValues;
unsigned int total_points, total_values, K, max_iterations;

#define THREADS 8



__global__ void updateCentroids(double *PointValues, double *KCentroids, 
								double *ClusteringValues, int total_points, int total_values, int K){
	
	int kevaluada = blockIdx.y * blockDim.y + threadIdx.y;
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int ind = j;
	
	float tmp = 0.0;
	int count = 0;
	if (j < total_values) {

		for (int i = 0; i<total_points; ++i, ind = ind + total_values) {
			if (kevaluada == ClusteringValues[i]) {
				tmp += PointValues[ind];
				++count;
			}
		}
		KCentroids[kevaluada * j + total_values] = tmp/count;
	}
}

void printClusters(double *PointValues, double *KCentroids, 
									 double *ClusteringValues);
									 
//void updateCentroids(double *PointValues, double *KCentroids, 
//										 double *ClusteringValues);
										 
bool updatePointDistances();

void CheckCudaError(char sms[], int line);

int main(int argc, char** argv) {

  unsigned int numBytesPointValues, numBytesKCentroids, 
							 numBytesClustering;
							 
  unsigned int nBlocksC, nThreadsC;
 
  cudaEvent_t E1, E2, E3, E4, E5;
  
  float TiempoTotal, TiempoUpdateCentroids, TiempoUpdatePointDistances;

  double *h_PointValues, *h_KCentroids, *h_ClusteringValues;
  
  double *d_PointValues, *d_KCentroids, *d_ClusteringValues;
  
  cin >> total_points >> total_values >> K >> max_iterations;
  
  if(K > total_points)
		cout << "INPUT ERROR: K CANT BE BIGGER THAN TOTAL POINTS" << endl;

	//Reservamos el expacio que necesitaremos en memoria
  numBytesKCentroids = K * total_values * sizeof(double);
  
  numBytesPointValues = total_points * total_values * sizeof(double);
  
  numBytesClustering = total_points * sizeof(double);
  

	//Declaramos los eventos
  cudaEventCreate(&E1);
  
  cudaEventCreate(&E2);
  
  cudaEventCreate(&E3);
  
  cudaEventCreate(&E4);
  
  cudaEventCreate(&E5);


  // Obtener Memoria en el host
  h_PointValues = (double*) malloc(numBytesPointValues); 
  
  h_KCentroids = (double*) malloc(numBytesKCentroids); 
  
  h_ClusteringValues = (double*) malloc(numBytesClustering);

			
	//Lectura de los valores
	for(int i = 0; i < total_points; i++) {

		for(int j = 0; j < total_values; j++) {
			double value;
			cin >> value;
			int ind = i * total_values + j;
			h_PointValues[ind] = value;
		}
		
	}
	vector<int> prohibited_indexes;
	srand(1);
	for(int i = 0; i < K; i++) {
		while(true)
		{
			int index_point = rand() % total_points;

			if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
					index_point) == prohibited_indexes.end())
			{
				prohibited_indexes.push_back(index_point);
				h_ClusteringValues[index_point] = i;
				break;
			}
		}
	}
	
	
	// Obtener Memoria en el device
	cudaMalloc((double**)&d_PointValues, numBytesPointValues); 
	
	cudaMalloc((double**)&d_KCentroids, numBytesKCentroids); 
	
	cudaMalloc((double**)&d_ClusteringValues, numBytesClustering); 
	
	CheckCudaError((char *) "Obtener Memoria en el device", __LINE__); 
	
	
	// Copiar datos desde el host en el device 
	cudaMemcpy(d_PointValues, h_PointValues, numBytesPointValues, 
				cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_KCentroids, h_KCentroids, numBytesKCentroids, 
				cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_ClusteringValues, h_ClusteringValues, 
				numBytesClustering, cudaMemcpyHostToDevice);
	CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);
	

	// Ejecutar el kernel 
	
	nThreadsC = THREADS;
	// nBlocks = Nfil/nThreads;  // Solo funciona bien si Nfil multiplo de nThreads
	cout << "nBlocksC: " << (total_values + nThreadsC - 1)/nThreadsC << endl;
	cout << "total_values: " << total_values << endl;
	cout << "nThreadsC: " << nThreadsC << endl;
	nBlocksC = (total_values + nThreadsC - 1)/nThreadsC;  // Funciona bien en cualquier caso

	dim3 dimGridC(nBlocksC, 1, 1);
	dim3 dimBlockC(nThreadsC, K, 1);
	
	printf("\n");
	printf("Kernel de su puta madre\n");
	printf("Dimension Block: %d x %d x %d (%d) threads\n", dimBlockC.x, dimBlockC.y, dimBlockC.z, dimBlockC.x * dimBlockC.y * dimBlockC.z);
	printf("Dimension Grid: %d x %d x %d (%d) blocks\n", dimGridC.x, dimGridC.y, dimGridC.z, dimGridC.x * dimGridC.y * dimGridC.z);
  
  
	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);
	
	updateCentroids<<<dimGridC,dimBlockC>>>(d_PointValues, d_KCentroids,
	  							d_ClusteringValues, total_points, total_values, K); 
	CheckCudaError((char *) "Invocar Kernel", __LINE__);
					
	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);
	/*int counter = 0;
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
	cudaEventSynchronize(E5);*/


  // Obtener el resultado desde el host 
	//cudaMemcpy(h_PointValues, d_PointValues, numBytesPointValues,
	//											cudaMemcpyDeviceToHost);
	cudaMemcpy(h_KCentroids, d_KCentroids, numBytesKCentroids,
								cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_ClusteringValues, d_ClusteringValues, numBytesClustering,
	//							cudaMemcpyDeviceToHost); 
								
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);
  
  printClusters(h_PointValues, h_KCentroids, h_ClusteringValues);
  

  // Liberar Memoria del device 
  cudaFree(d_PointValues); cudaFree(d_KCentroids); 
  cudaFree(d_ClusteringValues);

  cudaDeviceSynchronize();
  

  cudaEventElapsedTime(&TiempoUpdateCentroids, E1, E2);
  //cudaEventElapsedTime(&TiempoUpdatePointDistances, E3, E4);
  //cudaEventElapsedTime(&TiempoTotal,  E1, E5);
  
  cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);
  cudaEventDestroy(E4); cudaEventDestroy(E5);
 
  printf("Tiempo UpdateCentroids function: %4.6f milseg\n", 
		TiempoUpdateCentroids);
  /*printf("Tiempo UpdatePointDistances function: %4.6f milseg\n", 
		TiempoUpdatePointDistances);
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);*/
  

  free(h_PointValues); free(h_KCentroids); free(h_ClusteringValues);

}


/*bool updatePointDistances(){
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
}*/

/*void updateCentroids(double *PointValues, double *KCentroids, 
										 double *ClusteringValues){
						
	double *updatingK;
	updatingK.resize(KCentroids.size());
	for (int i = 0; i<ClusteringValues.size(); ++i) {
		vector<double> AddingK;
		for (int j = 0; j<PointValues[i].size(); ++j) {
			AddingK.push_back(PointValues[i*total_values+j]);//AddingK.push_back(PointValues[i][j]);
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
}*/


void printClusters(double *PointValues, double *KCentroids, 
									 double *ClusteringValues) {
										 
	for (int i = 0; i<K; ++i) {
		cout << "Centroid " << i << ": ";
		for (int j = 0; j<total_values; ++j) {
			int ind = i * total_values + j;
			cout << KCentroids[ind] << " ";
		}
		cout << endl;
	}
	for (int i = 0; i<total_points; ++i) {
		cout << "Point " << i << ": ";
		for (int j = 0; j<total_values; ++j) {
			int ind = i * total_values + j;
			cout << PointValues[ind] << " ";
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


