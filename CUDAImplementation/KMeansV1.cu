#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
using namespace std;

unsigned int total_points, total_values, K, max_iterations;

#define THREADS 1024
#define SIZE 16 
//#define POINT_DIM 2    

__global__ void updateCentroids(double *PointValues, double *KCentroids,
 int *updatingK, int *indexK, int total_points, int total_values, int K){
	
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = i * total_points;
	if (i<K) {
		int indcopy = ind;
		for (int j = 0; j<indexK[i]; j++, indcopy++){
			for (int k = 0; k<total_values; ++k) {
				KCentroids[i * total_values + k] += PointValues[updatingK[indcopy] * total_values + k];
			}
		}
	}
	__syncthreads();
	if (i<K) {
			for (int k = 0; k<total_values; ++k) {
				KCentroids[i * total_values + k] /= indexK[i];
			}
	}
			
}

/*__global__ void UDist2(int dim, int nk, int np, double *DK, double *TV, double *KV){
    __shared__ double sTV[SIZE * POINT_DIM];//fila completa, una K entera y un P entero x SIZE
    __shared__ double sKV[SIZE * POINT_DIM];
    
    int bx = blockIdx.x; int tx = threadIdx.x;
    int by = blockIdx.y; int ty = threadIdx.y;
    //los #SIZE threads q van a usar la K[N] y la P[M] cargan una parte de ambos, concretamente dim/SIZE valores +1 si no multiplo
    int row = bx * SIZE + tx;
    int col = by * SIZE + ty;
    int indaux = dim/SIZE;
    //carga paralela de sTV y sKV, t*indaux = particion q le toca
    for(int l= 0; l<indaux; l++){
        sTV[tx*dim+ty*indaux+l] = TV[row*dim+ty*indaux+l];
        sKV[ty*dim+tx*indaux+l] = KV[col*dim+ty*indaux+l];
    }
    //carga de las partes no multiplo
    int check = dim%SIZE;
    if(check > 0){
        int actual = tx-ty;
        actual = actual < 0 ? -actual : actual;
        if(actual < check){
            sTV[(tx+1)*dim-check+actual-1] = TV[(row+1)*dim-check+actual-1];//actual-1-check = pos del modulo, siempre q pasemos a fila siguiente
            sKV[(ty+1)*dim-check+actual-1] = KV[(col+1)*dim-check+actual-1];
        }
    }
    __syncthreads();
    //calculo
    if(row < nk && col < np){
        double tmp = 0.0;
        for(int k = 0; k<dim; k++){
            double aux = KV[row*dim+k] - TV[col*dim+k];
            tmp += aux*aux;
        }
        DK[row*np+col] = sqrt(tmp);
    }
}*/


__global__ void UDist(int dim, int nk, int np, double *DK, double *TV, double *KV){
    int col = blockIdx.y * blockDim.y + threadIdx.y;//inv
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < nk && col < np){
	//if(row >= 20 & row < 30)printf("row: %d col: %d\n",row,col);
        double tmp = 0.0;
        double aux;
        int ind1 = row*dim;
        int ind2 = col*dim;
        for(int k = 0; k<dim; k++){
            aux = KV[ind1+k] - TV[ind2+k];
            tmp += aux*aux;
        }
	
        DK[row*np+col] = sqrt(tmp);
    }
}

//__global__ void Kernel04(double *DK, int *Ind, int *gInd, double *gBD, int tdk) { //numelem es el numero de threads// optimo = numde k
	//unsigned int total = blockDim.x*gridDim.x;
  /*__shared__ int indexed[THREADS];
  __shared__ double sDK[THREADS];*/

	/*extern __shared__ double sDKcindexed[];//double

  // Cada thread carga 1 elemento desde la memoria global
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < tdk){
    sDKcindexed[tid] = DK[i];
    sDKcindexed[tdk+tid] = __int2double_rn(Ind[i]);
    __syncthreads();
  /*noo ned tras el impar x bloque
	//1r paso de la redux, evitar los impares
    if(tid == 0){
        if(THREADS%2 == 1){
            if(sDK[0]>sDK[tdk-1]){
                sDK[0] = sDK[tdk-1];
                indexed[tdk] = indexed[tdk-1];
            }
        }
    }
    __syncthreads();*/
//  }

  // Hacemos la reduccion en la memoria compartida
  /*int s = blockDim.x*(blockIdx.x+1) <= tdk ? blockDim.x/2 : (tdk%blockDim.x)/2;//tdk es menor q el nº threas lanzados x definicion
  for (s; s>0; s>>=1) {
    
    if(tid == 0){
        if(s%2==1){
            if(sDKcindexed[0]>sDKcindexed[2*s+1]){
                sDKcindexed[0] = sDKcindexed[2*s+1];
                sDKcindexed[tdk] = sDKcindexed[2*s+1+tdk];
            }
        }
    }
    if (tid < s){
        if(sDKcindexed[tid]>sDKcindexed[tid+s]){
            sDKcindexed[tid] = sDKcindexed[tid + s];
            sDKcindexed[tid+tdk] = sDKcindexed[tid +s +tdk];
        }
    }
    __syncthreads();
  }


  // El thread 0 escribe el resultado de este bloque en la memoria global
  if (tid == 0){
      gBD[blockIdx.x] = sDKcindexed[0];
      gInd[blockIdx.x] = __double2int_rn(sDKcindexed[tdk]);
  }

}*/




void printClusters(double *PointValues, double *KCentroids, 
									 int *ClusteringValues);
									 
									 
bool updatePointDistances();

int main(int argc, char** argv) {

  unsigned int numBytesPointValues, numBytesKCentroids, 
			 numBytesClustering, numBytesupdatingK, numBytesIndexK, numBytesDistMatrix;
							 
  unsigned int nBlocksFil, nThreadsC, nThreadsYeray, nBlocksCol, nBlocksYeray;
 
  cudaEvent_t Y0, Y1, Y2, Y3;
  
  float TiempoUpdateCentroids, TiempoUpdatePointDistances;

  double *h_PointValues, *h_KCentroids, *h_DistMatrix; 

  int *h_ClusteringValues, *h_indexK, *h_updatingK;

  
  double *d_PointValues, *d_KCentroids,  *d_DistMatrix; 
  
  int *d_ClusteringValues, *d_indexK, *d_updatingK;

  
  cin >> total_points >> total_values >> K >> max_iterations;
  
  if(K > total_points)
		cout << "INPUT ERROR: K CANT BE BIGGER THAN TOTAL POINTS" << endl;

	//Reservamos el expacio que necesitaremos en memoria
  numBytesKCentroids = K * total_values * sizeof(double);
  
  numBytesPointValues = total_points * total_values * sizeof(double);
  
  numBytesClustering = total_points * sizeof(int);
  
  numBytesupdatingK = K * total_points * sizeof(int);
  
  numBytesDistMatrix =  K * total_points * sizeof(double);
  
  numBytesIndexK = K * sizeof(int);
  

	//Declaramos los eventos
  cudaEventCreate(&Y0);
  
  cudaEventCreate(&Y1);
  
  cudaEventCreate(&Y2);
  
  cudaEventCreate(&Y3);


  // Obtener Memoria en el host
  h_PointValues = (double*) malloc(numBytesPointValues); 
  
  h_KCentroids = (double*) malloc(numBytesKCentroids); 
  
  h_ClusteringValues = (int*) malloc(numBytesClustering);
  
  h_updatingK = (int*) malloc(numBytesupdatingK); 
  
  h_indexK = (int*) malloc(numBytesIndexK);
  
  h_DistMatrix = (double*) malloc(numBytesDistMatrix);


			
	//Lectura de los valores
	for(int i = 0; i < total_points; i++) {

		for(int j = 0; j < total_values; j++) {
			double value;
			cin >> value;
			int ind = i * total_values + j;
			h_PointValues[ind] = value;
		}
		
	}
	
	
	for (int i = 0; i<total_points; ++i) {
		h_ClusteringValues[i] = 0;
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
	
	for (int i = 0; i<K; ++i) h_indexK[i] = 0;
	
	for (int i = 0; i<total_points; ++i) {
		int ind = h_ClusteringValues[i] * (total_points) + h_indexK[h_ClusteringValues[i]];
		h_updatingK[ind] = i;
		h_indexK[h_ClusteringValues[i]] = 1+h_indexK[h_ClusteringValues[i]];
	}
	
	for (int a = 0; a<K; ++a) {
		for (int b = 0; b<total_values; ++b) {
			h_KCentroids[a * total_values + b] = 0;
		}
	}	


	
	// Obtener Memoria en el device
	cudaMalloc((double**)&d_PointValues, numBytesPointValues); 
	
	cudaMalloc((double**)&d_KCentroids, numBytesKCentroids); 
	
	cudaMalloc((int**)&d_updatingK, numBytesupdatingK);
	
	cudaMalloc((int**)&d_indexK, numBytesIndexK); 
	
	cudaMalloc((int**)&d_ClusteringValues, numBytesClustering); 
		
	
	
	// Copiar datos desde el host en el device 
	cudaMemcpy(d_PointValues, h_PointValues, numBytesPointValues, 
				cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_KCentroids, h_KCentroids, numBytesKCentroids, 
				cudaMemcpyHostToDevice);
				
	cudaMemcpy(d_updatingK, h_updatingK, 
				numBytesupdatingK, cudaMemcpyHostToDevice);			
	
	cudaMemcpy(d_indexK, h_indexK, 
				numBytesIndexK, cudaMemcpyHostToDevice);	
	
	

	// Ejecutar el kernel 
	
	nThreadsC = THREADS;
	nBlocksFil = (K + nThreadsC - 1)/nThreadsC; 
	nBlocksCol = (total_values + nThreadsC - 1)/nThreadsC;

	cout << "nBlocksC: " << nBlocksFil << endl;
	cout << "total_values: " << total_values << endl;
	cout << "nThreadsC: " << nThreadsC << endl;
	

	dim3 dimGridC(1, nBlocksFil, 1);
	dim3 dimBlockC(1, nThreadsC, 1);
	
	printf("\n");
	printf("Kernel UpdateCentroids\n");
	printf("Dimension Block: %d x %d x %d (%d) threads\n", dimBlockC.x, dimBlockC.y, dimBlockC.z, dimBlockC.x * dimBlockC.y * dimBlockC.z);
	printf("Dimension Grid: %d x %d x %d (%d) blocks\n", dimGridC.x, dimGridC.y, dimGridC.z, dimGridC.x * dimGridC.y * dimGridC.z);
  
  
	cudaEventRecord(Y0, 0);
	cudaEventSynchronize(Y0);
	updateCentroids<<<dimGridC,dimBlockC>>>(d_PointValues, d_KCentroids, d_updatingK, d_indexK, total_points, total_values, K); 
	cudaEventRecord(Y1, 0);
	cudaEventSynchronize(Y1);
	
	cudaDeviceSynchronize();
	

	cudaMemcpy(h_KCentroids, d_KCentroids, numBytesKCentroids,
								cudaMemcpyDeviceToHost);
	
	
	//Yeray thoughts
	cudaMalloc((int**)&d_DistMatrix, numBytesDistMatrix); 
	
	
	// Copiar datos desde el host en el device 
	cudaMemcpy(d_PointValues, h_PointValues, numBytesPointValues, 
				cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_KCentroids, h_KCentroids, numBytesKCentroids, 
				cudaMemcpyHostToDevice);
				
	cudaMemcpy(d_DistMatrix, h_DistMatrix, 
				numBytesDistMatrix, cudaMemcpyHostToDevice);			
	
	
	nThreadsYeray = 16;
    nBlocksYeray = 16;
    
    dim3 dimGridY((K+nThreadsYeray-1)/nThreadsYeray, (total_points+nThreadsYeray-1)/nThreadsYeray+1, 1);
	dim3 dimBlockY(nThreadsYeray, nThreadsYeray, 1);
	
	printf("\n");
	printf("Kernel UDist2\n");
	printf("Dimension Block: %d x %d x %d (%d) threads\n", dimBlockY.x, dimBlockY.y, dimBlockY.z, dimBlockY.x * dimBlockY.y * dimBlockY.z);
	printf("Dimension Grid: %d x %d x %d (%d) blocks\n", dimGridY.x, dimGridY.y, dimGridY.z, dimGridY.x * dimGridY.y * dimGridY.z);
	
	cudaEventRecord(Y2, 0);
	cudaEventSynchronize(Y2);
	UDist<<<dimGridY,dimBlockY>>>(total_values, K, total_points, d_DistMatrix, d_PointValues, d_KCentroids);
	cudaEventRecord(Y3, 0);
	cudaEventSynchronize(Y3);
	
	cudaDeviceSynchronize();
	
	
	cudaMemcpy(h_DistMatrix, d_DistMatrix, numBytesDistMatrix,
								cudaMemcpyDeviceToHost);
	
	
	/*cout << endl << endl << endl << "RESULTADO FINAL" << endl;
	printClusters(h_PointValues, h_KCentroids, h_ClusteringValues);*/
	
	
	
	//CheckCudaError((char *) "Invocar Kernel", __LINE__);
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




  

  cudaEventElapsedTime(&TiempoUpdateCentroids, Y0, Y1);
  cudaEventElapsedTime(&TiempoUpdatePointDistances, Y2, Y3);
  //cudaEventElapsedTime(&TiempoTotal,  E1, E5);
  

 
  printf("Tiempo UpdateCentroids function: %4.6f milseg\n", 
		TiempoUpdateCentroids);
  printf("Tiempo UpdatePointDistances function: %4.6f milseg\n", 
		TiempoUpdatePointDistances);
  /*printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);*/
  
    cudaEventDestroy(Y0); cudaEventDestroy(Y1); cudaEventDestroy(Y2);
  cudaEventDestroy(Y3); 
  

   //Liberar Memoria del device 
	cudaFree(d_PointValues); cudaFree(d_KCentroids); 
	cudaFree(d_ClusteringValues); cudaFree(d_updatingK);
	cudaFree(d_indexK); cudaFree(d_DistMatrix);


	

	//Liberar memoria del host
  free(h_PointValues); free(h_KCentroids); free(h_ClusteringValues);
  free(h_updatingK); free(h_indexK); free(h_DistMatrix);

}


void printClusters(double *PointValues, double *KCentroids, 
									 int *ClusteringValues) {
										 
	for (int i = 0; i<K; ++i) {
		cout << "Centroid " << i << ": ";
		for (int j = 0; j<total_values; ++j) {
			int ind = i * total_values + j;
			cout << KCentroids[ind] << " ";
		}
		cout << endl;
	}
	/*for (int i = 0; i<total_points; ++i) {
		cout << "Point " << i << ": ";
		for (int j = 0; j<total_values; ++j) {
			int ind = i * total_values + j;
			cout << PointValues[ind] << " ";
		}
		cout << "is located on cluster: " << ClusteringValues[i] << endl;
	}*/
}


