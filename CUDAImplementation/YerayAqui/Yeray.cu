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

#define THREADS 1024



__global__ void updateCentroids(double *PointValues, double *KCentroids, 
								double *ClusteringValues, int total_points, int total_values, int K){
	
	int kevaluada = blockIdx.y * blockDim.y + threadIdx.y;
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int ind = j;
	
	float tmp = 0.0;
	int count = 0;
	if (j < total_values) {

		for (int i = 0; i<total_points; ++i, ind = ind + total_values) {
			//printf("kevaluada: %d \n",kevaluada);
			if (kevaluada == ClusteringValues[i]) {
				tmp += PointValues[ind];
				++count;
			}
		}
		//printf("tmp: %d \n",tmp);
		KCentroids[kevaluada * total_values + j] = tmp/count;
	}
}

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

__global__ void Kernel04(double *DK, int *Ind, int *gInd, double *gBD, int tdk) { //numelem es el numero de threads// optimo = numde k
	//unsigned int total = blockDim.x*gridDim.x;
  /*__shared__ int indexed[THREADS];
  __shared__ double sDK[THREADS];*/

	extern __shared__ double sDKcindexed[];//double

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
  }

  // Hacemos la reduccion en la memoria compartida
  int s = blockDim.x*(blockIdx.x+1) <= tdk ? blockDim.x/2 : (tdk%blockDim.x)/2;//tdk es menor q el nº threas lanzados x definicion
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

}


void printClusters(double *PointValues, double *KCentroids, 
									 double *ClusteringValues);
									 
//void updateCentroids(double *PointValues, double *KCentroids, 
//										 double *ClusteringValues);
										 
bool updatePointDistances();

void CheckCudaError(char sms[], int line);

int main(int argc, char** argv) {

  unsigned int numBytesPointValues, numBytesKCentroids, 
							 numBytesClustering, numBytesDistMatrix;
							 
  unsigned int nBlocksC, nThreadsC, nBlocksYeray,nThreadsYeray;
 
  cudaEvent_t E1, E2, E3, E4, E5;
  
  float TiempoTotal, TiempoUpdateCentroids, TiempoUpdatePointDistances;

  double *h_PointValues, *h_KCentroids, *h_ClusteringValues, *h_DistMatrix;
  
  double *d_PointValues, *d_KCentroids, *d_ClusteringValues, *d_DistMatrix;

	double *d_distres, *d_aux; 
	int *d_indexres, *d_indexaux;
  
  cin >> total_points >> total_values >> K >> max_iterations;
  
  if(K > total_points)
		cout << "INPUT ERROR: K CANT BE BIGGER THAN TOTAL POINTS" << endl;

	//Reservamos el expacio que necesitaremos en memoria
  numBytesKCentroids = K * total_values * sizeof(double);
  
  numBytesPointValues = total_points * total_values * sizeof(double);
  
  numBytesClustering = total_points * sizeof(double);
  
  numBytesDistMatrix = total_points * K * sizeof(double);
 
  

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
  
  h_DistMatrix = (double*) malloc(numBytesClustering);
  

			
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
				//cout << "index_point: " << index_point << endl;
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
    
    cudaMalloc((double**)&d_DistMatrix, numBytesDistMatrix);
	
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
	
	nThreadsC = total_values;
	nBlocksC = (total_values + nThreadsC - 1)/nThreadsC;  // Funciona bien en cualquier caso
	//cout << "nBlocksC: " << (total_values + nThreadsC - 1)/nThreadsC << endl;
	//cout << "total_values: " << total_values << endl;
	//cout << "nThreadsC: " << nThreadsC << endl;
	

	dim3 dimGridC(nBlocksC, 1, 1);
	dim3 dimBlockC(nThreadsC, K, 1);
	
	//printf("\n");
	//printf("Kernel de su puta madre\n");
	//printf("Dimension Block: %d x %d x %d (%d) threads\n", dimBlockC.x, dimBlockC.y, dimBlockC.z, dimBlockC.x * dimBlockC.y * dimBlockC.z);
	//printf("Dimension Grid: %d x %d x %d (%d) blocks\n", dimGridC.x, dimGridC.y, dimGridC.z, dimGridC.x * dimGridC.y * dimGridC.z);
  
  
	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);
	
	updateCentroids<<<dimGridC,dimBlockC>>>(d_PointValues, d_KCentroids,
					d_ClusteringValues, total_points, total_values, K);
	
	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);

	  // Obtener el resultado desde el host 
	//cudaMemcpy(h_PointValues, d_PointValues, numBytesPointValues,
	//											cudaMemcpyDeviceToHost);
	cudaMemcpy(h_KCentroids, d_KCentroids, numBytesKCentroids,
								cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_ClusteringValues, d_ClusteringValues, numBytesClustering,
	//							cudaMemcpyDeviceToHost); 
								
  //CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);
    //cout << "AFTER UPDATING CENTROIDS: " << endl;
	//printClusters(h_PointValues, h_KCentroids, h_ClusteringValues);
	
	
	
	cout << "PRE ALL; " << endl << endl;
	cout << "KCENTROIDS: " << endl;
	for (int i = 0; i<K; ++i) {
		for (int j = 0; j<total_values; ++j) {
			int ind = i * total_values + j;
			cout << h_KCentroids[ind] << " ";
		}
		cout << endl;
	}
	
	
		// Copiar datos desde el host en el device 
	cudaMemcpy(d_PointValues, h_PointValues, numBytesPointValues, 
				cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_KCentroids, h_KCentroids, numBytesKCentroids, 
				cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_ClusteringValues, h_ClusteringValues, 
				numBytesClustering, cudaMemcpyHostToDevice);
				
	cudaMemcpy(d_DistMatrix, h_DistMatrix, 
				numBytesDistMatrix, cudaMemcpyHostToDevice);
				
	CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

    
    //para el calculo de distancias
    nThreadsYeray = 16;
    nBlocksYeray = 16;

	
	//cout << "PORAKI" << endl;
    
    dim3 dimGridY((K+nThreadsYeray-1)/nThreadsYeray, (total_points+nThreadsYeray-1)/nThreadsYeray+1, 1);
	dim3 dimBlockY(nThreadsYeray, nThreadsYeray, 1);
	
	/*printf("\n");
	printf("Kernel de su puta madre\n");
	printf("Dimension Block: %d x %d x %d (%d) threads\n", dimBlockY.x, dimBlockY.y, dimBlockY.z, dimBlockY.x * dimBlockY.y * dimBlockY.z);
	printf("Dimension Grid: %d x %d x %d (%d) blocks\n", dimGridY.x, dimGridY.y, dimGridY.z, dimGridY.x * dimGridY.y * dimGridY.z);*/
	
	cudaEventRecord(E3, 0);
	cudaEventSynchronize(E3);
	UDist<<<dimGridY,dimBlockY>>>(total_values, K, total_points, d_DistMatrix, d_PointValues, d_KCentroids);
	cudaEventRecord(E4, 0);
	cudaEventSynchronize(E4);
    cudaFree(d_PointValues); cudaFree(d_KCentroids); 
  //cudaFree(d_ClusteringValues); cudaFree(d_DistMatrix);
    cudaMemcpy(h_DistMatrix, d_DistMatrix, numBytesDistMatrix,
								cudaMemcpyDeviceToHost);
	
	cout << "DIST MATRIX; " << endl;
	for (int i = 0; i<K; ++i) {
		for (int j = 0; j<total_points; ++j) {
			int ind = i * total_points + j;
			cout << h_DistMatrix[ind] << " ";
		}
		cout << endl;
	}
								
   
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



  

	 //YERAY THINGS
  
  
     //jajaja
    ///calculo de nuevo vecindario
    cout << "PORAKI2" << endl;
    dim3 dimGridY2(1, 1, 1);
	dim3 dimBlockY2(nThreadsYeray, nBlocksYeray, 1);
    
    //unsigned int numJumpBytes = total_points * sizeof(double);
    
    
    //for(int l = 0; l<K; l++){
    //    indexaux[l] = l;
    //}
    
    //numblocks es la dim.x del grid
	int gridtastic = K/16;
cout << "PORAKI2.1" << endl;
	gridtastic += K%16 == 0 ? 0 : 1;
cout << "PORAKI2.2" << endl;
    int *h_indexaux = (int*) malloc(K*sizeof(int));
cout << "PORAKI2.3" << endl;
    //double *h_distres = (double*) malloc(gridtastic*sizeof(double));//THREADS/16 => numblocks
cout << "PORAKI2.4" << endl;
    //int *h_indexres = (int*) malloc(gridtastic*sizeof(int));
cout << "PORAKI2.5" << endl;
    double *h_aux = (double*) malloc(K*sizeof(double));
	cout <<"gridtastic "<< gridtastic*sizeof(int) << endl;
    cout << "PORAKI3" << endl;
    cudaMalloc((double**)&d_distres, gridtastic*sizeof(double)); 
	cout << "PORAKI3.1" << endl;
	cudaMalloc((int**)&d_indexres, gridtastic*sizeof(int));
	cout << "PORAKI3.2" << endl;
cout <<"gridtastico "<< K*sizeof(double) << endl;
	cudaMalloc((double**)&d_aux, K*sizeof(double));
cout << "PORAKI3.3" << endl;
	cudaMalloc((double**)&d_indexaux, K*sizeof(double));
	
	
    cout << "PORAKI4" << endl;
    bool ferran = true;
    for(int i = 0; i<total_points; i++){
        //1º iter
        
        //carga vector de indices inicial
        //int *h_indexaux = (int*) malloc(K*sizeof(int));
        for(int l = 0; l<K; l++){
            h_indexaux[l] = l;
        }
        //carga el vector de distancias inicial
        //double *aux = (double*) malloc(K*(double));
        for(int kk = 0; kk<K; kk++){
            h_aux[kk] = h_DistMatrix[kk*total_points+i];
        }
        cout << "PORAKI5 pre lol" << endl;
        //pasa a device mem
        
        cudaMemcpy(d_aux, h_aux, K*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_indexaux, h_indexaux, K*sizeof(double), cudaMemcpyHostToDevice);
        
        cout << "PORAKI6" << endl;
	cout << "distres" << endl;
	for(int ll = 0; ll<gridtastic; ll++){
		 cout << h_indexaux[ll] << endl;
	}
	cout << "indexaux" << endl;
	for(int ll = 0; ll<gridtastic; ll++){
		 cout << h_indexaux[ll] << endl;
	}

        //reduccion solo funciona en dim.x
        Kernel04<<<dimGridY2, dimBlockY2, gridtastic*2*sizeof(double)>>>(d_aux, d_indexaux, d_indexres, d_distres, K);
        
        cudaMemcpy(h_aux, d_distres, gridtastic*sizeof(double), cudaMemcpyDeviceToHost);//grid.x = numblocks = THREADS/16
        cudaMemcpy(h_indexaux, d_indexres, (gridtastic*sizeof(int)), cudaMemcpyDeviceToHost);
        cout << "PORAKI7" << endl;
	cout << "distres" << endl;
	for(int ll = 0; ll<gridtastic; ll++){
		 cout << h_indexaux[ll] << endl;
	}
	cout << "indexaux" << endl;
	for(int ll = 0; ll<gridtastic; ll++){
		 cout << h_indexaux[ll] << endl;
	}
        
        
        ///16 = numtheards.x
        int hf = gridtastic/16;
        hf += gridtastic%16 == 0 ? 0 : 1;
	cout << "PORAKI8" << endl;
        while(hf>1){
            cudaMemcpy(d_aux, h_aux, hf*sizeof(double), cudaMemcpyHostToDevice);//hf = #result de la redux anterior
            cudaMemcpy(d_indexaux, h_indexaux, hf*sizeof(double), cudaMemcpyHostToDevice);
            
            cout << "PORAKI9" << endl;
		dim3 gridmolona(hf,1,1);
		dim3 blockmolon(16,1,1);
		//print
		cout <<"solo los " << hf << endl;
		for(int pri = 0; pri<K; pri++){
			cout <<"la menor actual es "<< d_aux <<" en el K "<< h_indexaux[pri] << endl;
		}
		//end print

		int sig = hf/16;//16 = num threads;
            sig += hf%16 == 0 ? 0 : 1;
            Kernel04<<<gridmolona, blockmolon, sig*2*sizeof(double)>>>(d_aux, d_indexaux, d_indexres, d_distres, hf);
            cout << "PORAKI10 cool "<< hf << endl;
            hf = sig;
            
            cudaMemcpy(h_aux, d_distres, hf*sizeof(double), cudaMemcpyDeviceToHost);//grid.x = numblocks
            cudaMemcpy(h_indexaux, d_indexres, hf*sizeof(int), cudaMemcpyDeviceToHost);

		//print
		cout <<"solo los " << hf << endl;
		for(int pri = 0; pri<K; pri++){
			cout <<"la menor actual es "<< d_aux <<" en el K "<< h_indexaux[pri] << endl;
		}
		//end print
        }

	cout << "PORAKI11" << endl;
        if(ferran & h_ClusteringValues[i] != h_indexaux[0]){
            ferran = false;
        }
	cout <<"el punto "<<i<<" al cluster "<<h_indexaux[0]<<endl;
        h_ClusteringValues[i] = h_indexaux[0];
    }/**/
    
  
   
	//cout << "PORAKI12" << endl;

	cout << "POST" << endl;
	printClusters(h_PointValues, h_KCentroids, h_ClusteringValues);
  // Liberar Memoria del device 
 // cudaFree(d_PointValues); cudaFree(d_KCentroids); 
  cudaFree(d_ClusteringValues); cudaFree(d_DistMatrix);
 /*jajaja	cudaFree(d_aux);cudaFree(d_distres);
	cudaFree(d_indexres);cudaFree(d_indexaux);
*/
  cudaDeviceSynchronize();
  

  cudaEventElapsedTime(&TiempoUpdateCentroids, E1, E2);
  cudaEventElapsedTime(&TiempoUpdatePointDistances, E3, E4);
  //cudaEventElapsedTime(&TiempoTotal,  E1, E5);
 
 
  printf("Tiempo UpdateCentroids function: %4.6f milseg\n", 
		TiempoUpdateCentroids);
  printf("Tiempo UpdatePointDistances function: %4.6f milseg\n", 
		TiempoUpdatePointDistances);
 /* printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);*/
 
 
	cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);
  cudaEventDestroy(E4); cudaEventDestroy(E5);
   

  free(h_PointValues); free(h_KCentroids); free(h_ClusteringValues);
  free(h_DistMatrix);
 /*jajaja	free(h_distres);free(h_aux); 
	free(h_indexres);free(h_indexaux);*/

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












//2dim/size*k*p
//2dim*k*p
/*__global__ void UDist2(int dim, int nk, int np, double *DK, double *TV, double *KV){
    __shared__ double sTV[SIZE][dim];//fila completa, una K entera y un P entero x SIZE
    __shared__ double sKV[SIZE][dim];
    
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

