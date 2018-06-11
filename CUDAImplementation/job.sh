#!/bin/bash
export PATH=/Soft/cuda/8.0.61/bin:$PATH
### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N KMeans 
# Cambiar el shell
#$ -S /bin/bash
echo "DATASET 1"
./KMeans.exe < ../datasets/dataset1.txt
echo "DATASET 2"
./KMeans.exe < ../datasets/dataset2.txt
echo "DATASET 3"
./KMeans.exe < ../datasets/dataset3.txt
