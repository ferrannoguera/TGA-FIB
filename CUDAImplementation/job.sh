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
echo "TEST 1: 500 Puntos x 10 Dimensiones x 10 K"
echo " "
echo "Sequencial:"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansSeq.exe < ../datasets/dataset1.txt
echo " "
echo "Paralelo"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansV1.exe < ../datasets/dataset1.txt
echo " "
echo " "
echo "TEST 2: 500 Puntos x 100 Dimensiones x 10 K"
echo " "
echo "Sequencial:"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansSeq.exe < ../datasets/dataset2.txt
echo " "
echo "Paralelo"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansV1.exe < ../datasets/dataset2.txt
echo " "
echo " "
echo "TEST 3: 500 Puntos x 300 Dimensiones x 10 K"
echo " "
echo "Sequencial:"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansSeq.exe < ../datasets/dataset3.txt
echo " "
echo "Paralelo"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansV1.exe < ../datasets/dataset3.txt
echo " "
echo " "
echo "TEST 4: 500 Puntos x 10 Dimensiones x 100 K"
echo " "
echo "Sequencial:"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansSeq.exe < ../datasets/dataset4.txt
echo " "
echo "Paralelo"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansV1.exe < ../datasets/dataset4.txt
echo " "
echo " "
echo "TEST 5: 500 Puntos x 10 Dimensiones x 300 K"
echo " "
echo "Sequencial:"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansSeq.exe < ../datasets/dataset5.txt
echo " "
echo "Paralelo"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansV1.exe < ../datasets/dataset5.txt
echo " "
echo " "
echo "TEST 6: 2750 Puntos x 500 Dimensiones x 500 K"
echo " "
echo "Sequencial:"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansSeq.exe < ../datasets/dataset6.txt
echo " "
echo "Paralelo"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansV1.exe < ../datasets/dataset6.txt
echo " "
echo " "
echo "TEST 7: 2750 Puntos x 750 Dimensiones x 500 K"
echo " "
echo "Sequencial:"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansSeq.exe < ../datasets/dataset7.txt
echo " "
echo "Paralelo"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansV1.exe < ../datasets/dataset7.txt
echo " "
echo " "
echo "TEST 8: 2750 Puntos x 1000 Dimensiones x 500 K"
echo " "
echo "Sequencial:"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansSeq.exe < ../datasets/dataset8.txt
echo " "
echo "Paralelo"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansV1.exe < ../datasets/dataset8.txt
echo " "
echo " "
echo "TEST 9: 2750 Puntos x 500 Dimensiones x 750 K"
echo " "
echo "Sequencial:"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansSeq.exe < ../datasets/dataset9.txt
echo " "
echo "Paralelo"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansV1.exe < ../datasets/dataset9.txt
echo " "
echo " "
echo "TEST 10: 2750 Puntos x 500 Dimensiones x 1000 K"
echo " "
echo "Sequencial:"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansSeq.exe < ../datasets/dataset10.txt
echo " "
echo "Paralelo"
nvprof --unified-memory-profiling off --print-gpu-summary ./KMeansV1.exe < ../datasets/dataset10.txt
