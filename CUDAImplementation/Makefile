CUDA_HOME   = /Soft/cuda/8.0.61


NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -arch=compute_35 -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DPINNED=1

EXE00           = KMeansSeq.exe
EXE01           = KMeansV1.exe


OBJ00           = KMeansSeq.o
OBJ01           = KMeansV1.o


default: $(EXE00)

KMeansSeq.o: KMeansSeq.cu
	$(NVCC) -c -o $@ KMeansSeq.cu $(NVCC_FLAGS) $(PROG_FLAGS)
KMeansV1.o: KMeansV1.cu
	$(NVCC) -c -o $@ KMeansV1.cu $(NVCC_FLAGS) $(PROG_FLAGS)



$(EXE00): $(OBJ00)
	$(NVCC) $(OBJ00) -o $(EXE00) $(LD_FLAGS)

$(EXE01): $(OBJ01)
	$(NVCC) $(OBJ01) -o $(EXE01) $(LD_FLAGS)




all:    $(EXE00) $(EXE01) 

clean:
	rm -rf *.o KMeans*.exe

