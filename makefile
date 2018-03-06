NVCC = nvcc

FLAGS =  -arch=sm_30  -g -G 
LIB = -lm -L/usr/local/cuda/lib64 -lcusparse
CUSP_DIR = ../cusp

all: 
	#$(NVCC) $(FLAGS)  -o serialize serialize.cu $(LIB)
	#$(NVCC) $(FLAGS)  -o test test.cu *.o $(LIB)
	#$(NVCC) $(FLAGS)  -o cusparsemgpu cusparse.cu *.o $(LIB)
	$(NVCC) $(FLAGS)  -o wpk1 wpk1.cu -I$(CUSP_DIR) $(LIB)
#	$(NVCC) $(FLAGS)  -o wpk2 wpk2.cu *.o $(LIB)
	#$(NVCC) $(FLAGS)  -o optimization optimization.cu $(LIB)
	#$(NVCC) $(FLAGS)  -o optimization2 optimize2.cu $(LIB)	
	#$(NVCC) $(FLAGS)  -o optimization optimize.cu $(LIB)
	#$(NVCC) $(FLAGS)  -o reorder reorder.cu $(LIB)
