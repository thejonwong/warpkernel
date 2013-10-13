NVCC = nvcc

FLAGS =  -arch=sm_20 -O3 # -g -G -m64 -arch=sm_20 -Xptxas -abi=no
LIB = -lm -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -lcusparse -I. -lcuda \
	-lboost_serialization 

all: 
	#$(NVCC) $(FLAGS)  -o serialize serialize.cu $(LIB)
	#$(NVCC) $(FLAGS)  -o test test.cu *.o $(LIB)
	#$(NVCC) $(FLAGS)  -o cusparsemgpu cusparse.cu *.o $(LIB)
	$(NVCC) $(FLAGS)  -o wpk1 wpk1.cu *.o $(LIB)
	$(NVCC) $(FLAGS)  -o wpk2 wpk2.cu *.o $(LIB)
	#$(NVCC) $(FLAGS)  -o optimization optimization.cu $(LIB)
	#$(NVCC) $(FLAGS)  -o optimization2 optimize2.cu $(LIB)	
	#$(NVCC) $(FLAGS)  -o optimization optimize.cu $(LIB)
	#$(NVCC) $(FLAGS)  -o reorder reorder.cu $(LIB)