// General
#include <iostream>
#include <algorithm>
#include <sstream>
#include <assert.h>
// Warpkernel
#include "warpkernel.hpp"

// cusp
#include <cusp/coo_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/detail/timer.h>
#include <cusp/hyb_matrix.h>

// mgpu
#include "../benchmark.h"

// boost
// stats
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>

#define ValueType double
#define IndexType int
#define DeviceSpace cusp::device_memory
#define CPUSpace cusp::host_memory

struct rand_float {
  ValueType operator() ()
  {
    return ((ValueType)(rand() % 100))/100. - 0.3;
  }
};


int main(int argc, char *argv[]) {

  std::string matrixfilename = argv[1];
  int ntests = 1;
  if (argc == 3) ntests = atoi(argv[2]);

  cusp::coo_matrix<IndexType, ValueType, CPUSpace> B;
  cusp::io::read_matrix_market_file(B, matrixfilename.c_str());

  cusp::csr_matrix<IndexType, ValueType, CPUSpace> A = B;

  uint N = A.num_cols;
  uint nz = A.num_entries;

  // open up data file
  std::string filename;
  size_t pos = matrixfilename.find_last_of("/");
  std::string matrixname;
  if (pos != std::string::npos )
    matrixname.assign(matrixfilename.begin()+pos+1, matrixfilename.end());
  else
    matrixname = matrixfilename;
 
  std::string datapath = "./data/" + matrixname + "_results_cusparsemgpu.txt";
  std::cout << "Starting data file = " << datapath << std::endl;
  std::ofstream datafile(datapath.c_str());
  warpkernel::startDatafile(datafile, nz,N,ntests);

  cusp::array1d<ValueType, CPUSpace> x(N,0);
  //  thrust::generate(x.begin(),x.end(), rand_float());


  cusp::array1d<ValueType, CPUSpace> y(N);

  // setup multiple run mean accumulators
  // find global minimum and maximum

  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min> > mgpustats;
  int fastestValuesPerThread=4;


  bool lastiter = true;
    // cusp multiplication
    {



      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
      cusp::csr_matrix<IndexType, ValueType, DeviceSpace> A1 = A;
      cusp::array1d<ValueType, DeviceSpace> dx = x;
      cusp::array1d<ValueType, DeviceSpace> dy = y;

      for (int t = 0; t < ntests; t++) {
	cusp::detail::timer cusptimer;
	cusptimer.start();
	cusp::multiply(A1,dx,dy);
	ValueType measuredtime = cusptimer.seconds_elapsed();
	statstime(measuredtime);		
      }		
      y = dy;
	
      if (lastiter) {
	std::cout << "cusp gpu time " 
		  << std::scientific << boost::accumulators::mean(statstime) << std::endl;
	warpkernel::addData(datafile, "cusp-csr", boost::accumulators::mean(statstime), -1, -1, -1, -1);
      }
    }

    // Modern GPU Benchmarks

    {
      cusp::array1d<ValueType, DeviceSpace> dx = x;

      sparseEngine_t mgpu;
      sparseStatus_t status = sparseCreate("/home/jonathan/mgpu/sparse/src/cubin/", &mgpu);

      if(SPARSE_STATUS_SUCCESS != status) {
	printf("Could not create MGPU Sparse object: %s.\n",
	       sparseStatusString(status));
	return 0;
      }

      std::auto_ptr<SparseMatrix<double> > m;
      std::string err;

      bool success = ReadSparseMatrix(matrixfilename.c_str(), SparseOrderRow, 
				      &m, err);


      std::vector<int> rowIndices, colIndices;
      std::vector<double> sparseValues;
      DeinterleaveMatrix(*m, rowIndices, colIndices, sparseValues);
  
      int threads[] = {4,6,8,10,12,16};
      for(int vals = 0; vals < 6 ; vals++) {
	int valuesPerThread = threads[vals];
	sparseMat_t mat = 0;
	status = sparseMatCreate(mgpu, m->height, m->width, SPARSE_PREC_REAL8, valuesPerThread,
				 SPARSE_INPUT_CSR, (int)m->elements.size(), &(A.values[0]),
				 &(A.row_offsets[0]), &(A.column_indices[0]), &mat);
  

	if(SPARSE_STATUS_SUCCESS != status) return status;

	cusp::array1d<ValueType, DeviceSpace> dy(N);  

	
	boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
	for (int i=0;i<ntests;i++) {
	  cusp::detail::timer mgputime; mgputime.start();
	  status = sparseMatVecDMul(mgpu, 1.0, mat, (CUdeviceptr) thrust::raw_pointer_cast(&dx[0]), 0.0, 
				    (CUdeviceptr)thrust::raw_pointer_cast(&dy[0]));
	  ValueType measuretime = mgputime.seconds_elapsed();
	  statstime(measuretime);
	}



	std::cout << "mgpu time" << valuesPerThread << "\t" <<  boost::accumulators::mean(statstime)  << " s "  << " status :" << status << std::endl;
	
	cusp::array1d<ValueType, CPUSpace> ycheck = dy;

	bool check = true;
	for(int i=0;i<N;i++) {
	  if (abs(y[i]-ycheck[i]) > 1E-5) {
	    check = false;
	    break;
	  }
	}

	if (check) {
	  std::stringstream kernelname;
	  kernelname << "mgpu_" << valuesPerThread;
	  warpkernel::addData(datafile, (char *) (kernelname.str()).c_str(),  boost::accumulators::mean(statstime) , -1,-1,-1,-1);


	  mgpustats(boost::accumulators::mean(statstime));
	  if (boost::accumulators::min(mgpustats) == boost::accumulators::mean(statstime)) fastestValuesPerThread = valuesPerThread;
	}


      }


    }

    // CUSPARSE
    {

      cusp::array1d<ValueType, DeviceSpace> dx = x;
      cusparseStatus_t status2;
      cusparseHandle_t handle = 0; // cusparse library handle
      cusparseMatDescr_t descra;

      /* initialize cusparse library */
      status2 = cusparseCreate(&handle);
      if (status2 != CUSPARSE_STATUS_SUCCESS) {
	return EXIT_FAILURE;
      }

      /* create and setup matrix descriptor */
      status2= cusparseCreateMatDescr(&descra);
      if (status2 != CUSPARSE_STATUS_SUCCESS) {
	printf("Matrix descriptor initialization failed\n");
	return EXIT_FAILURE;
      }
      cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);

      cusp::array1d<ValueType, DeviceSpace> dy(N);  
      cusp::csr_matrix<IndexType, ValueType, DeviceSpace> dA = A;

      
      /* exercise Level 2 routines (csrmv) */
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
      for(int i=0;i<ntests;i++){
	cusp::detail::timer cusparse;
	cusparse.start();
	status2 = cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, 1.0,
				 descra, 
				 thrust::raw_pointer_cast(&dA.values[0]),
				 thrust::raw_pointer_cast(&dA.row_offsets[0]), 
				 thrust::raw_pointer_cast(&dA.column_indices[0]), 
				 thrust::raw_pointer_cast(&dx[0]),
				 0.0, 
				 thrust::raw_pointer_cast(&dy[0]));
	ValueType measuretime = cusparse.seconds_elapsed();
	statstime(measuretime);
      }



      printf("%f time elapsed for multiplication cusparse\n", boost::accumulators::mean(statstime) );  

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;

      bool check = true;
      for(int i=0;i<N;i++) {
	if (abs(y[i]-ycheck[i]) > 1E-5) {
	  check = false;
	  break;
	}
      }

      if (check) {
	  std::stringstream kernelname;
	  kernelname << "cusparse";
	  warpkernel::addData(datafile, (char *) (kernelname.str()).c_str(),  boost::accumulators::mean(statstime) , -1,-1,-1,-1);
      }

    }



    std::stringstream mgpuname;
    mgpuname << "mgpuall_"<< fastestValuesPerThread;
    warpkernel::addData(datafile, (char*)(mgpuname.str()).c_str(), boost::accumulators::min(mgpustats), -1, -1, -1, -1);


}
