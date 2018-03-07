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
#include <cusp/hyb_matrix.h>

#define ValueType double
#define IndexType int
#define DeviceSpace cusp::device_memory
#define CPUSpace cusp::host_memory

#define WARP_SIZE 32

struct rand_float {
  ValueType operator() ()
  {
    return ((ValueType)(rand() % 100))/100. - 0.3;
  }
};

using namespace warpkernel;

int main(int argc, char *argv[]) {

  // Define stats collection
  stats stats_all, stats_n, stats_r, stats_rx;
  stats stats_csr, stats_hyb;
  
  std::string matrixfilename = argv[1];
  int ntests = 1;
  if (argc == 3) ntests = atoi(argv[2]);

  cusp::coo_matrix<IndexType, ValueType, CPUSpace> B;
  cusp::io::read_matrix_market_file(B, matrixfilename.c_str());

  printf("Finished Loading %s\n", matrixfilename.c_str());
  
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
 
  std::string datapath = "./data/" + matrixname + "_results_wpk1.txt";
  std::cout << "Starting data file = " << datapath << std::endl;
  std::ofstream datafile(datapath.c_str());

  cusp::array1d<ValueType, CPUSpace> x(N,0);
  thrust::generate(x.begin(),x.end(), rand_float());

  // solution
  cusp::array1d<ValueType, CPUSpace> y(N);


  bool lastiter = true;
  // cusp multiplication
  {
    profiler csr;
    cusp::csr_matrix<IndexType, ValueType, DeviceSpace> A1 = A;
    cusp::array1d<ValueType, DeviceSpace> dx = x;
    cusp::array1d<ValueType, DeviceSpace> dy = y;

    for (int t = 0; t < ntests; t++) {
      timer cusptimer;
      cusp::multiply(A1,dx,dy);
      ValueType measuredtime = cusptimer.seconds_elapsed();
      csr.add(measuredtime);
    }		
    y = dy;
	
    if (lastiter) {
      printf("CUSP CSR avg time (%d runs) = %3.3e [s]\n", csr.count, csr.avg());
      stats_csr.add(csr.avg());
    }
  }

  // cusp-hyb
  {
    profiler hyb;
    cusp::hyb_matrix<IndexType, ValueType, DeviceSpace> A1 = A;
    cusp::array1d<ValueType, DeviceSpace> dx = x;
    cusp::array1d<ValueType, DeviceSpace> dy = y;

    for (int t = 0; t < ntests; t++) {
      timer cusptimer;
      cusp::multiply(A1,dx,dy);
      ValueType measuredtime = cusptimer.seconds_elapsed();
      hyb.add(measuredtime);
    }

    y = dy;

    if (lastiter) {
      printf("CUSP HYB avg time (%d runs) = %3.3e [s]\n", hyb.count, hyb.avg());
      stats_hyb.add(hyb.avg());
    }
  }

  // warp kernel tests
  {
    cusp::array1d<ValueType, DeviceSpace> dx = x;
    warpkernel::structure kernel1;
    kernel1.scan(nz, N, A);

    cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
    kernel1.reorder_columns_coalesced(reorder_cols);

    cusp::array1d<ValueType, DeviceSpace> dreordered_x;
    kernel1.reorder_x(x, dreordered_x);

    warpkernel::engine<ValueType, IndexType> eng(kernel1,
						 &(A.values[0]),
						 &(A.column_indices[0]));
   
    std::cout << "warp kernel 1" << std::endl;


    
    for (int warps_per_block = 1; warps_per_block <= 16; warps_per_block ++) {
      std::cout << std::endl;
      uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
      uint blocksize = warps_per_block * WARP_SIZE;
    
      // normal kernel
      {
	profiler norm;
	cusp::array1d<ValueType, DeviceSpace> dy(N,0);  

	for (int t = 0; t < ntests; t++) {
	  ValueType measuretime = eng.run<true>(nblocks, blocksize,
						thrust::raw_pointer_cast(&dx[0]),
						thrust::raw_pointer_cast(&dy[0]));
	  norm.add(measuretime);
	  cudaUnbindTexture(x_tex);
	}
	cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	if (eng.verify(y,ycheck) && lastiter)
	  {
	    printf("WPK1 avg time (%d runs) = %e (warps/block = %d)\n",
		   norm.count, norm.avg(), warps_per_block);

	    stats_all.add(norm.avg());
	    stats_n.add(norm.avg());
	    
	  } else exit(1);

      }
      cusp::array1d<IndexType, DeviceSpace> restore_col = eng.device_colinds;

      // reordered kernel without rowmap
      {
	profiler reordered;
	cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
	eng.device_colinds = reorder_cols;

	for (int t = 0; t < ntests; t++) {
	  ValueType measuretime = eng.run_x<true>(nblocks, blocksize,
						  thrust::raw_pointer_cast(&dreordered_x[0]),
						  thrust::raw_pointer_cast(&dy[0]));
	  reordered.add(measuretime);
	  cudaUnbindTexture(x_tex);
	}
	cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	if (eng.verify_x(y,ycheck) && lastiter) {
	  printf("WPK1_r avg time (%d runs) = %e (warps/block = %d)\n",\
		 reordered.count, reordered.avg(), warps_per_block); 
	  stats_all.add(reordered.avg());
	  stats_r.add(reordered.avg());

	} else
	  printf("Failed\n");

	// reset columns
	eng.device_colinds = restore_col;

      }
      // Normal kernel to counter act cache effect
      {
	cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
	ValueType measuretime = eng.run<true>(nblocks, blocksize,
					      thrust::raw_pointer_cast(&dx[0]),
					      thrust::raw_pointer_cast(&dy[0]));
	
	cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      } 


      // reordered kernel with rowmap
      {
	profiler rowmap;
	cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
	eng.device_colinds = reorder_cols;
	
	for (int t = 0; t < ntests; t++) {
	  ValueType measuretime = eng.run<true>(nblocks, blocksize,
						thrust::raw_pointer_cast(&dreordered_x[0]),
						thrust::raw_pointer_cast(&dy[0]));
	  rowmap.add(measuretime);
	  cudaUnbindTexture(x_tex);
	}
	cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	if (eng.verify(y,ycheck) && lastiter) {

	  printf("WPK1_rx avg time (%d runs) = %e (warps/block = %d)\n",
		 rowmap.count, rowmap.avg(), warps_per_block);
	  stats_all.add(rowmap.avg());
	  stats_rx.add(rowmap.avg());


	} else
	  printf("failed\n");
	eng.device_colinds = restore_col;

      }    
    }
  }

  printf("\n");
  
  // Summary
  printf("CUSP CSR = %e [s]\n", stats_csr.Min());
  printf("CUSP HYB = %e [s]\n", stats_hyb.Min());

  double fastest = min(stats_csr.Min(), stats_hyb.Min());
  
  printf("Fasted WPK1 (all) = %e [s], %2.2fx faster\n", stats_all.Min(), fastest/stats_all.Min());
  printf("Fasted WPK1       = %e [s], %2.2fx faster\n", stats_n.Min() , fastest/stats_n.Min());
  printf("Fasted WPK1_r     = %e [s], %2.2fx faster\n", stats_r.Min() , fastest/stats_r.Min());
  printf("Fasted WPK1_rx    = %e [s], %2.2fx faster\n", stats_rx.Min(), fastest/stats_rx.Min());

  
  
}
