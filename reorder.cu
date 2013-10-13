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

#define DeviceSpace cusp::device_memory
#define CPUSpace cusp::host_memory

struct rand_float {
  double operator() ()
  {
    return ((double)(rand() % 100))/100. - 0.3;
  }
};


#define ValueType double
#define IndexType int

int main(int argc, char *argv[]) {

  bool cache = true;
  int warps_per_block = 1; 
  int warps_per_block2= 2;
  std::string matrixfilename = argv[1];
  int ntests = 1;
  int threshold = 4;
  if (argc >2 ) ntests = atoi(argv[2]);
  if (argc >3) cache = (1==atoi(argv[3]));
  if (argc >4) warps_per_block = atoi(argv[4]);
  if (argc >5) threshold = atoi(argv[5]);
  if (argc >6) warps_per_block2 = atoi(argv[6]);

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
 
  std::string datapath = "./data/" + matrixname + "_reorder_" + (cache ? "cache" : "nocache" ) + ".txt";
  std::cout << "Starting data file = " << datapath << std::endl;
  std::ofstream datafile(datapath.c_str());
  warpkernel::startDatafile(datafile, nz,N,ntests);

  cusp::array1d<ValueType, CPUSpace> x(N,0);
  thrust::generate(x.begin(),x.end(), rand_float());


  cusp::array1d<ValueType, CPUSpace> y(N);

  // cusp-hyb
  {
    boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

    cusp::hyb_matrix<IndexType, ValueType, DeviceSpace> A1 = A;
    cusp::array1d<ValueType, DeviceSpace> dx = x;
    cusp::array1d<ValueType, DeviceSpace> dy(N,0);

    for (int t = 0; t < ntests; t++) {
      cusp::detail::timer cusptimer;
      cusptimer.start();
      cusp::multiply(A1,dx,dy);
      ValueType measuredtime = cusptimer.seconds_elapsed();
      statstime(measuredtime);
    }

    y = dy;


    std::cout << "cusp-hyb gpu time " 
	      << std::scientific << boost::accumulators::mean(statstime) << std::endl;

  }


  { 
    warpkernel::structure kernel1;
    kernel1.scan(nz, N, A);

    uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
    uint blocksize = warps_per_block * WARP_SIZE;

    cusp::array1d<ValueType, DeviceSpace> dx = x;
    cusp::array1d<ValueType, DeviceSpace> dy(N,0);  

    warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng(kernel1,
									&(A.values[0]),
									&(A.column_indices[0]));

    std::cout << "wpk1 time : " << eng.run<true>(nblocks, blocksize,
						 thrust::raw_pointer_cast(&dx[0]),
						 thrust::raw_pointer_cast(&dy[0])) << std:: endl;


    cusp::array1d<ValueType, CPUSpace> ycheck = dy;

    std::cout << (eng.verify(y,ycheck) ? "Passed" : "Failed" ) << std::endl;

    std::cout << "Reorder values on the GPU" << std::endl;

    // copy over values first
    cusp::array1d<ValueType, DeviceSpace> dA_values = A.values;
    cusp::array1d<IndexType, DeviceSpace> dcolinds = A.column_indices;

    // overwrite engine values
    cusp::array1d<uint, DeviceSpace> dreorder_rows(kernel1.reorder_rows.begin(),
						   kernel1.reorder_rows.end());

    // time GPU reordering
    cusp::detail::timer GPUreorder; GPUreorder.start();
    thrust::scatter(dA_values.begin(), dA_values.end(),
		    dreorder_rows.begin(),
		    eng.device_values.begin());
    ValueType GPUreorder_time = GPUreorder.seconds_elapsed();
    std::cout << "GPU reordering time: " << GPUreorder_time << std::endl;

    warpkernel::addData(datafile, "reorderGPU1", GPUreorder_time, -1, -1, -1, -1);    

    std::cout << "wpk1 (thrust GPU reordered) time : " 
	      << eng.run<true>(nblocks, blocksize,thrust::raw_pointer_cast(&dx[0]),
			       thrust::raw_pointer_cast(&dy[0])) << std:: endl;


    // time CPU reordering
    cusp::array1d<ValueType, CPUSpace> new_values(kernel1.allocate_nz,0);

    cusp::detail::timer CPUreorder; CPUreorder.start();
    for (int i=0;i<nz; i++) {
      new_values[kernel1.reorder_rows[i]] = A.values[i];
    }
    ValueType CPUreorder_time = CPUreorder.seconds_elapsed();
    std::cout << "CPU reordering time: " << CPUreorder_time << std::endl;

    warpkernel::addData(datafile, "reorderCPU1", CPUreorder_time, -1, -1, -1, -1);    

    eng.device_values = new_values;

    std::cout << "wpk1 (CPU reordered) time : " 
	      << eng.run<true>(nblocks, blocksize,thrust::raw_pointer_cast(&dx[0]),
			       thrust::raw_pointer_cast(&dy[0])) << std:: endl;


    std::cout << std::endl << "GPU vs. CPU time" << std::endl;
    std::cout << std::scientific << GPUreorder_time << "\t" << std::scientific << CPUreorder_time << std::endl;

  }   

 { 
    warpkernel::structure2 kernel1;
    kernel1.scan(nz, N, A, threshold);

    uint nblocks = (kernel1.nwarps + warps_per_block2-1)/warps_per_block2;
    uint blocksize = warps_per_block2 * WARP_SIZE;

    cusp::array1d<ValueType, DeviceSpace> dx = x;
    cusp::array1d<ValueType, DeviceSpace> dy(N,0);  

    warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng(kernel1,
									&(A.values[0]),
									&(A.column_indices[0]));

    std::cout << "wpk1 time : " << eng.run<true>(nblocks, blocksize,
						 thrust::raw_pointer_cast(&dx[0]),
						 thrust::raw_pointer_cast(&dy[0])) << std:: endl;


    cusp::array1d<ValueType, CPUSpace> ycheck = dy;

    std::cout << (eng.verify(y,ycheck) ? "Passed" : "Failed" ) << std::endl;

    std::cout << "Reorder values on the GPU" << std::endl;

    // copy over values first
    cusp::array1d<ValueType, DeviceSpace> dA_values = A.values;
    cusp::array1d<IndexType, DeviceSpace> dcolinds = A.column_indices;

    // overwrite engine values
    cusp::array1d<uint, DeviceSpace> dreorder_rows(kernel1.reorder_rows.begin(),
						   kernel1.reorder_rows.end());

    // time GPU reordering
    cusp::detail::timer GPUreorder; GPUreorder.start();
    thrust::scatter(dA_values.begin(), dA_values.end(),
		    dreorder_rows.begin(),
		    eng.device_values.begin());
    ValueType GPUreorder_time = GPUreorder.seconds_elapsed();
    std::cout << "GPU reordering time: " << GPUreorder_time << std::endl;

    warpkernel::addData(datafile, "reorderGPU2", GPUreorder_time, -1, -1, -1, -1);    

    std::cout << "wpk2 (thrust GPU reordered) time : " 
	      << eng.run<true>(nblocks, blocksize,thrust::raw_pointer_cast(&dx[0]),
			       thrust::raw_pointer_cast(&dy[0])) << std:: endl;


    // time CPU reordering
    cusp::array1d<ValueType, CPUSpace> new_values(kernel1.allocate_nz,0);

    cusp::detail::timer CPUreorder; CPUreorder.start();
    for (int i=0;i<nz; i++) {
      new_values[kernel1.reorder_rows[i]] = A.values[i];
    }
    ValueType CPUreorder_time = CPUreorder.seconds_elapsed();
    std::cout << "CPU reordering time: " << CPUreorder_time << std::endl;

    warpkernel::addData(datafile, "reorderCPU2", CPUreorder_time, -1, -1, -1, -1);    

    eng.device_values = new_values;

    std::cout << "wpk2 (CPU reordered) time : " 
	      << eng.run<true>(nblocks, blocksize,thrust::raw_pointer_cast(&dx[0]),
			       thrust::raw_pointer_cast(&dy[0])) << std:: endl;


    std::cout << std::endl << "GPU vs. CPU time" << std::endl;
    std::cout << std::scientific << GPUreorder_time << "\t" << std::scientific << CPUreorder_time << std::endl;

  } 


}
