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

// boost
// stats
// #include <boost/accumulators/accumulators.hpp>
// #include <boost/accumulators/statistics/stats.hpp>
// #include <boost/accumulators/statistics/mean.hpp>
// #include <boost/accumulators/statistics/min.hpp>

#define WARP_SIZE 32

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

using namespace warpkernel;

int main(int argc, char *argv[]) {

  // Define stats collection
  stats stats_all, stats_n, stats_r, stats_rx;
  stats stats_csr, stats_hyb;
  
  std::string matrixfilename = argv[1];
  int ntests = 1;
  if (argc >= 3) ntests = atoi(argv[2]);

  int minthreshold = 1;
  int maxthreshold = 1;
  if (argc >= 4) minthreshold = atoi(argv[3]);
  if (argc >= 5) maxthreshold = atoi(argv[4]);

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
 
  std::string datapath = "./data/" + matrixname + "_results_wpk2.txt";
  if (argc==5)
    datapath = "./data/" + matrixname + "_results_wpk2_" + argv[3] + "_" + argv[4] + ".txt";
  std::cout << "Starting data file = " << datapath << std::endl;
  std::ofstream datafile(datapath.c_str());
  //  warpkernel::startDatafile(datafile, nz,N,ntests);

  cusp::array1d<ValueType, CPUSpace> x(N,0);
  thrust::generate(x.begin(),x.end(), rand_float());


  cusp::array1d<ValueType, CPUSpace> y(N);

  // setup multiple run mean accumulators
  // find global minimum and maximum

  bool lastiter = true;
    // cusp multiplication
    {

      cusp::csr_matrix<IndexType, ValueType, DeviceSpace> A1 = A;
      cusp::array1d<ValueType, DeviceSpace> dx = x;
      cusp::array1d<ValueType, DeviceSpace> dy = y;

      profiler csr;
      for (int t = 0; t < ntests; t++) {
	timer cusptimer;
	cusp::multiply(A1,dx,dy);
	ValueType measuredtime = cusptimer.seconds_elapsed();
	csr.add(measuredtime);
      }		

      y = dy;
	
      if (lastiter) {
	printf("CUSP CSR avg time (%d runs) = %3.3e [s]\n", csr.count, csr.avg());
	stats_csr.add(csr.avg(),"CSR");
      }
    }

    // cusp-hyb
    {

      cusp::hyb_matrix<IndexType, ValueType, DeviceSpace> A1 = A;
      cusp::array1d<ValueType, DeviceSpace> dx = x;
      cusp::array1d<ValueType, DeviceSpace> dy = y;

      profiler hyb;
      for (int t = 0; t < ntests; t++) {
	timer cusptimer;
	cusp::multiply(A1,dx,dy);
	ValueType measuredtime = cusptimer.seconds_elapsed();
	hyb.add(measuredtime);
      }

      y = dy;

      if (lastiter) {
	printf("CUSP HYB avg time (%d runs) = %3.3e [s]\n", hyb.count, hyb.avg());
	stats_hyb.add(hyb.avg(),"HYB");
      }
    }

    int inc; // holds min_max increment
    
    // test warpkernel2
    {
      cusp::array1d<ValueType, DeviceSpace> dx = x;

      warpkernel::structure2 kernel2;
      kernel2.scan(nz, N, A, (int)N); // use maximum threshold

      int max_nz = kernel2.max_nz;
      int min_nz = kernel2.min_nz;  

      std::cout << "warp kernel 2" << std::endl;

      if (argc > 3) {
	max_nz = maxthreshold;
	min_nz = minthreshold;
      }

      printf("min_nz = %d, max_nz %d\n", min_nz, max_nz);
      
      // Modified to only test out 5 values in the min-max range just to speed things up
      inc = (max_nz-min_nz)*0.1 > 0 ? (max_nz-min_nz)*0.1 : 1;
      for (int threshold = min_nz; threshold <= max_nz; threshold+=inc) {
	std::cout << std::endl;
	kernel2.scan(nz, N, A, threshold); // use maximum threshold

	cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
	kernel2.reorder_columns_coalesced(reorder_cols);

	cusp::array1d<ValueType, DeviceSpace> dreordered_x;
	kernel2.reorder_x(x, dreordered_x);


	warpkernel::engine<ValueType, IndexType> eng(kernel2,
						     &(A.values[0]),
						     &(A.column_indices[0]));

	for(int warps_per_block = 2; warps_per_block <= 16; warps_per_block ++) {
	  uint nblocks = (kernel2.nwarps + warps_per_block -1)/warps_per_block;
	  uint blocksize = warps_per_block * WARP_SIZE;    

	  if (nblocks > 65536) continue; // skip if nblocks too high

	  std::cout << std::endl;

	  cusp::array1d<IndexType, DeviceSpace> restore_col = eng.device_colinds;

	  // normal kernel
	  {
	    cusp::array1d<ValueType, DeviceSpace> dy(N,0.);
	    profiler norm;

	  for (int t = 0; t < ntests; t++) {
	    ValueType measuretime = eng.run<true>(nblocks, blocksize,
						  thrust::raw_pointer_cast(&dx[0]),
						  thrust::raw_pointer_cast(&dy[0]));
	    cudaUnbindTexture(x_tex);
	    norm.add(measuretime);
	  }
	  // statstime(totaltime/ntests);
	    cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	    if (eng.verify(y,ycheck)) {
	      printf("WPK2 avg time (%d runs) = %e (warps/block = %d, threshold = %d)\n",
		     norm.count, norm.avg(), warps_per_block, threshold);

	      std::ostringstream fmt_string;
	      fmt_string << "warps/block = " << warps_per_block << " th = " << threshold;
	      stats_all.add(norm.avg(), fmt_string.str());
	      stats_n.add(norm.avg(), fmt_string.str());

	    }
	    else {
	      printf("Failed\n");
	      exit(1);
	    }
	  } 

	  // reordered kernel with rowmap
	  {
	    cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
	    eng.device_colinds = reorder_cols;
	    profiler reordered;
	  for (int t = 0; t < ntests; t++) {
	    ValueType measuretime = eng.run<true>(nblocks, blocksize,
						  thrust::raw_pointer_cast(&dreordered_x[0]),
						  thrust::raw_pointer_cast(&dy[0]));
	    cudaUnbindTexture(x_tex);
	    reordered.add(measuretime);
	  }
	    cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	    if (eng.verify(y,ycheck) && lastiter) {
	      printf("WPK2_rx avg time (%d runs) = %e (warps/block = %d, threshold = %d)\n",
		     reordered.count, reordered.avg(), warps_per_block, threshold);


	      std::ostringstream fmt_string;
	      fmt_string << "warps/block = " << warps_per_block << " th = " << threshold;

	      stats_all.add(reordered.avg(), fmt_string.str());
	      stats_rx.add(reordered.avg(), fmt_string.str());


	    } //else exit(1);
	    eng.device_colinds = restore_col;

	  }    

	  // normal kernel counter balances effect of cache
	  {
	    cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
	    ValueType measuretime = eng.run<true>(nblocks, blocksize,
						  thrust::raw_pointer_cast(&dx[0]),
						  thrust::raw_pointer_cast(&dy[0]));
	    cudaUnbindTexture(x_tex);
	    cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	  }


	  // reordered kernel without rowmap
	  {
	    cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
	    eng.device_colinds = reorder_cols;
	    profiler timer;
	    
	    for (int t = 0; t < ntests; t++) {
	      ValueType measuretime = eng.run_x<true>(nblocks, blocksize,
						      thrust::raw_pointer_cast(&dreordered_x[0]),
						      thrust::raw_pointer_cast(&dy[0]));
	      timer.add(measuretime);
	      cudaUnbindTexture(x_tex);
	    }

	    cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	    if (eng.verify_x(y,ycheck) && lastiter) {
	      printf("WPK2_r avg time (%d runs) = %e (warps/block = %d, threshold = %d)\n",
		     timer.count, timer.avg(), warps_per_block, threshold);

	      std::ostringstream fmt_string;
	      fmt_string << "warps/block = " << warps_per_block << " th = " << threshold;

	      stats_all.add(timer.avg(), fmt_string.str());
	      stats_r.add(timer.avg(), fmt_string.str());

	    }// else exit(1);
	    eng.device_colinds = restore_col;

	  }

	}
      }
    

    }


  printf("\n");
  
  // Summary
  printf("CUSP CSR = %e [s]\n", stats_csr.Min());
  printf("CUSP HYB = %e [s]\n", stats_hyb.Min());

  double fastest = min(stats_csr.Min(), stats_hyb.Min());
  
  printf("Fasted WPK2 (all) = %e [s], %2.2fx faster (%s)\n", stats_all.Min(), fastest/stats_all.Min(), stats_all.Min_str().c_str());
  printf("Fasted WPK2       = %e [s], %2.2fx faster (%s)\n", stats_n.Min() , fastest/stats_n.Min(), stats_n.Min_str().c_str());
  printf("Fasted WPK2_r     = %e [s], %2.2fx faster (%s)\n", stats_r.Min() , fastest/stats_r.Min(), stats_r.Min_str().c_str());
  printf("Fasted WPK2_rx    = %e [s], %2.2fx faster (%s)\n", stats_rx.Min(), fastest/stats_rx.Min(), stats_rx.Min_str().c_str());
  printf("Test increment = %d\n", inc);

}
