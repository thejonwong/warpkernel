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
 
  std::string datapath = "./data/" + matrixname + "_results_wpk1.txt";
  std::cout << "Starting data file = " << datapath << std::endl;
  std::ofstream datafile(datapath.c_str());
  warpkernel::startDatafile(datafile, nz,N,ntests);

  cusp::array1d<ValueType, CPUSpace> x(N,0);
  thrust::generate(x.begin(),x.end(), rand_float());


  cusp::array1d<ValueType, CPUSpace> y(N);

  // setup multiple run mean accumulators
  // find global minimum and maximum

  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk1all;
  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk1;
  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk1no;
  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk1rex;
  int wk1allblock, wk1block, wk1noblock, wk1rexblock;

  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk2all;
  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk2;
  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk2no;
  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk2rex;
  int wk2allblock, wk2block, wk2noblock, wk2rexblock;
  int wk2allth, wk2th, wk2noth, wk2rexth;

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

    // cusp-hyb
    {
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      cusp::hyb_matrix<IndexType, ValueType, DeviceSpace> A1 = A;
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
	std::cout << "cusp-hyb gpu time " 
		  << std::scientific << boost::accumulators::mean(statstime) << std::endl;
	warpkernel::addData(datafile, "cusp-hyb", boost::accumulators::mean(statstime), -1, -1, -1, -1);
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

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng(kernel1,
									  &(A.values[0]),
									  &(A.column_indices[0]));
   
      std::cout << "warp kernel 1" << std::endl;

      for (int warps_per_block = 1; warps_per_block <= 8; warps_per_block ++) {
	std::cout << std::endl;
	uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
	uint blocksize = warps_per_block * WARP_SIZE;
    
	// normal kernel
	{
	  cusp::array1d<ValueType, DeviceSpace> dy(N,0);  

	  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
	  for (int t = 0; t < ntests; t++) {
	    ValueType measuretime = eng.run<true>(nblocks, blocksize,
						  thrust::raw_pointer_cast(&dx[0]),
						  thrust::raw_pointer_cast(&dy[0]));
	    cudaUnbindTexture(x_tex);
	    statstime(measuretime);
	  }
	  cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	  if (eng.verify(y,ycheck) && lastiter)
	    {
	      std::cout << "warpkernel (" << nblocks << "," << blocksize <<") time = " 
			<< std::scientific << boost::accumulators::mean(statstime) << std::endl;
	      warpkernel::addData(datafile, "warpkernel", boost::accumulators::mean(statstime), kernel1, blocksize);
	      
	      wk1all(boost::accumulators::mean(statstime));
	      wk1(boost::accumulators::mean(statstime));
	      if (boost::accumulators::min(wk1all) == boost::accumulators::mean(statstime)) wk1allblock = blocksize;
	      if (boost::accumulators::min(wk1) == boost::accumulators::mean(statstime)) wk1block = blocksize;
	    } //else exit(1);

	}
	cusp::array1d<IndexType, DeviceSpace> restore_col = eng.device_colinds;

	// reordered kernel without rowmap
	{
	  cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
	  eng.device_colinds = reorder_cols;

	  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
	  for (int t = 0; t < ntests; t++) {
	    ValueType measuretime = eng.run_x<true>(nblocks, blocksize,
						    thrust::raw_pointer_cast(&dreordered_x[0]),
						    thrust::raw_pointer_cast(&dy[0]));
	    cudaUnbindTexture(x_tex);
	    statstime(measuretime);
	  }
	  cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	  if (eng.verify_x(y,ycheck) && lastiter) {
	    std::cout << "warpkernel reorder w/o rowmap (" << nblocks << "," << blocksize <<") time = " 
		      << std::scientific << boost::accumulators::mean(statstime) << std::endl;
	    warpkernel::addData(datafile, "warpkernel_no", boost::accumulators::mean(statstime), kernel1, blocksize);

	    wk1all(boost::accumulators::mean(statstime));
	    wk1no(boost::accumulators::mean(statstime));
	    if (boost::accumulators::min(wk1all) == boost::accumulators::mean(statstime)) wk1allblock = blocksize;
	    if (boost::accumulators::min(wk1no) == boost::accumulators::mean(statstime)) wk1noblock = blocksize;

	  } 
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
	  cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
	  eng.device_colinds = reorder_cols;

	  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
	  for (int t = 0; t < ntests; t++) {
	    ValueType measuretime = eng.run<true>(nblocks, blocksize,
						  thrust::raw_pointer_cast(&dreordered_x[0]),
						  thrust::raw_pointer_cast(&dy[0]));
	    cudaUnbindTexture(x_tex);
	    statstime(measuretime);
	  }
	  cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	  if (eng.verify(y,ycheck) && lastiter) {
	    std::cout << "warpkernel reorder (" << nblocks << "," << blocksize <<") time = " 
		      << std::scientific << boost::accumulators::mean(statstime) << std::endl;
	    warpkernel::addData(datafile, "warpkernel_rex", boost::accumulators::mean(statstime), kernel1, blocksize);

	    wk1all(boost::accumulators::mean(statstime));
	    wk1rex(boost::accumulators::mean(statstime));
	    if (boost::accumulators::min(wk1all) == boost::accumulators::mean(statstime)) wk1allblock = blocksize;
	    if (boost::accumulators::min(wk1rex) == boost::accumulators::mean(statstime)) wk1rexblock = blocksize;

	  } 
	  eng.device_colinds = restore_col;

	}    

      }

    }


    warpkernel::addData(datafile, "wpk1all", boost::accumulators::min(wk1all), -1, -1, -1, wk1allblock);
    warpkernel::addData(datafile, "wpk1", boost::accumulators::min(wk1), -1, -1, -1, wk1block);
    warpkernel::addData(datafile, "wpk1no", boost::accumulators::min(wk1no), -1, -1, -1, wk1noblock);
    warpkernel::addData(datafile, "wpk1rex", boost::accumulators::min(wk1rex), -1, -1, -1, wk1rexblock);

}
