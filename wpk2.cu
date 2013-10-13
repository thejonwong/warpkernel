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

  int minthreshold = 1;
  int maxthreshold = 1;
  if (argc >= 4) minthreshold = atoi(argv[3]);
  if (argc == 5) maxthreshold = atoi(argv[4]);

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
  warpkernel::startDatafile(datafile, nz,N,ntests);

  cusp::array1d<ValueType, CPUSpace> x(N,0);
  thrust::generate(x.begin(),x.end(), rand_float());


  cusp::array1d<ValueType, CPUSpace> y(N);

  // setup multiple run mean accumulators
  // find global minimum and maximum

  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk2all;
  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk2;
  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk2no;
  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::min>  > wk2rex;
  int wk2allblock, wk2block, wk2noblock, wk2rexblock;
  int wk2allth, wk2th, wk2noth, wk2rexth;

  bool lastiter = true;
    // cusp multiplication
    {



      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
      cusp::csr_matrix<IndexType, ValueType, DeviceSpace> A1 = A;
      cusp::array1d<ValueType, DeviceSpace> dx = x;
      cusp::array1d<ValueType, DeviceSpace> dy = y;

      cusp::detail::timer cusptimer;
      cusptimer.start();
      for (int t = 0; t < ntests; t++) {
	cusp::multiply(A1,dx,dy);
      }		
      ValueType measuredtime = cusptimer.seconds_elapsed()/ntests;
      statstime(measuredtime);		
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

      for (int threshold = min_nz; threshold <= max_nz; threshold ++) {
	std::cout << std::endl;
	kernel2.scan(nz, N, A, threshold); // use maximum threshold

	cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
	kernel2.reorder_columns_coalesced(reorder_cols);

	cusp::array1d<ValueType, DeviceSpace> dreordered_x;
	kernel2.reorder_x(x, dreordered_x);


	warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng(kernel2,
									     &(A.values[0]),
									     &(A.column_indices[0]));

	for(int warps_per_block = 2; warps_per_block <= 8; warps_per_block ++) {
	  uint nblocks = (kernel2.nwarps + warps_per_block -1)/warps_per_block;
	  uint blocksize = warps_per_block * WARP_SIZE;    

	  if (nblocks > 65536) continue; // skip if nblocks too high

	  std::cout << std::endl;

	  cusp::array1d<IndexType, DeviceSpace> restore_col = eng.device_colinds;

	  // normal kernel
	  {
	    cusp::array1d<ValueType, DeviceSpace> dy(N,0.);  
	  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
	  ValueType totaltime = 0;
	  for (int t = 0; t < ntests; t++) {
	    ValueType measuretime = eng.run<true>(nblocks, blocksize,
						  thrust::raw_pointer_cast(&dx[0]),
						  thrust::raw_pointer_cast(&dy[0]));
	    cudaUnbindTexture(x_tex);
	    totaltime += measuretime;
	  }
	  statstime(totaltime/ntests);
	    cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	    if (eng.verify(y,ycheck)) {
	      std::cout << "warpkernel2 (" << nblocks << "," << blocksize << "," << threshold << ") time = " 
			<< std::scientific << boost::accumulators::mean(statstime) << std::endl;
	      std::stringstream kernelname;
	      kernelname << "warpkernel2_" << threshold;
	      warpkernel::addData(datafile, (char *) (kernelname.str()).c_str(), 
				  boost::accumulators::mean(statstime), 
				  kernel2, blocksize);

	      wk2all(boost::accumulators::mean(statstime));
	      wk2(boost::accumulators::mean(statstime));
	      if (boost::accumulators::min(wk2all) == boost::accumulators::mean(statstime)) {
		wk2allblock = blocksize;
		wk2allth = threshold;
	      }
	      if (boost::accumulators::min(wk2) == boost::accumulators::mean(statstime)) {
		wk2block = blocksize;
		wk2th = threshold;
	      }
	    } 
	  }

	  // reordered kernel with rowmap
	  {
	    cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
	    eng.device_colinds = reorder_cols;
	  boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
	  ValueType totaltime = 0;
	  for (int t = 0; t < ntests; t++) {
	    ValueType measuretime = eng.run<true>(nblocks, blocksize,
						  thrust::raw_pointer_cast(&dreordered_x[0]),
						  thrust::raw_pointer_cast(&dy[0]));
	    cudaUnbindTexture(x_tex);
	    totaltime += measuretime;
	  }
	  statstime(totaltime/ntests);
	    cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	    if (eng.verify(y,ycheck) && lastiter) {
	      std::cout << "warpkernel2 reorder (" << nblocks << "," << blocksize << "," << threshold << ") time = " 
			<< std::scientific << boost::accumulators::mean(statstime) << std::endl;

	      std::stringstream kernelname;
	      kernelname << "warpkernel2_" << threshold << "rex";
	      warpkernel::addData(datafile, (char *) (kernelname.str()).c_str(), 
				  boost::accumulators::mean(statstime), 
				  kernel2, blocksize);

	      wk2all(boost::accumulators::mean(statstime));
	    wk2rex(boost::accumulators::mean(statstime));
	    if (boost::accumulators::min(wk2all) == boost::accumulators::mean(statstime)) {
	      wk2allblock = blocksize;
	      wk2allth = threshold;
	    }
	    if (boost::accumulators::min(wk2rex) == boost::accumulators::mean(statstime)) {
	      wk2rexblock = blocksize;
	      wk2rexth = threshold;
	    }

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
	    boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
	    ValueType totaltime = 0;
	  for (int t = 0; t < ntests; t++) {
	    ValueType measuretime = eng.run_x<true>(nblocks, blocksize,
						    thrust::raw_pointer_cast(&dreordered_x[0]),
						    thrust::raw_pointer_cast(&dy[0]));
	    cudaUnbindTexture(x_tex);
	    totaltime += measuretime;
	  }
	  statstime(totaltime/ntests);
	    cusp::array1d<ValueType, CPUSpace> ycheck = dy;
	    if (eng.verify_x(y,ycheck) && lastiter) {
	      std::cout << "warpkernel2 reorder w/o rowmap (" << nblocks << "," << blocksize << "," << threshold << ") time = " 
			<< std::scientific << boost::accumulators::mean(statstime) << std::endl;

	      std::stringstream kernelname;
	      kernelname << "warpkernel2_" << threshold << "no";
	      warpkernel::addData(datafile, (char *) (kernelname.str()).c_str(), 
				  boost::accumulators::mean(statstime), 
				  kernel2, blocksize);

	      wk2all(boost::accumulators::mean(statstime));
	    wk2no(boost::accumulators::mean(statstime));
	    if (boost::accumulators::min(wk2all) == boost::accumulators::mean(statstime)) {
	      wk2allblock = blocksize;
	      wk2allth = threshold;
	    }
	    if (boost::accumulators::min(wk2no) == boost::accumulators::mean(statstime)) {
	      wk2noblock = blocksize;
	      wk2noth = threshold;
	    }
	    }// else exit(1);
	    eng.device_colinds = restore_col;

	  }

	}
      }
    

    }


    std::stringstream wpk2allname;
    wpk2allname << "wpk2all_" << wk2allth;
    warpkernel::addData(datafile,  (char *) (wpk2allname.str()).c_str(), boost::accumulators::min(wk2all), -1, -1, -1, wk2allblock);
    std::stringstream wpk2name;
    wpk2name << "wpk2_" << wk2th;
    warpkernel::addData(datafile,  (char *)(wpk2name.str()).c_str(), boost::accumulators::min(wk2), -1, -1, -1, wk2block);
    std::stringstream wpk2noname;
    wpk2noname << "wpk2no_" << wk2noth;
    warpkernel::addData(datafile,  (char *)(wpk2noname.str()).c_str(), boost::accumulators::min(wk2no), -1, -1, -1, wk2noblock);
    std::stringstream wpk2rexname;
    wpk2rexname << "wpk2rex_" << wk2rexth;
    warpkernel::addData(datafile,  (char *)(wpk2rexname.str()).c_str(), boost::accumulators::min(wk2rex), -1, -1, -1, wk2rexblock);


}
