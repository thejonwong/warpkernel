// General
#include <iostream>
#include <algorithm>

// Warpkernel
#include "warpkernel.hpp"

// cusp
#include <cusp/coo_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/detail/timer.h>

#define ValueType double
#define IndexType int
#define DeviceSpace cusp::device_memory
#define CPUSpace cusp::host_memory

struct rand_float {
  float operator() ()
  {
    return ((float)(rand() % 100))/100. + -0.3;
  }
};


#define STR(s) #s
#define allocatemsg(dev) "Mem alloc error (" STR(dev) ")\n"
#define allocate(dev,host,devsize) \
  printf("%s", cudaMalloc( &dev, devsize) == cudaSuccess ? "" : allocatemsg(dev)) ; \
  cudaMemcpy(dev, host, devsize, cudaMemcpyHostToDevice);

#define transfermsg(dev) "Transfer Error (" STR(dev) ")\n"
#define retrieve(dev,host,devsize)                                              \
  printf("%s", cudaMemcpy( host, dev, devsize, cudaMemcpyDeviceToHost)==cudaSuccess ? "" : transfermsg(dev));


bool checkPairOrder(std::pair<uint,uint> pair1, std::pair<uint,uint> pair2) {
  return pair1.second > pair2.second || 
    (pair1.second == pair2.second && pair1.first < pair2.first);
}

/*******/

int main(int argc, char *argv[]) {

  std::string matrixfilename = argv[1];

  cusp::coo_matrix<IndexType, ValueType, CPUSpace> B;
  cusp::io::read_matrix_market_file(B, matrixfilename.c_str());

  cusp::csr_matrix<IndexType, ValueType, CPUSpace> A = B;

  uint N = A.num_cols;
  uint nz = A.num_entries;

  cusp::array1d<ValueType, CPUSpace> x(N);
  thrust::generate(x.begin(),x.end(), rand_float());

  // cusp multiplication
  cusp::array1d<ValueType, CPUSpace> y(N);
  {
    cusp::csr_matrix<IndexType, ValueType, DeviceSpace> A1 = A;
    cusp::array1d<ValueType, DeviceSpace> dx = x;
    cusp::array1d<ValueType, DeviceSpace> dy = y;

    cusp::detail::timer cusptimer;
    cusptimer.start();
    cusp::multiply(A1,dx,dy);
    std::cout << "cusp gpu time " 
	      << std::scientific << cusptimer.seconds_elapsed() << std::endl;
    y = dy;
  }
#define ntests 1
  // test framework for kernel1
  {

    warpkernel::structure kernel1;
    kernel1.scan(nz, N, A);

    // allocate arrays
    cusp::array1d<ValueType, DeviceSpace> device_values; // nz
    cusp::array1d<IndexType, DeviceSpace> device_colinds; // nz
    cusp::array1d<IndexType, DeviceSpace> device_row_map; // nrows
    cusp::array1d<uint, DeviceSpace> device_max_nz_per_row; // nwarps
    cusp::array1d<IndexType, DeviceSpace> device_warp_offsets; // nwarps
    cusp::array1d<ValueType, DeviceSpace> dx = x;
    cusp::array1d<ValueType, DeviceSpace> dy(N);

    kernel1.process(thrust::raw_pointer_cast(&A.values[0]),
    		    thrust::raw_pointer_cast(&A.column_indices[0]),
    		    device_values, device_colinds,
    		    device_row_map, device_max_nz_per_row,
    		    device_warp_offsets);

    // bind texture to dx
    cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));

    uint warps_per_block = 4;
    uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
    uint blocksize = warps_per_block * WARP_SIZE;

    cusp::detail::timer kerneltime;
    kerneltime.start();
    for(int i=0;i<ntests;i++)
    warpkernel::warpKernel<true> <<< nblocks, blocksize >>>
      (kernel1.nrows,
       thrust::raw_pointer_cast(&device_values[0]),
       thrust::raw_pointer_cast(&device_colinds[0]),
       thrust::raw_pointer_cast(&device_row_map[0]),
       thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
       thrust::raw_pointer_cast(&device_warp_offsets[0]),
       thrust::raw_pointer_cast(&dx[0]),
       thrust::raw_pointer_cast(&dy[0]));

    ValueType measuretime = kerneltime.seconds_elapsed()/ntests;

    std::cout << "warpkernel time " << std::scientific << measuretime
	      << " nwarps " << kernel1.nwarps << " nblocks " << nblocks << std::endl;

    cusp::array1d<ValueType, CPUSpace> ycheck = dy;

    // Check results
    {
      bool pass = true;
      for (int i=0; i< kernel1.nrows; i++){
	if (abs((y[i]-ycheck[i])/y[i]) > 1E-8) pass = false;
      }
      std::cout << "Comparing y[i] vs. ycheck[i] : " << (pass == true ? "Passed" : "Failed" ) << std::endl;
    }

    // Serialize results
    std::string serialfile = matrixfilename + ".k1";
    kernel1.save((char *)serialfile.c_str());
    std::cout << "Serializing " + matrixfilename << std::endl;

    warpkernel::structure kernel2;
    kernel2.load((char *)serialfile.c_str());
    std::cout << "Load " << matrixfilename << std::endl;

    warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng(kernel2, thrust::raw_pointer_cast(&A.values[0]),
			     thrust::raw_pointer_cast(&A.column_indices[0]));
 
    measuretime = eng.run<true>(nblocks, blocksize, thrust::raw_pointer_cast(&dx[0]),
	    thrust::raw_pointer_cast(&dy[0]));

    std::cout << "warpkernel (serial) time " << std::scientific << measuretime
	      << " nwarps " << kernel2.nwarps << " nblocks " << nblocks << std::endl;

    // Check results
    {
      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      bool pass = true;
      for (int i=0; i< kernel1.nrows; i++){
	if (abs((y[i]-ycheck[i])/y[i]) > 1E-8) pass = false;
      }
      std::cout << "Comparing y[i] vs. ycheck[i] : " << (pass == true ? "Passed" : "Failed" ) << std::endl;
    }

    // check reorderx results
    std::vector<IndexType> Rinv(N);
    cusp::array1d<ValueType, CPUSpace> x_col = x;
    cusp::array1d<IndexType, CPUSpace> colinds = A.column_indices;
    cusp::array1d<ValueType, DeviceSpace> dx_col;

    kernel2.reorder_columns(colinds);
    cusp::array1d<ValueType, CPUSpace> x_col2 = x;
    kernel2.reorder_x(x_col2, dx_col);

    warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng2(kernel2,
									 thrust::raw_pointer_cast(&A.values[0]),
									 thrust::raw_pointer_cast(&colinds[0]));

    eng2.run_x<true>(nblocks, blocksize, thrust::raw_pointer_cast(&dx_col[0]), thrust::raw_pointer_cast(&dy[0]));

    {
      cusp::array1d<ValueType, CPUSpace> y2 = dy;

      for(int i=0;i<N;i++) {
	if (abs(y[kernel2.row_map[i]]-y2[i]) > 1E-10) {
	  std::cout << i << "\t" << std::scientific << y[kernel2.row_map[i]] << "\t" << std::scientific << y2[i] << std::endl;
	  exit(0);
	}
      }
      std::cout << "reordering seems to work" << std::endl;
    }
		       

  }

  // testing kernel2
  {
    warpkernel::structure2 kernel;
    kernel.scan(nz, N, A, 6);

    // allocate arrays
    cusp::array1d<ValueType, DeviceSpace> dx = x;
    cusp::array1d<ValueType, DeviceSpace> dy(N);

    warpkernel::engine<ValueType,IndexType, warpkernel::structure2> eng(kernel, thrust::raw_pointer_cast(&A.values[0]),
						thrust::raw_pointer_cast(&A.column_indices[0]));

    uint warps_per_block = 6;
    uint nblocks = (kernel.nwarps + warps_per_block-1)/warps_per_block;
    uint blocksize = warps_per_block * WARP_SIZE;

    ValueType measuretime = eng.run<true>(nblocks, blocksize, 
				    thrust::raw_pointer_cast(&dx[0]), thrust::raw_pointer_cast(&dy[0]));

    std::cout << "warpkernel time " << std::scientific << measuretime
	      << " nwarps " << kernel.nwarps << " nblocks " << nblocks << std::endl;

    cusp::array1d<ValueType, CPUSpace> ycheck = dy;

    // Check results
    {
      bool pass = true;
      for (int i=0; i< kernel.nrows; i++){
	if (abs((y[i]-ycheck[i])/y[i]) > 1E-8) pass = false;
      }
      std::cout << "Comparing y[i] vs. ycheck[i] : " << (pass == true ? "Passed" : "Failed" ) << std::endl;
    }

    // Serialize results
    std::string serialfile = matrixfilename + ".k2";
    kernel.save((char *)serialfile.c_str());
    std::cout << "Serializing " + matrixfilename << std::endl;

    warpkernel::structure2 kernel2;
    kernel2.load((char *)serialfile.c_str());

    // check reorderx results
    cusp::array1d<IndexType, CPUSpace> reorder_col = A.column_indices;
    // reorder columns indices of x
    kernel2.reorder_columns(reorder_col);
    // // reorder x
    cusp::array1d<ValueType, CPUSpace> x2 = x;
    kernel2.reorder_x(x2,dx);

    warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng2(kernel2,
									  &A.values[0],
									  &reorder_col[0]);

    std::cout << "warpkernel2 time " << std::scientific <<
      eng2.run_x<true> (nblocks, blocksize, 
			thrust::raw_pointer_cast(&dx[0]), 
			thrust::raw_pointer_cast(&dy[0]))
	      << std::endl;

    // Check results
    {
      ycheck = dy;
      bool pass = eng2.verify_x(y,ycheck);
      std::cout << "reordering :" << (pass == true ? "Passed" : "Failed" ) << std::endl;
    }

    std::cout << "warpkernel2 time " << std::scientific <<
    eng2.run<true> (nblocks, blocksize,
		     thrust::raw_pointer_cast(&dx[0]), 
		     thrust::raw_pointer_cast(&dy[0]))
	      << std::endl;

    // Check results
    {
      ycheck = dy;
      bool pass = eng2.verify(y,ycheck);
      std::cout << "Comparing y[i] vs. ycheck[i] : " << (pass == true ? "Passed" : "Failed" ) << std::endl;
    }


  }

 
  

}
