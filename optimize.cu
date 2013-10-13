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

template <bool usecache>
__inline__ __device__ double fetch_cache(const int& i, const double* x) {
  if (usecache) {
    int2 v = tex1Dfetch(x_tex,i);
    return __hiloint2double(v.y, v.x);
  } else {
    return x[i];
  }
}

/**** Kernels *******/

template <bool usecache, typename ValueType, typename IndexType >
__global__ void warpKernel_nocoalesced(uint nrows, ValueType* A, IndexType *colinds, 
				       IndexType *rowmap, uint* maxrows, IndexType *warp_offset,
				       ValueType* x, ValueType* y) {

  const uint tid = threadIdx.x;
  const uint id = tid  + blockIdx.x * blockDim.x;
  const uint wid = tid & (WARP_SIZE-1);
  const uint warpid = id / WARP_SIZE;

  if (id < nrows) {

    uint maxnz = maxrows[warpid];
    IndexType toffset = warp_offset[warpid] + wid * maxnz;
    maxnz += toffset;

    ValueType sum = A[toffset] * fetch_cache<usecache> (colinds[toffset],x);
    for(toffset ++; toffset < maxnz; toffset ++) {
      sum += A[toffset] * fetch_cache<usecache> (colinds[toffset],x);
    }

    y[rowmap[id]] = sum;
  }
}


/********************/

template<typename R>
bool verify(R orig, R comp, uint nrows) {
  bool check = true;
  for (int i=0; i< nrows; i++) {
    if (abs((orig[i]-comp[i])/orig[i]) > 1E-5) {
      std::cout << orig[i] << "\t" << comp[i] << "\t" << i << std::endl;
      check = false;
      return check;
    }
  }
  return check;
}

template<typename R>
bool verify_x(R orig, R comp, uint nrows, int *row_map) {
  bool temp = true;
  for (int i=0; i< nrows; i++) {
    if (abs((orig[row_map[i]]-comp[i])/orig[row_map[i]]) > 1E-5) {
      std::cout << orig[row_map[i]] << "," << comp[i] << " : " << i << std::endl;
      temp = false;
      return temp;
    }
  }
  return temp;
}

bool checkPairOrder(std::pair<uint,uint> pair1, std::pair<uint,uint> pair2) {
  return pair1.second > pair2.second || 
    (pair1.second == pair2.second && pair1.first < pair2.first);
}


template<typename ValueType>
void sort(ValueType *rows, std::vector<std::pair<uint,uint> > & nnz_r,
	  uint nrows, uint &nwarps, uint &nznrows) {

  nwarps = (nrows + WARP_SIZE-1)/(WARP_SIZE);
  nnz_r.resize(nrows);
  nznrows = 0;
  // Re-arrange rows to reach our assumptions
  for (int w = 0; w < nwarps; w++) {
    for (int r = WARP_SIZE * w; r < nrows && r < WARP_SIZE*(w+1); r++) {
      uint rowsize = rows[r+1] - rows[r];
      nnz_r[r] = std::make_pair(r,rowsize);
      if (rowsize > 0)
	nznrows++;
    }
  }

  // sort by rowsize
  std::sort( nnz_r.begin(), nnz_r.end(), checkPairOrder);
}

template <bool bSort,typename ValueType, typename IndexType>
void scan(uint & nz, uint & nrows, 
	  ValueType * A, IndexType * rows, IndexType *colinds,
	  uint & nwarps, uint & nznrows,
	  uint & allocate_nz,
	  std::vector<uint> & reorder_rows, // new_values[reorder_rows[i]] = A[i]
	  std::vector<int> & warp_offsets,
	  std::vector<uint> & max_nz_rows,
	  std::vector<int> &row_map,
	  std::vector<int> &row_map_inv) {

  std::vector<std::pair<uint,uint> > nnz_r; // non-zeros per row
  if (bSort) {
    sort(rows, nnz_r, nrows, nwarps, nznrows);
  } else {
    nnz_r.resize(nrows);
    nwarps = (nrows + WARP_SIZE-1)/(WARP_SIZE);
    nznrows = 0;
    // Re-arrange rows to reach our assumptions
    for (int w = 0; w < nwarps; w++) {
      for (int r = WARP_SIZE * w; r < nrows && r < WARP_SIZE*(w+1); r++) {
	uint rowsize = rows[r+1] - rows[r];
	nnz_r[r] = std::make_pair(r,rowsize);
	if (rowsize > 0)
	  nznrows++;
      }
    }
  }

  std::vector<int> map(nrows,0);
  for(int r = 0; r < nrows; r++)
    map[r] = nnz_r[r].first;

  row_map.resize(map.size());
  row_map = map;
  nrows = row_map.size();
  row_map_inv.resize(nrows);
  for(int i=0;i<nrows;i++) {
    row_map_inv[row_map[i]] = i;
  }

  nwarps = (nznrows + WARP_SIZE-1)/WARP_SIZE;

  std::vector<uint> A_w(nwarps,0); // max non-zeros per row
  std::vector<uint> nnz_imin(nwarps,nrows); // minimum non-zeros per row
  std::vector<uint> nnz_imax(nwarps,0); // maximum non-zeros per row

  // Use sorted row-sizes to calculate A_w, nnz_w, etc.
  for (int w = 0; w < nwarps; w++) {
    for (int r = WARP_SIZE * w; r < nznrows && r < WARP_SIZE*(w+1); r++) {
      uint rowsize = nnz_r[r].second;
      if (rowsize < nnz_imin[w]) nnz_imin[w] = rowsize; // min
      if (rowsize > nnz_imax[w]) nnz_imax[w] = rowsize; // max
    }

    A_w[w] = nnz_imax[w];
  }
  max_nz_rows = A_w;
    
  // set warp_offsets and allocate_nz;
  warp_offsets.resize(nwarps+1);
  warp_offsets[0] = 0;
  for(int w = 0; w < nwarps; w++) {
    assert(A_w[w] > 0);
    warp_offsets[w+1] = warp_offsets[w] + A_w[w] * WARP_SIZE;
  }
  allocate_nz = warp_offsets[nwarps];

  // Generate reordering map for future use
  reorder_rows.resize(nz,0);
  for (int w_s = 0; w_s < nwarps; w_s++) {
    for (int r_s = WARP_SIZE * w_s; r_s < nznrows 
	   && r_s < WARP_SIZE * (w_s+1); r_s++) {
      int r = nnz_r[r_s].first;
      int rowsize = nnz_r[r_s].second;
      for(int i = 0; i < rowsize; i++) {
	assert(rows[r]+i < nz);
	reorder_rows[rows[r] + i] =
	  warp_offsets[w_s] + (r_s % WARP_SIZE) + i*WARP_SIZE;
      }
    }
  }
}


template <bool bSort,typename ValueType, typename IndexType>
void scan_nocoalesced(uint & nz, uint & nrows, 
		      ValueType * A, IndexType * rows, IndexType *colinds,
		      uint & nwarps, uint & nznrows,
		      uint & allocate_nz,
		      std::vector<uint> & reorder_rows, // new_values[reorder_rows[i]] = A[i]
		      std::vector<int> & warp_offsets,
		      std::vector<uint> & max_nz_rows,
		      std::vector<int> &row_map,
		      std::vector<int> &row_map_inv){

  std::vector<std::pair<uint,uint> > nnz_r; // non-zeros per row
  if (bSort) {
    sort(rows, nnz_r, nrows, nwarps, nznrows);
  } else {
    nnz_r.resize(nrows);
    nwarps = (nrows + WARP_SIZE-1)/(WARP_SIZE);
    nznrows = 0;
    // Re-arrange rows to reach our assumptions
    for (int w = 0; w < nwarps; w++) {
      for (int r = WARP_SIZE * w; r < nrows && r < WARP_SIZE*(w+1); r++) {
	uint rowsize = rows[r+1] - rows[r];
	nnz_r[r] = std::make_pair(r,rowsize);
	if (rowsize > 0)
	  nznrows++;
      }
    }
  }

  std::vector<int> map(nrows,0);
  for(int r = 0; r < nrows; r++)
    map[r] = nnz_r[r].first;

  row_map.resize(map.size());
  row_map = map;
  nrows = row_map.size();
  row_map_inv.resize(nrows);
  for(int i=0;i<nrows;i++) {
    row_map_inv[row_map[i]] = i;
  }

  nwarps = (nznrows + WARP_SIZE-1)/WARP_SIZE;

  std::vector<uint> A_w(nwarps,0); // max non-zeros per row
  std::vector<uint> nnz_imin(nwarps,nrows); // minimum non-zeros per row
  std::vector<uint> nnz_imax(nwarps,0); // maximum non-zeros per row

  // Use sorted row-sizes to calculate A_w, nnz_w, etc.
  for (int w = 0; w < nwarps; w++) {
    for (int r = WARP_SIZE * w; r < nznrows && r < WARP_SIZE*(w+1); r++) {
      uint rowsize = nnz_r[r].second;
      if (rowsize < nnz_imin[w]) nnz_imin[w] = rowsize; // min
      if (rowsize > nnz_imax[w]) nnz_imax[w] = rowsize; // max
    }

    A_w[w] = nnz_imax[w];
  }
  max_nz_rows = A_w;
    
  // set warp_offsets and allocate_nz;
  warp_offsets.resize(nwarps+1);
  warp_offsets[0] = 0;
  for(int w = 0; w < nwarps; w++) {
    assert(A_w[w] > 0);
    warp_offsets[w+1] = warp_offsets[w] + A_w[w] * WARP_SIZE;
  }
  allocate_nz = warp_offsets[nwarps];

  // Generate reordering map for future use
  reorder_rows.resize(nz);
  for (int w_s = 0; w_s < nwarps; w_s++) {
    for (int r_s = WARP_SIZE * w_s; r_s < nrows 
	   && r_s < WARP_SIZE * (w_s+1); r_s++) {
      int r = nnz_r[r_s].first;
      int rowsize = nnz_r[r_s].second;
      for(int i = 0; i < rowsize; i++) {
	reorder_rows[rows[r] + i] =
	  warp_offsets[w_s] + (r_s % WARP_SIZE) * max_nz_rows[w_s] + i; // undid reodering
      }
    }
  }
}


/********

	 Purpose of this executable is to examine different effects of optimization

*******/

#define ValueType double
#define IndexType int

int main(int argc, char *argv[]) {

  bool cache = true;
  int warps_per_block = 1; 
  std::string matrixfilename = argv[1];
  int ntests = 1;
  if (argc >2 ) ntests = atoi(argv[2]);
  if (argc >3) cache = (1==atoi(argv[3]));
  if (argc >4) warps_per_block = atoi(argv[4]);

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
 
  std::string datapath = "./data/" + matrixname + "_optimize_" + (cache ? "cache" : "nocache" ) + ".txt";
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
    cusp::array1d<ValueType, DeviceSpace> dy = y;

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
    warpkernel::addData(datafile, "cusp-hyb", boost::accumulators::mean(statstime), -1, -1, -1, -1);

  }


  // warp kernel tests
  {
    cusp::array1d<ValueType, DeviceSpace> dx = x;
    //      kernel1.scan(nz, N, A);

    // cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
    // kernel1.reorder_columns_coalesced(reorder_cols);

    // cusp::array1d<ValueType, DeviceSpace> dreordered_x;
    // kernel1.reorder_x(x, dreordered_x);

    warpkernel::structure kernel1(N, nz, 0);



    cusp::array1d<IndexType, DeviceSpace> restore_col;
   
    std::cout << "warp kernel 1" << std::endl;

    std::cout << std::endl;
    
    // normal kernel
    {

      scan<true>(kernel1.nz, kernel1.nrows,
		 &(A.values[0]), &(A.row_offsets[0]), &(A.column_indices[0]), 
		 kernel1.nwarps, kernel1.nznrows, kernel1.allocate_nz,
		 kernel1.reorder_rows, 
		 kernel1.warp_offsets,
		 kernel1.max_nz_rows,
		 kernel1.row_map,
		 kernel1.row_map_inv);


      uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
      uint blocksize = warps_per_block * WARP_SIZE;

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng(kernel1,
									  &(A.values[0]),
									  &(A.column_indices[0]));


      restore_col = eng.device_colinds;


      cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
      ValueType measuretime = 0;
      for (int t = 0; t < ntests; t++) {
	if (cache == true) {
	  measuretime += eng.run<true>(nblocks, blocksize, thrust::raw_pointer_cast(&dx[0]), thrust::raw_pointer_cast(&dy[0]));
	} else {
	  measuretime += eng.run<false>(nblocks, blocksize, thrust::raw_pointer_cast(&dx[0]), thrust::raw_pointer_cast(&dy[0]));
	}
      }
      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify(y,ycheck,N)) {
	std::cout << "warpkernel (" << nblocks << "," << blocksize <<") time = " 
		  << std::scientific << measuretime/ntests << std::endl;
	warpkernel::addData(datafile, "warpkernel", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed" << std::endl;
    }
    //

    // not coalesced
    {
      scan_nocoalesced<true>(kernel1.nz, kernel1.nrows,
	 &(A.values[0]), &(A.row_offsets[0]), &(A.column_indices[0]), 
	 kernel1.nwarps, kernel1.nznrows, kernel1.allocate_nz,
	 kernel1.reorder_rows, 
	 kernel1.warp_offsets,
	 kernel1.max_nz_rows,
	 kernel1.row_map,
	 kernel1.row_map_inv);

      uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
      uint blocksize = warps_per_block * WARP_SIZE;

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng(kernel1,
									  &(A.values[0]),
									  &(A.column_indices[0]));

      cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
      ValueType measuretime = 0;
      if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));
      for (int i=0; i < ntests; i++) {

	if (cache == true) {
	  cusp::detail::timer t;	t.start();
	  warpKernel_nocoalesced<true> <<< nblocks, blocksize >>>
	    (kernel1.nrows,
	     thrust::raw_pointer_cast(&eng.device_values[0]),
	     thrust::raw_pointer_cast(&eng.device_colinds[0]),
	     thrust::raw_pointer_cast(&eng.device_row_map[0]),
	     thrust::raw_pointer_cast(&eng.device_max_nz_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_warp_offsets[0]),
	     thrust::raw_pointer_cast(&dx[0]),
	     thrust::raw_pointer_cast(&dy[0]));
	  measuretime += t.seconds_elapsed();
	} else {
	  cusp::detail::timer t;	t.start();
	  warpKernel_nocoalesced<false> <<< nblocks, blocksize >>>
	    (kernel1.nrows,
	     thrust::raw_pointer_cast(&eng.device_values[0]),
	     thrust::raw_pointer_cast(&eng.device_colinds[0]),
	     thrust::raw_pointer_cast(&eng.device_row_map[0]),
	     thrust::raw_pointer_cast(&eng.device_max_nz_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_warp_offsets[0]),
	     thrust::raw_pointer_cast(&dx[0]),
	     thrust::raw_pointer_cast(&dy[0]));
	  measuretime += t.seconds_elapsed();
	}  
      }
      
      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify(y,ycheck,N)) {
	std::cout << "warpkernel nocoalesced(" << nblocks << "," << blocksize <<") time = " 
		  << std::scientific << measuretime/ntests << std::endl;
	warpkernel::addData(datafile, "warpkernel_nocoalesced", measuretime/ntests, kernel1, blocksize);
      } else std::cout << "Failed nocoalesced" << std::endl;


    }
    //

    // no sorting
    {

      scan<false>(kernel1.nz, kernel1.nrows,
		 &(A.values[0]), &(A.row_offsets[0]), &(A.column_indices[0]), 
		 kernel1.nwarps, kernel1.nznrows, kernel1.allocate_nz,
		 kernel1.reorder_rows, 
		 kernel1.warp_offsets,
		 kernel1.max_nz_rows,
		 kernel1.row_map,
		 kernel1.row_map_inv);


      uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
      uint blocksize = warps_per_block * WARP_SIZE;

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng(kernel1,
									  &(A.values[0]),
									  &(A.column_indices[0]));


      restore_col = eng.device_colinds;


      cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
      ValueType measuretime = 0;
      for (int t = 0; t < ntests; t++) {
	if (cache == true) {
	  measuretime += eng.run<true>(nblocks, blocksize, thrust::raw_pointer_cast(&dx[0]), thrust::raw_pointer_cast(&dy[0]));
	} else {
	  measuretime += eng.run<false>(nblocks, blocksize, thrust::raw_pointer_cast(&dx[0]), thrust::raw_pointer_cast(&dy[0]));
	}
      }
      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify(y,ycheck,N)) {
	std::cout << "warpkernel (" << nblocks << "," << blocksize <<") time = " 
		  << std::scientific << measuretime/ntests << std::endl;
	warpkernel::addData(datafile, "warpkernel", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed" << std::endl;
    }
    //

    // normal kernel with reordered x and columninds
    {
      kernel1.scan(nz, N, A);

      uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
      uint blocksize = warps_per_block * WARP_SIZE;

      cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
      {
  	cusp::detail::timer reordercolstimer; reordercolstimer.start();
  	kernel1.reorder_columns_coalesced(reorder_cols);
  	ValueType reordertime = reordercolstimer.seconds_elapsed();
  	std::cout << "reorder column time " << reordertime << std::endl;
  	warpkernel::addData(datafile, "reorder_col_time_cache_rex", reordertime, -1, -1, -1, -1);
      }
      
      cusp::array1d<ValueType, DeviceSpace> dreordered_x;
      {
  	cusp::detail::timer reorderxtimer; reorderxtimer.start();
  	kernel1.reorder_x(x, dreordered_x);

  	ValueType reordertime = reorderxtimer.seconds_elapsed();
  	std::cout << "reorder x time " << reordertime << std::endl;
  	warpkernel::addData(datafile, "reorder_x_time_cache_rex", reordertime, -1, -1, -1, -1);
      
      }      

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng(kernel1,
									  &(A.values[0]),
									  &(A.column_indices[0]));

      eng.device_colinds = reorder_cols;

      cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
      ValueType measuretime = 0;
      for (int t = 0; t < ntests; t++) {
	if (cache == true) {
	  measuretime += eng.run_x<true>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), thrust::raw_pointer_cast(&dy[0]));
	} else {
	  measuretime += eng.run_x<false>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), thrust::raw_pointer_cast(&dy[0]));
	}
      }
      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify_x(y,ycheck,N,&kernel1.row_map[0])) {
	std::cout << "warpkernel_rex(" << nblocks << "," << blocksize <<") time = " 
		  << std::scientific << measuretime/ntests << std::endl;
	warpkernel::addData(datafile, "warpkernel_rex", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel_rex" << std::endl;

      eng.device_colinds = restore_col;
    }
    //

    // normal kernel with reordered x and columninds with rowmapping
    {
      kernel1.scan(nz, N, A);

      uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
      uint blocksize = warps_per_block * WARP_SIZE;

      cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
      {
  	cusp::detail::timer reordercolstimer; reordercolstimer.start();
  	kernel1.reorder_columns_coalesced(reorder_cols);
  	ValueType reordertime = reordercolstimer.seconds_elapsed();
  	std::cout << "reorder column time " << reordertime << std::endl;
  	warpkernel::addData(datafile, "reorder_col_time_cache_rex", reordertime, -1, -1, -1, -1);
      }
      
      cusp::array1d<ValueType, DeviceSpace> dreordered_x;
      {
  	cusp::detail::timer reorderxtimer; reorderxtimer.start();
  	kernel1.reorder_x(x, dreordered_x);

  	ValueType reordertime = reorderxtimer.seconds_elapsed();
  	std::cout << "reorder x time " << reordertime << std::endl;
  	warpkernel::addData(datafile, "reorder_x_time_cache_rex", reordertime, -1, -1, -1, -1);
      
      }      

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng(kernel1,
									  &(A.values[0]),
									  &(A.column_indices[0]));

      eng.device_colinds = reorder_cols;

      cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
      ValueType measuretime = 0;
      for (int t = 0; t < ntests; t++) {
	if (cache == true) {
	  measuretime += eng.run<true>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), thrust::raw_pointer_cast(&dy[0]));
	} else {
	  measuretime += eng.run<false>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), thrust::raw_pointer_cast(&dy[0]));
	}
      }
      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify(y,ycheck,N)) {
	std::cout << "warpkernel_rex_rowmap(" << nblocks << "," << blocksize <<") time = " 
		  << std::scientific << measuretime/ntests << std::endl;
	warpkernel::addData(datafile, "warpkernel_rex_rowmap", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel_rex_rowmap" << std::endl;

      eng.device_colinds = restore_col;
    }
    //

    // normal kernel with reordered x and columninds and A.values
    {
      kernel1.scan(nz, N, A);

      uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
      uint blocksize = warps_per_block * WARP_SIZE;

      cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
      {
  	cusp::detail::timer reordercolstimer; reordercolstimer.start();
  	kernel1.reorder_columns_rowsort(reorder_cols, A.row_offsets);
  	ValueType reordertime = reordercolstimer.seconds_elapsed();
  	std::cout << "reorder column time " << reordertime << std::endl;
  	warpkernel::addData(datafile, "reorder_col_time_nocache_rex", reordertime, -1, -1, -1, -1);
      }
      
      cusp::array1d<ValueType, DeviceSpace> dreordered_x;
      {
  	cusp::detail::timer reorderxtimer; reorderxtimer.start();
  	kernel1.reorder_x(x, dreordered_x);

  	ValueType reordertime = reorderxtimer.seconds_elapsed();
  	std::cout << "reorder x time " << reordertime << std::endl;
  	warpkernel::addData(datafile, "reorder_x_time_nocache_rex", reordertime, -1, -1, -1, -1);
      
      }

      cusp::array1d<ValueType, CPUSpace> A_new_values(nz);
      {
  	cusp::detail::timer reorderAtimer; reorderAtimer.start();
      for(int i=0;i< nz; i++) {
  	A_new_values[i] = A.values[kernel1.reorder_A_rows[i]];
      }
      ValueType reordertime = reorderAtimer.seconds_elapsed();
  	std::cout << "reorder A time " << reordertime << std::endl;
  	warpkernel::addData(datafile, "reorder_A_time_nocache_rex", reordertime, -1, -1, -1, -1);
      
      }      

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng(kernel1,
									  &(A_new_values[0]),
									  &(reorder_cols[0]));

      cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
      ValueType measuretime = 0;
      for (int t = 0; t < ntests; t++) {
	if (cache == true) {
	  measuretime += eng.run_x<true>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), thrust::raw_pointer_cast(&dy[0]));
	} else {
	  measuretime += eng.run_x<false>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), thrust::raw_pointer_cast(&dy[0]));
	}
      }
      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify_x(y,ycheck,N,&kernel1.row_map[0])) {
	std::cout << "warpkernel_rex_rowsort(" << nblocks << "," << blocksize <<") time = " 
		  << std::scientific << measuretime/ntests << std::endl;
	warpkernel::addData(datafile, "warpkernel_rex_rowsort", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel_rex_rowsort" << std::endl;

    }
    //


  }
}
