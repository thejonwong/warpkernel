// General
#include <iostream>
#include <algorithm>
#include <sstream>
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


#define DeviceSpace cusp::device_memory
#define CPUSpace cusp::host_memory

struct rand_float {
  double operator() ()
  {
    return 1.1;//((double)(rand() % 100))/100. -0.3;
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

/**********/

template <bool usecache, typename ValueType, typename IndexType>
__global__ void warpKernel2_noregister(IndexType nrows, int nwarps, ValueType* A, IndexType *colinds, 
			    IndexType *rowmap, uint* maxrows, IndexType *warp_offset,
			    uint* reduction, uint* rows_offset_warp,
			    ValueType* x , ValueType* y) {

  const uint tid = threadIdx.x;
  const uint id = tid  + blockIdx.x * blockDim.x;
  const uint wid = tid & (WARP_SIZE-1);
  const uint warpid = id / WARP_SIZE;

  extern volatile __shared__ ValueType sumvalues[];

  if (warpid >= nwarps) return;
  const uint offsets = reduction[warpid];
  const uint row_start = rows_offset_warp[warpid];  
  const uint rowid = row_start + wid/offsets;

  if (rowid < nrows) {

    IndexType toffset = warp_offset[warpid] + wid;
    const uint maxnz = maxrows[warpid] * WARP_SIZE + toffset;
    ValueType sum = A[toffset] * fetch_cache<usecache> (colinds[toffset],x);

    for(toffset += WARP_SIZE; toffset<maxnz; toffset += WARP_SIZE) {
      sum += A[toffset] * fetch_cache<usecache> (colinds[toffset],x);
    }
    sumvalues[tid] = sum;

    // // possible reduction
    for (int i = 1; i< offsets; i <<= 1) {
      if (offsets > i ) {
	sumvalues[tid] += sumvalues[tid+i];
      }
    }

    if ((wid & (offsets-1)) == 0) {
      y[rowmap[rowid]] = sumvalues[tid]; 
    }
  }
}


/**********/


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

/*********/

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
void sort(ValueType *rows, std::vector<std::pair<uint,uint> > & nnz_r, uint & nwarps, uint nrows) {
    nwarps = (nrows + WARP_SIZE-1)/(WARP_SIZE);
    nnz_r.resize(nrows);
    uint nznrows = 0;
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

template <typename ValueType, typename IndexType>
void scan(uint & nz, uint & nrows, 
	  ValueType * A, IndexType * rows, IndexType *colinds, 
	  uint & nwarps, uint & allocate_nz,
	  std::vector<uint> & reorder_rows, // new_values[reorder_rows[i]] = A[i]
	  std::vector<int> & warp_offsets,
	  std::vector<uint> & max_nz_rows,
	  std::vector<int> &row_map_,
	  std::vector<int> &row_map_inv_, uint &nznrows) {

  std::vector<std::pair<uint,uint> > nnz_r; // non-zeros per row
  sort(rows, nnz_r, nwarps, nrows);

  std::vector<int> row_map(nrows);
  for(int r = 0; r < nrows; r++)
    row_map[r] = nnz_r[r].first;
  row_map_ = row_map;

  std::vector<int> row_map_inv(nrows);
  for(int i=0;i<nrows;i++) {
    row_map_inv[row_map[i]] = i;
  }
  row_map_inv_ = row_map_inv;

  std::vector<uint> A_w(nwarps); // max non-zeros per row
  std::vector<uint> nnz_imin(nwarps,nrows); // minimum non-zeros per row
  std::vector<uint> nnz_imax(nwarps); // maximum non-zeros per row

  // Use sorted row-sizes to calculate A_w, nnz_w, etc.
  for (int w = 0; w < nwarps; w++) {
    for (int r = WARP_SIZE * w; r < nrows && r < WARP_SIZE*(w+1); r++) {
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
	  warp_offsets[w_s] + (r_s % WARP_SIZE) + i*WARP_SIZE;
      }
    }
  }
}

/******* scan no reorder coalesce ****/

template <typename ValueType, typename IndexType>
void scan_nocoalesced(uint & nz, uint & nrows, 
		      ValueType * A, IndexType * rows, IndexType *colinds, 
		      uint & nwarps, uint & allocate_nz,
		      std::vector<uint> & reorder_rows, // new_values[reorder_rows[i]] = A[i]
		      std::vector<int> & warp_offsets,
		      std::vector<uint> & max_nz_rows,
		      std::vector<int> &row_map_,
		      std::vector<int> &row_map_inv_) {

  std::vector<std::pair<uint,uint> > nnz_r; // non-zeros per row
  sort(rows, nnz_r, nwarps, nrows);

  std::vector<int> row_map(nrows);
  for(int r = 0; r < nrows; r++)
    row_map[r] = nnz_r[r].first;
  row_map_ = row_map;

  std::vector<int> row_map_inv(nrows);
  for(int i=0;i<nrows;i++) {
    row_map_inv[row_map[i]] = i;
  }
  row_map_inv_ = row_map_inv;

  std::vector<uint> A_w(nwarps); // max non-zeros per row
  std::vector<uint> nnz_imin(nwarps,nrows); // minimum non-zeros per row
  std::vector<uint> nnz_imax(nwarps); // maximum non-zeros per row

  // Use sorted row-sizes to calculate A_w, nnz_w, etc.
  for (int w = 0; w < nwarps; w++) {
    for (int r = WARP_SIZE * w; r < nrows && r < WARP_SIZE*(w+1); r++) {
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

/******* scan no reorder coalesce no sort ****/

template <typename ValueType, typename IndexType>
void scan_nocoalesced_nosort(uint & nz, uint & nrows, 
			     ValueType * A, IndexType * rows, IndexType *colinds, 
			     uint & nwarps, uint & allocate_nz,
			     std::vector<uint> & reorder_rows, // new_values[reorder_rows[i]] = A[i]
			     std::vector<int> & warp_offsets,
			     std::vector<uint> & max_nz_rows,
			     std::vector<int> &row_map_,
			     std::vector<int> &row_map_inv_) {

  std::vector<std::pair<uint,uint> > nnz_r; // non-zeros per row
  nwarps = (nrows + WARP_SIZE-1)/(WARP_SIZE);
    
  // Re-arrange rows to reach our assumptions
  for (int w = 0; w < nwarps; w++) {
    for (int r = WARP_SIZE * w; r < nrows && r < WARP_SIZE*(w+1); r++) {
      uint rowsize = rows[r+1] - rows[r];
      if (rowsize > 0)
	nnz_r.push_back(std::make_pair(r,rowsize));
    }
  }

  std::vector<int> row_map(nrows);
  for(int r = 0; r < nrows; r++)
    row_map[r] = nnz_r[r].first;
  row_map_ = row_map;

  std::vector<int> row_map_inv(nrows);
  for(int i=0;i<nrows;i++) {
    row_map_inv[row_map[i]] = i;
  }
  row_map_inv_ = row_map_inv;

  std::vector<uint> A_w(nwarps); // max non-zeros per row
  std::vector<uint> nnz_imin(nwarps,nrows); // minimum non-zeros per row
  std::vector<uint> nnz_imax(nwarps); // maximum non-zeros per row

  // Use sorted row-sizes to calculate A_w, nnz_w, etc.
  for (int w = 0; w < nwarps; w++) {
    for (int r = WARP_SIZE * w; r < nrows && r < WARP_SIZE*(w+1); r++) {
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

/******* scan no sort ****/

template <typename ValueType, typename IndexType>
void scan_nosort(uint & nz, uint & nrows, 
		 ValueType * A, IndexType * rows, IndexType *colinds, 
		 uint & nwarps, uint & allocate_nz,
		 std::vector<uint> & reorder_rows, // new_values[reorder_rows[i]] = A[i]
		 std::vector<int> & warp_offsets,
		 std::vector<uint> & max_nz_rows,
		 std::vector<int> &row_map_,
		 std::vector<int> &row_map_inv_) {

  std::vector<std::pair<uint,uint> > nnz_r; // non-zeros per row
  nwarps = (nrows + WARP_SIZE-1)/(WARP_SIZE);
    
  // Re-arrange rows to reach our assumptions
  for (int w = 0; w < nwarps; w++) {
    for (int r = WARP_SIZE * w; r < nrows && r < WARP_SIZE*(w+1); r++) {
      uint rowsize = rows[r+1] - rows[r];
      if (rowsize > 0)
	nnz_r.push_back(std::make_pair(r,rowsize));
    }
  }

  std::vector<int> row_map(nrows);
  for(int r = 0; r < nrows; r++)
    row_map[r] = nnz_r[r].first;
  row_map_ = row_map;

  std::vector<int> row_map_inv(nrows);
  for(int i=0;i<nrows;i++) {
    row_map_inv[row_map[i]] = i;
  }
  row_map_inv_ = row_map_inv;

  std::vector<uint> A_w(nwarps); // max non-zeros per row
  std::vector<uint> nnz_imin(nwarps,nrows); // minimum non-zeros per row
  std::vector<uint> nnz_imax(nwarps); // maximum non-zeros per row

  // Use sorted row-sizes to calculate A_w, nnz_w, etc.
  for (int w = 0; w < nwarps; w++) {
    for (int r = WARP_SIZE * w; r < nrows && r < WARP_SIZE*(w+1); r++) {
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
	  warp_offsets[w_s] + (r_s % WARP_SIZE) + i*WARP_SIZE;
      }
    }
  }
}


/********* Reorder */ 

template<typename IndexType, typename ValueType>
void reorder(ValueType *A, IndexType * colinds,
	     cusp::array1d<ValueType, DeviceSpace> &device_values, // allocate nz
	     cusp::array1d<IndexType, DeviceSpace> &device_colinds, // allocate nz
	     uint nz, uint allocate_nz,
	     std::vector<uint> reorder_rows)
{
  cusp::array1d<ValueType, CPUSpace> new_values(allocate_nz,0);
  cusp::array1d<IndexType, CPUSpace> new_colinds(allocate_nz,0);
    
  device_values.resize(allocate_nz);
  device_colinds.resize(allocate_nz);

  for(int i=0; i< nz; i++) {
    new_values[reorder_rows[i]] = A[i];
    new_colinds[reorder_rows[i]] = colinds[i];
  }

  device_values = new_values;
  device_colinds = new_colinds;
}


/* warpkernel1 process */

// preform the reordering
template<typename IndexType, typename ValueType>
void process(ValueType *A, IndexType * colinds,
	     cusp::array1d<ValueType, DeviceSpace> &device_values, // nz
	     cusp::array1d<IndexType, DeviceSpace> &device_colinds, // nz
	     cusp::array1d<IndexType, DeviceSpace> &device_row_map, // nrows
	     cusp::array1d<uint, DeviceSpace> &device_max_nz_per_row, // nwarps
	     cusp::array1d<IndexType, DeviceSpace> &device_warp_offsets, // nwarps
	     uint nz, uint allocate_nz,
	     std::vector<uint> reorder_rows,
	     std::vector<int> warp_offsets,
	     std::vector<uint> max_nz_rows,
	     std::vector<int> row_map)
{

  reorder(A,colinds, device_values, device_colinds,
	  nz, allocate_nz, reorder_rows);

  device_row_map = row_map;
  device_max_nz_per_row = max_nz_rows;
  device_warp_offsets = warp_offsets;

}



/********

	 Purpose of this executable is to examine different effects of optimization

*******/
#define ValueType double
#define IndexType int

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
 
  std::string datapath = "./data/" + matrixname + "_optimize.txt";
  std::cout << "Starting data file = " << datapath << std::endl;
  std::ofstream datafile(datapath.c_str());
  warpkernel::startDatafile(datafile, N, nz,ntests);

  cusp::array1d<ValueType, CPUSpace> x(N, 1.0);
  //  thrust::generate(x.begin(),x.end(), rand_float());

  cusp::array1d<ValueType, CPUSpace> y(N);

  {
    boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;
    // cusp hyb multiplication
    for (int i=0;i<ntests;i++)
      {

    
	cusp::hyb_matrix<IndexType, ValueType, DeviceSpace> A1 = A;
	cusp::array1d<ValueType, DeviceSpace> dx = x;
	cusp::array1d<ValueType, DeviceSpace> dy = y;

	cusp::detail::timer cusptimer;
	cusptimer.start();
	cusp::multiply(A1,dx,dy);
	ValueType measuredtime = cusptimer.seconds_elapsed();
	statstime(measuredtime);				
	y = dy;
	
      }

    std::cout << "cusp gpu time " 
	      << std::scientific << boost::accumulators::mean(statstime) << std::endl;
    warpkernel::addData(datafile, "cusp-csr", boost::accumulators::mean(statstime), -1, -1, -1, -1);

  }

  //  setup for warpkernel1
  cusp::array1d<ValueType, DeviceSpace> dx = x;

  warpkernel::structure kernel1( N, nz, 0);

  std::vector<uint> reorder_rows;  
  std::vector<int> warp_offsets;
  std::vector<uint> max_nz_rows;
  std::vector<int> row_map;
  std::vector<int> row_map_inv;

  uint nznrows;

  scan(kernel1.nz, kernel1.nrows, 
       &(A.values[0]), &(A.row_offsets[0]), &(A.column_indices[0]), 
       kernel1.nwarps, kernel1.allocate_nz,
       reorder_rows, 
       warp_offsets,
       max_nz_rows,
       row_map,
       row_map_inv, nznrows);

  uint warps_per_block = 6;
  uint nblocks = (kernel1.nwarps + warps_per_block -1)/warps_per_block;
  uint blocksize = warps_per_block * WARP_SIZE;

  // original warpkernel1 with cache
  {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
    // allocate arrays
    cusp::array1d<ValueType, DeviceSpace> device_values; // nz
    cusp::array1d<IndexType, DeviceSpace> device_colinds; // nz
    cusp::array1d<IndexType, DeviceSpace> device_row_map; // nrows
    cusp::array1d<uint, DeviceSpace> device_max_nz_per_row; // nwarps
    cusp::array1d<IndexType, DeviceSpace> device_warp_offsets; // nwarps
    cusp::array1d<uint, DeviceSpace> device_threads_per_row; // offsets - nwarps
    cusp::array1d<uint, DeviceSpace> device_row_offset_warp; // rows - nwarps

    process(&(A.values[0]), &(A.column_indices[0]),
  	    device_values, device_colinds,
  	    device_row_map, device_max_nz_per_row,
  	    device_warp_offsets,
  	    kernel1.nz, kernel1.allocate_nz,
  	    reorder_rows,
  	    warp_offsets,
  	    max_nz_rows,
  	    row_map);

    const bool cache = true; 
    boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

    if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));
    for (int i=0; i < ntests; i++) {

      cusp::detail::timer t;
      t.start();
      warpkernel::warpKernel<cache> <<< nblocks, blocksize >>>
  	(kernel1.nrows,
  	 thrust::raw_pointer_cast(&device_values[0]),
  	 thrust::raw_pointer_cast(&device_colinds[0]),
  	 thrust::raw_pointer_cast(&device_row_map[0]),
  	 thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
  	 thrust::raw_pointer_cast(&device_warp_offsets[0]),
  	 thrust::raw_pointer_cast(&dx[0]),
  	 thrust::raw_pointer_cast(&dy[0]));
      ValueType measuretime = t.seconds_elapsed();
      statstime(measuretime);
    }

    cusp::array1d<ValueType, CPUSpace> ycheck = dy;  

    if (verify(y,ycheck,N)) {
      std::cout << "current warpkernel_cache time =" << boost::accumulators::mean(statstime) << std::endl;
      warpkernel::addData(datafile, "warpkernel_cache", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
    } else std::cout << "failed to verify warp_kernel_cache" << std::endl;
  }

  //  warpkernel.hpp
  { 
    cusp::array1d<ValueType, DeviceSpace> dx = x;
    warpkernel::structure kernel1;
    kernel1.scan(nz, N, A);

    uint warps_per_block = 6;
    uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
    uint blocksize = warps_per_block * WARP_SIZE;

    warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng(kernel1,
									&(A.values[0]),
									&(A.column_indices[0]));
    
    cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
    
    ValueType measuretime = 0;
    for (int t = 0; t < ntests; t++) {
      measuretime += eng.run<true>(nblocks, blocksize,
				   thrust::raw_pointer_cast(&dx[0]),
				   thrust::raw_pointer_cast(&dy[0]));
    }
    cusp::array1d<ValueType, CPUSpace> ycheck = dy;
    if (eng.verify(y,ycheck))
      {
	std::cout << "warpkernel (" << nblocks << "," << blocksize <<") time = " 
		  << std::scientific << measuretime/ntests << std::endl;
	    } else {
      std::cout << "Failed original warpkernel.hpp" << std::endl;
    }

  }

  // original warpkernel1 no cache
  {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
    // allocate arrays
    cusp::array1d<ValueType, DeviceSpace> device_values; // nz
    cusp::array1d<IndexType, DeviceSpace> device_colinds; // nz
    cusp::array1d<IndexType, DeviceSpace> device_row_map; // nrows
    cusp::array1d<uint, DeviceSpace> device_max_nz_per_row; // nwarps
    cusp::array1d<IndexType, DeviceSpace> device_warp_offsets; // nwarps
    cusp::array1d<uint, DeviceSpace> device_threads_per_row; // offsets - nwarps
    cusp::array1d<uint, DeviceSpace> device_row_offset_warp; // rows - nwarps

    process(&(A.values[0]), &(A.column_indices[0]),
  	    device_values, device_colinds,
  	    device_row_map, device_max_nz_per_row,
  	    device_warp_offsets,
  	    kernel1.nz, kernel1.allocate_nz,
  	    reorder_rows,
  	    warp_offsets,
  	    max_nz_rows,
  	    row_map);

    const bool cache = false; 
    boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

    if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));
    for (int i=0; i < ntests; i++) {

      cusp::detail::timer t;
      t.start();
      warpkernel::warpKernel<cache> <<< nblocks, blocksize >>>
  	(kernel1.nrows,
  	 thrust::raw_pointer_cast(&device_values[0]),
  	 thrust::raw_pointer_cast(&device_colinds[0]),
  	 thrust::raw_pointer_cast(&device_row_map[0]),
  	 thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
  	 thrust::raw_pointer_cast(&device_warp_offsets[0]),
  	 thrust::raw_pointer_cast(&dx[0]),
  	 thrust::raw_pointer_cast(&dy[0]));
      ValueType measuretime = t.seconds_elapsed();
      statstime(measuretime);
    }

    cusp::array1d<ValueType, CPUSpace> ycheck = dy;  

    if (verify(y,ycheck,N)) {
      std::cout << "current warpkernel_nocache time =" << boost::accumulators::mean(statstime) << std::endl;
      warpkernel::addData(datafile, "warpkernel_nocache", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
    } else std::cout << "failed to verify warp_kernel_nocache" << std::endl;
  }

  // original warpkernel1 no reordering with cache
  {

    // allocate arrays
    cusp::array1d<ValueType, DeviceSpace> device_values; // nz
    cusp::array1d<IndexType, DeviceSpace> device_colinds; // nz
    cusp::array1d<IndexType, DeviceSpace> device_row_map; // nrows
    cusp::array1d<uint, DeviceSpace> device_max_nz_per_row; // nwarps
    cusp::array1d<IndexType, DeviceSpace> device_warp_offsets; // nwarps
    cusp::array1d<uint, DeviceSpace> device_threads_per_row; // offsets - nwarps
    cusp::array1d<uint, DeviceSpace> device_row_offset_warp; // rows - nwarps

    scan_nocoalesced(kernel1.nz, kernel1.nrows, 
  		     &(A.values[0]), &(A.row_offsets[0]), &(A.column_indices[0]), 
  		     kernel1.nwarps, kernel1.allocate_nz,
  		     reorder_rows, 
  		     warp_offsets,
  		     max_nz_rows,
  		     row_map,
  		     row_map_inv);

    process(&(A.values[0]), &(A.column_indices[0]),
  	    device_values, device_colinds,
  	    device_row_map, device_max_nz_per_row,
  	    device_warp_offsets,
  	    kernel1.nz, kernel1.allocate_nz,
  	    reorder_rows,
  	    warp_offsets,
  	    max_nz_rows,
  	    row_map);

    // original warpkernel1 not coalesced with cache
    {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
      const bool cache = true; 
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));
      for (int i=0; i < ntests; i++) {

  	cusp::detail::timer t;
  	t.start();
  	warpKernel_nocoalesced<cache> <<< nblocks, blocksize >>>
  	  (kernel1.nrows,
  	   thrust::raw_pointer_cast(&device_values[0]),
  	   thrust::raw_pointer_cast(&device_colinds[0]),
  	   thrust::raw_pointer_cast(&device_row_map[0]),
  	   thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
  	   thrust::raw_pointer_cast(&device_warp_offsets[0]),
  	   thrust::raw_pointer_cast(&dx[0]),
  	   thrust::raw_pointer_cast(&dy[0]));
  	ValueType measuretime = t.seconds_elapsed();
  	statstime(measuretime);
      }

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;  

      if (verify(y,ycheck,N)) {
  	std::cout << "current warpkernel_cache_nocoalesced time =" << boost::accumulators::mean(statstime) << std::endl;
  	warpkernel::addData(datafile, "warpkernel_cache_nocoalesced", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
      } else std::cout << "failed to verify warp_kernel_cache_nocoalesced" << std::endl;
    }

    // original warpkernel1 not coalesced with no cache
    {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
      const bool cache = false; 
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));
      for (int i=0; i < ntests; i++) {

  	cusp::detail::timer t;
  	t.start();
  	warpKernel_nocoalesced<cache> <<< nblocks, blocksize >>>
  	  (kernel1.nrows,
  	   thrust::raw_pointer_cast(&device_values[0]),
  	   thrust::raw_pointer_cast(&device_colinds[0]),
  	   thrust::raw_pointer_cast(&device_row_map[0]),
  	   thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
  	   thrust::raw_pointer_cast(&device_warp_offsets[0]),
  	   thrust::raw_pointer_cast(&dx[0]),
  	   thrust::raw_pointer_cast(&dy[0]));
  	ValueType measuretime = t.seconds_elapsed();
  	statstime(measuretime);
      }

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;  

      if (verify(y,ycheck,N)) {
  	std::cout << "current warpkernel_nocache_nocoalesced time =" << boost::accumulators::mean(statstime) << std::endl;
  	warpkernel::addData(datafile, "warpkernel_nocache_nocoalesced", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
      } else std::cout << "failed to verify warp_kernel_nocahce_nocoalesced" << std::endl;
    }

  }

  // original warpkernel1 no reordering with cache no sorting
  {

    // allocate arrays
    cusp::array1d<ValueType, DeviceSpace> device_values; // nz
    cusp::array1d<IndexType, DeviceSpace> device_colinds; // nz
    cusp::array1d<IndexType, DeviceSpace> device_row_map; // nrows
    cusp::array1d<uint, DeviceSpace> device_max_nz_per_row; // nwarps
    cusp::array1d<IndexType, DeviceSpace> device_warp_offsets; // nwarps
    cusp::array1d<uint, DeviceSpace> device_threads_per_row; // offsets - nwarps
    cusp::array1d<uint, DeviceSpace> device_row_offset_warp; // rows - nwarps

    scan_nocoalesced_nosort(kernel1.nz, kernel1.nrows, 
  			    &(A.values[0]), &(A.row_offsets[0]), &(A.column_indices[0]), 
  			    kernel1.nwarps, kernel1.allocate_nz,
  			    reorder_rows, 
  			    warp_offsets,
  			    max_nz_rows,
  			    row_map,
  			    row_map_inv);

    process(&(A.values[0]), &(A.column_indices[0]),
  	    device_values, device_colinds,
  	    device_row_map, device_max_nz_per_row,
  	    device_warp_offsets,
  	    kernel1.nz, kernel1.allocate_nz,
  	    reorder_rows,
  	    warp_offsets,
  	    max_nz_rows,
  	    row_map);

    // original warpkernel1 not coalesced with cache no sort
    {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
      const bool cache = true; 
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));
      for (int i=0; i < ntests; i++) {

  	cusp::detail::timer t;
  	t.start();
  	warpKernel_nocoalesced<cache> <<< nblocks, blocksize >>>
  	  (kernel1.nrows,
  	   thrust::raw_pointer_cast(&device_values[0]),
  	   thrust::raw_pointer_cast(&device_colinds[0]),
  	   thrust::raw_pointer_cast(&device_row_map[0]),
  	   thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
  	   thrust::raw_pointer_cast(&device_warp_offsets[0]),
  	   thrust::raw_pointer_cast(&dx[0]),
  	   thrust::raw_pointer_cast(&dy[0]));
  	ValueType measuretime = t.seconds_elapsed();
  	statstime(measuretime);
      }

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;  

      if (verify(y,ycheck,N)) {
  	std::cout << "current warpkernel_cache_nocoalesced_nosort time =" << boost::accumulators::mean(statstime) << std::endl;
  	warpkernel::addData(datafile, "warpkernel_cache_nocoalesced_nosort", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
      } else std::cout << "failed to verify warp_kernel_cache_nocoalesced_nosort" << std::endl;
    }

    // original warpkernel1 not coalesced with no cache no sort
    {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
      const bool cache = false; 
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));
      for (int i=0; i < ntests; i++) {

  	cusp::detail::timer t;
  	t.start();
  	warpKernel_nocoalesced<cache> <<< nblocks, blocksize >>>
  	  (kernel1.nrows,
  	   thrust::raw_pointer_cast(&device_values[0]),
  	   thrust::raw_pointer_cast(&device_colinds[0]),
  	   thrust::raw_pointer_cast(&device_row_map[0]),
  	   thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
  	   thrust::raw_pointer_cast(&device_warp_offsets[0]),
  	   thrust::raw_pointer_cast(&dx[0]),
  	   thrust::raw_pointer_cast(&dy[0]));
  	ValueType measuretime = t.seconds_elapsed();
  	statstime(measuretime);
      }

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;  

      if (verify(y,ycheck,N)) {
  	std::cout << "current warpkernel_nocache_nocoalesced_nosort time =" << boost::accumulators::mean(statstime) << std::endl;
  	warpkernel::addData(datafile, "warpkernel_nocache_nocoalesced_nosort", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
      } else std::cout << "failed to verify warp_kernel_cache_nocache_nocoalesced_nosort" << std::endl;
    }

    // original warpkernel1 no reordering with cache
    {

      // allocate arrays
      cusp::array1d<ValueType, DeviceSpace> device_values; // nz
      cusp::array1d<IndexType, DeviceSpace> device_colinds; // nz
      cusp::array1d<IndexType, DeviceSpace> device_row_map; // nrows
      cusp::array1d<uint, DeviceSpace> device_max_nz_per_row; // nwarps
      cusp::array1d<IndexType, DeviceSpace> device_warp_offsets; // nwarps
      cusp::array1d<uint, DeviceSpace> device_threads_per_row; // offsets - nwarps
      cusp::array1d<uint, DeviceSpace> device_row_offset_warp; // rows - nwarps

      scan_nosort(kernel1.nz, kernel1.nrows, 
  		  &(A.values[0]), &(A.row_offsets[0]), &(A.column_indices[0]), 
  		  kernel1.nwarps, kernel1.allocate_nz,
  		  reorder_rows, 
  		  warp_offsets,
  		  max_nz_rows,
  		  row_map,
  		  row_map_inv);

      process(&(A.values[0]), &(A.column_indices[0]),
  	      device_values, device_colinds,
  	      device_row_map, device_max_nz_per_row,
  	      device_warp_offsets,
  	      kernel1.nz, kernel1.allocate_nz,
  	      reorder_rows,
  	      warp_offsets,
  	      max_nz_rows,
  	      row_map);

      // original warpkernel1 not coalesced with cache
      {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
  	const bool cache = true; 
  	boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

  	if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));
  	for (int i=0; i < ntests; i++) {

  	  cusp::detail::timer t;
  	  t.start();
  	  warpkernel::warpKernel<cache> <<< nblocks, blocksize >>>
  	    (kernel1.nrows,
  	     thrust::raw_pointer_cast(&device_values[0]),
  	     thrust::raw_pointer_cast(&device_colinds[0]),
  	     thrust::raw_pointer_cast(&device_row_map[0]),
  	     thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
  	     thrust::raw_pointer_cast(&device_warp_offsets[0]),
  	     thrust::raw_pointer_cast(&dx[0]),
  	     thrust::raw_pointer_cast(&dy[0]));
  	  ValueType measuretime = t.seconds_elapsed();
  	  statstime(measuretime);
  	}

  	cusp::array1d<ValueType, CPUSpace> ycheck = dy;  

  	if (verify(y,ycheck,N)) {
  	  std::cout << "current warpkernel_cache_nosort time =" << boost::accumulators::mean(statstime) << std::endl;
  	  warpkernel::addData(datafile, "warpkernel_cache_nosort", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
  	} else std::cout << "failed to verify warp_kernel_cache_nosort" << std::endl;
      }

      // original warpkernel1 not coalesced with no cache
      {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
  	const bool cache = false; 
  	boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

  	if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));
  	for (int i=0; i < ntests; i++) {

  	  cusp::detail::timer t;
  	  t.start();
  	  warpkernel::warpKernel<cache> <<< nblocks, blocksize >>>
  	    (kernel1.nrows,
  	     thrust::raw_pointer_cast(&device_values[0]),
  	     thrust::raw_pointer_cast(&device_colinds[0]),
  	     thrust::raw_pointer_cast(&device_row_map[0]),
  	     thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
  	     thrust::raw_pointer_cast(&device_warp_offsets[0]),
  	     thrust::raw_pointer_cast(&dx[0]),
  	     thrust::raw_pointer_cast(&dy[0]));
  	  ValueType measuretime = t.seconds_elapsed();
  	  statstime(measuretime);
  	}

  	cusp::array1d<ValueType, CPUSpace> ycheck = dy;  

  	if (verify(y,ycheck,N)) {
  	  std::cout << "current warpkernel_nocache_nosort time =" << boost::accumulators::mean(statstime) << std::endl;
  	  warpkernel::addData(datafile, "warpkernel_nocache_nosort", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
  	} else std::cout << "failed to verify warp_kernel_nocache_nosort" << std::endl;
      }

    }

    // warpkernel1 with reorderedx with cache
    {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
      kernel1.scan(nz, N, A);
      
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

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng1(kernel1, &(A.values[0]), &(A.column_indices[0]));

      const bool cache = true; 
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      eng1.device_colinds = reorder_cols;

      for(int i=0;i< ntests;i++) {
  	ValueType measuretime = eng1.run_x<cache>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), 
  						  thrust::raw_pointer_cast(&dy[0]));
  	statstime(measuretime);
      } 

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify_x(y,ycheck,N, &(kernel1.row_map[0]))) {
  	  std::cout << "current warpkernel_cache_rex time =" << boost::accumulators::mean(statstime) << std::endl;
  	  warpkernel::addData(datafile, "warpkernel_cache_rex", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
      } else std::cout << "failed to verify warp_kernel_cache_rex" << std::endl;

    }

    // warpkernel1 with reorderedx no cache
    {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
      kernel1.scan(nz, N, A);
      

      cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
      {
  	cusp::detail::timer reordercolstimer; reordercolstimer.start();
  	kernel1.reorder_columns_coalesced(reorder_cols);
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

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng1(kernel1, &(A.values[0]), &(A.column_indices[0]));

      const bool cache = false; 
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      eng1.device_colinds = reorder_cols;

      for(int i=0;i< ntests;i++) {
  	ValueType measuretime = eng1.run_x<cache>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), 
  						  thrust::raw_pointer_cast(&dy[0]));
  	statstime(measuretime);
      } 

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify_x(y,ycheck,N, &(kernel1.row_map[0]))) {
  	  std::cout << "current warpkernel_nocache_rex time =" << boost::accumulators::mean(statstime) << std::endl;
  	  warpkernel::addData(datafile, "warpkernel_nocache_rex", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
      } else std::cout << "failed to verify warp_kernel_nocache_rex" << std::endl;

    }


    // warpkernel1 with reorderedx and reordered A with cache
    {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
      kernel1.scan(nz, N, A);

      cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
      {
  	cusp::detail::timer reordercolstimer; reordercolstimer.start();
  	kernel1.reorder_columns_rowsort(reorder_cols, A.row_offsets);
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

      cusp::array1d<ValueType, CPUSpace> A_new_values(nz);
      {
  	cusp::detail::timer reorderAtimer; reorderAtimer.start();
      for(int i=0;i< nz; i++) {
  	A_new_values[i] = A.values[kernel1.reorder_A_rows[i]];
      }
      ValueType reordertime = reorderAtimer.seconds_elapsed();
  	std::cout << "reorder A time " << reordertime << std::endl;
  	warpkernel::addData(datafile, "reorder_A_time_cache_rex", reordertime, -1, -1, -1, -1);
      
      }

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng1(kernel1, 
  									   &A_new_values[0],
  									   &reorder_cols[0]);

      const bool cache = true; 
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      for(int i=0;i< ntests;i++) {
  	ValueType measuretime = eng1.run_x<cache>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), 
  						  thrust::raw_pointer_cast(&dy[0]));
  	statstime(measuretime);
      } 

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify_x(y,ycheck,N, &(kernel1.row_map[0]))) {
  	  std::cout << "current warpkernel_cache_rex_reA time =" << boost::accumulators::mean(statstime) << std::endl;
  	  warpkernel::addData(datafile, "warpkernel_cache_rex_reA", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
      } else std::cout << "failed to verify warp_kernel_cache_rex_reA" << std::endl;

    }

    // warpkernel1 with reorderedx and reordered A with nocache
    {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
      kernel1.scan(nz, N, A);

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

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng1(kernel1, 
  									   &A_new_values[0],
  									   &reorder_cols[0]);

      const bool cache = false; 
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      for(int i=0;i< ntests;i++) {
  	ValueType measuretime = eng1.run_x<cache>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), 
  						  thrust::raw_pointer_cast(&dy[0]));
  	statstime(measuretime);
      } 

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify_x(y,ycheck,N, &(kernel1.row_map[0]))) {
  	  std::cout << "current warpkernel_nocache_rex_reA time =" << boost::accumulators::mean(statstime) << std::endl;
  	  warpkernel::addData(datafile, "warpkernel_nocache_rex_reA", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
      } else std::cout << "failed to verify warp_kernel_nocache_rex_reA" << std::endl;

    }

    // warpkernel1 with reorderedx with cache remap
    {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
      kernel1.scan(nz, N, A);
      
      cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
      kernel1.reorder_columns_coalesced(reorder_cols);

      cusp::array1d<ValueType, DeviceSpace> dreordered_x;
      kernel1.reorder_x(x, dreordered_x);

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng1(kernel1, &(A.values[0]), &(A.column_indices[0]));

      const bool cache = true; 
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      eng1.device_colinds = reorder_cols;

      for(int i=0;i< ntests;i++) {
  	ValueType measuretime = eng1.run_x<cache>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), 
  						  thrust::raw_pointer_cast(&dy[0]));
  	statstime(measuretime);
      } 

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify_x(y,ycheck,N, &(kernel1.row_map[0]))) {
  	  std::cout << "current warpkernel_cache_rexmap time =" << boost::accumulators::mean(statstime) << std::endl;
  	  warpkernel::addData(datafile, "warpkernel_cache_rexmap", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
      } else std::cout << "failed to verify warp_kernel_cache_rexmap" << std::endl;

    }

    // warpkernel1 with reorderedx no cache remap
    {
    cusp::array1d<ValueType, DeviceSpace> dy(N, 0);
      kernel1.scan(nz, N, A);
      
      cusp::array1d<IndexType, CPUSpace> reorder_cols = A.column_indices;
      kernel1.reorder_columns_coalesced(reorder_cols);

      cusp::array1d<ValueType, DeviceSpace> dreordered_x;
      kernel1.reorder_x(x, dreordered_x);

      warpkernel::engine<ValueType, IndexType, warpkernel::structure> eng1(kernel1, &(A.values[0]), &(A.column_indices[0]));

      const bool cache = false; 
      boost::accumulators::accumulator_set<ValueType, boost::accumulators::stats<boost::accumulators::tag::mean>  > statstime;

      eng1.device_colinds = reorder_cols;

      for(int i=0;i< ntests;i++) {
  	ValueType measuretime = eng1.run<cache>(nblocks, blocksize, thrust::raw_pointer_cast(&dreordered_x[0]), 
  						  thrust::raw_pointer_cast(&dy[0]));
  	statstime(measuretime);
      } 

      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify(y,ycheck,N)) {
  	  std::cout << "current warpkernel_nocache_rexmap time =" << boost::accumulators::mean(statstime) << std::endl;
  	  warpkernel::addData(datafile, "warpkernel_nocache_rexmap", boost::accumulators::mean(statstime), kernel1.allocate_nz, kernel1.nwarps, nblocks, blocksize);      
      } else std::cout << "failed to verify warp_kernel_nocache_rexmap" << std::endl;

    }

  }

}
