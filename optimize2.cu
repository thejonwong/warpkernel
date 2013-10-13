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
template <bool usecache, typename ValueType, typename IndexType>
__global__ void warpKernel2_noregister_noremap(IndexType nrows, int nwarps, ValueType* A, IndexType *colinds, 
			    uint* maxrows, IndexType *warp_offset,
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
      y[rowid] = sumvalues[tid]; 
    }
  }
}


/**********/

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
	  std::vector<int> &row_map_inv,
	  uint & nmax_per_thread,
	  uint & min_nz, uint & max_nz,
	  std::vector<uint> & row_offset_warp,
	  std::vector<uint> & threads_per_row)
 {

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


  std::vector<uint> A_w; // max non-zeros per row
  std::vector<uint> nnz_imin(nwarps,nrows); // minimum non-zeros per row
  std::vector<uint> nnz_imax(nwarps); // maximum non-zeros per row
  std::vector<uint> rows_per_warp;
  std::vector<uint> rows_w(nwarps); // actual rows per warp

  // find global minimum and maximum
  boost::accumulators::accumulator_set<uint, boost::accumulators::stats<boost::accumulators::tag::min, boost::accumulators::tag::max> > acc;

  // Use sorted row-sizes to calculate A_w, nnz_w, etc.
  for (int w = 0; w < nwarps; w++) {
    for (int r = WARP_SIZE * w; r < nznrows && r < WARP_SIZE*(w+1); r++) {
      uint rowsize = nnz_r[r].second;
      assert(rowsize < nrows);
      if (rowsize < nnz_imin[w]) nnz_imin[w] = rowsize; // min
      if (rowsize > nnz_imax[w]) nnz_imax[w] = rowsize; // max
      rows_w[w] += 1;
      if (rowsize > 0)
	acc(rowsize);
    }

    assert(nnz_imax[w] < nrows);
    A_w.push_back( nnz_imax[w]);
    rows_per_warp.push_back(WARP_SIZE);
  }

  min_nz = boost::accumulators::min(acc);
  max_nz = boost::accumulators::max(acc);

  // split up the warps according to the threshold parameter
  {

    std::vector<uint>::iterator warp_it;
    std::vector<uint>::iterator rows_it;
    std::vector<uint>::iterator rowscount_it;

    rows_it = rows_per_warp.begin();
    rowscount_it = rows_w.begin();

    IndexType thmax_per_row = 32;
    IndexType nmax_per_warp;
    for(warp_it = A_w.begin(); warp_it < A_w.end(); warp_it++) {
      if (*warp_it > nmax_per_thread) {
	nmax_per_warp = nmax_per_thread;
	// determine if it fits the requirements
	if (*warp_it > nmax_per_thread * thmax_per_row) {
	  //	    std::cout << "Exceeds nmax_per_thread * thmax_per_row assumptions - split for row" << std::endl;
	  nmax_per_warp = (*warp_it+thmax_per_row-1)/thmax_per_row;
	}

	uint temp = *warp_it;
	for(int i=2; i <= thmax_per_row; i*=2) {
	  if (temp <= nmax_per_warp * i) {
	    uint nz_per_thread = (temp+i-1)/i;
	    *warp_it = nz_per_thread;

	    if (*rowscount_it * i < WARP_SIZE) {
	      // last warp in the sorted nnz_r list
	      *rows_it = WARP_SIZE/i;

	    } else {
	      *rows_it = WARP_SIZE/i;
	      *rowscount_it = WARP_SIZE/i;

	      for(int j=0; j<i-1; j++) {
		assert(nz_per_thread <= nmax_per_warp);
		warp_it = A_w.insert(warp_it, nz_per_thread);
		rows_it = rows_per_warp.insert(rows_it, WARP_SIZE/i);
		rowscount_it = rows_w.insert(rowscount_it, WARP_SIZE/i);
	      }
    
	    }

	    warp_it += i-1;
	    rows_it += i-1 ;
	    rowscount_it += i-1;
	  
	    break;
	  }
	}
      }

      rowscount_it += 1;
      rows_it += 1;
    }
 
    nwarps = A_w.size();
  }
    
  max_nz_rows = A_w;

  // set warp_offsets and allocate_nz;
  warp_offsets.resize(nwarps+1);
  warp_offsets[0] = 0;
  for(int w = 0; w < nwarps; w++) {
    assert(A_w[w]>0);
    warp_offsets[w+1] = warp_offsets[w] + A_w[w] * WARP_SIZE;
  }
  allocate_nz = warp_offsets[nwarps];

  threads_per_row.resize(nwarps,0);
  for(int w=0; w<nwarps; w++) {
    threads_per_row[w] = WARP_SIZE/rows_per_warp[w];
  }

  // Define row_offset_warp
  {
    row_offset_warp.resize(nwarps+1,0); // starting row of the warp

    for(int w = 0; w< nwarps; w++) {
      row_offset_warp[w+1] = row_offset_warp[w] + rows_w[w];
    }
    row_offset_warp[nwarps] = nrows;
  }

  // Use the modified reordering map
  reorder_rows.resize(nz);
  for (int w_s = 0; w_s < nwarps; w_s++) {
    for (int r = 0; r < rows_per_warp[w_s] 
	   && row_offset_warp[w_s] + r < nznrows; r++ ) {
      int r_s = row_offset_warp[w_s] + r;
      int row = nnz_r[r_s].first;
      int rowsize = nnz_r[r_s].second;

      for (int tid = 0; tid < threads_per_row[w_s]; tid++) {
	for (int i = 0; i + tid * A_w[w_s] < rowsize && i < A_w[w_s]; i++) {
	  reorder_rows[rows[row] + i + tid * A_w[w_s] ] =
	    warp_offsets[w_s] + (tid + r * threads_per_row[w_s]) + i * WARP_SIZE;
	}
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

  IndexType threshold = 6;
  bool cache = true;
  int warps_per_block = 1; 
  std::string matrixfilename = argv[1];
  int ntests = 1;
  if (argc >2 ) ntests = atoi(argv[2]);
  if (argc >3) cache = (1==atoi(argv[3]));
  if (argc > 4) warps_per_block = atoi(argv[4]);
  if (argc > 5) threshold = atoi(argv[5]);

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
 
  std::string datapath = "./data/" + matrixname + "_optimize2_" + (cache ? "cache" : "nocache" ) + ".txt";
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

    warpkernel::structure2 kernel1(N, nz, 0, threshold);
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
		 kernel1.row_map_inv,
		 kernel1.nmax_per_thread,
		 kernel1.min_nz,kernel1.max_nz,
		 kernel1.row_offset_warp,
		 kernel1.threads_per_row);


      uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
      uint blocksize = warps_per_block * WARP_SIZE;

      warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng(kernel1,
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
	std::cout << "warpkernel2 (" << nblocks << "," << blocksize <<") time = " 
		  << std::scientific << measuretime/ntests << std::endl;
	warpkernel::addData(datafile, "warpkernel2", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel2" << std::endl;
    }
    //

    // normal kernel no sorting
    {
      
      scan<false>(kernel1.nz, kernel1.nrows,
		 &(A.values[0]), &(A.row_offsets[0]), &(A.column_indices[0]), 
		 kernel1.nwarps, kernel1.nznrows, kernel1.allocate_nz,
		 kernel1.reorder_rows, 
		 kernel1.warp_offsets,
		 kernel1.max_nz_rows,
		 kernel1.row_map,
		 kernel1.row_map_inv,
		 kernel1.nmax_per_thread,
		 kernel1.min_nz,kernel1.max_nz,
		 kernel1.row_offset_warp,
		 kernel1.threads_per_row);


      uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
      uint blocksize = warps_per_block * WARP_SIZE;

      warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng(kernel1,
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
	std::cout << "warpkernel2_nosort (" << nblocks << "," << blocksize <<") time = " 
		  << std::scientific << measuretime/ntests << std::endl;
	warpkernel::addData(datafile, "warpkernel2_nosort", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel2_nosort" << std::endl;
    }
    //

    // normal kernel with reordered x and columninds
    {
      kernel1.scan(nz, N, A, threshold);

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

      warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng(kernel1,
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
    	std::cout << "warpkernel2_rex(" << nblocks << "," << blocksize <<") time = " 
    		  << std::scientific << measuretime/ntests << std::endl;
    	warpkernel::addData(datafile, "warpkernel2_rex", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel2_rex" << std::endl;

      eng.device_colinds = restore_col;
    }
    //

    // normal kernel with reordered x and columninds with rowmapping
    {
      kernel1.scan(nz, N, A,threshold);

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

      warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng(kernel1,
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
    	std::cout << "warpkernel2_rex_rowmap(" << nblocks << "," << blocksize <<") time = " 
    		  << std::scientific << measuretime/ntests << std::endl;
    	warpkernel::addData(datafile, "warpkernel2_rex_rowmap", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel2_rex_rowmap" << std::endl;

      eng.device_colinds = restore_col;
    }
    //

    // normal kernel with reordered x and columninds and A.values
    {
      kernel1.scan(nz, N, A, threshold);

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

      warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng(kernel1,
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
    	std::cout << "warpkernel2_rex_rowsort(" << nblocks << "," << blocksize <<") time = " 
    		  << std::scientific << measuretime/ntests << std::endl;
    	warpkernel::addData(datafile, "warpkernel2_rex_rowsort", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel2_rex_rowsort" << std::endl;

    }
    //

    /************************************* no register versions ***************************/

    // normal kernel no register
    {
      
      scan<true>(kernel1.nz, kernel1.nrows,
		 &(A.values[0]), &(A.row_offsets[0]), &(A.column_indices[0]), 
		 kernel1.nwarps, kernel1.nznrows, kernel1.allocate_nz,
		 kernel1.reorder_rows, 
		 kernel1.warp_offsets,
		 kernel1.max_nz_rows,
		 kernel1.row_map,
		 kernel1.row_map_inv,
		 kernel1.nmax_per_thread,
		 kernel1.min_nz,kernel1.max_nz,
		 kernel1.row_offset_warp,
		 kernel1.threads_per_row);


      uint nblocks = (kernel1.nwarps + warps_per_block-1)/warps_per_block;
      uint blocksize = warps_per_block * WARP_SIZE;

      warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng(kernel1,
									  &(A.values[0]),
									  &(A.column_indices[0]));


      cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
      ValueType measuretime = 0;
      if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dx[0]));
      for (int t = 0; t < ntests; t++) {
	if (cache == true) {
	  cusp::detail::timer t;
	  t.start();
	  warpKernel2_noregister<true> <<< nblocks, blocksize, (blocksize+16)*sizeof(ValueType) >>>
	    ((int) kernel1.nznrows, (int) kernel1.nwarps,
	     thrust::raw_pointer_cast(&eng.device_values[0]),
	     thrust::raw_pointer_cast(&eng.device_colinds[0]),
	     thrust::raw_pointer_cast(&eng.device_row_map[0]),
	     thrust::raw_pointer_cast(&eng.device_max_nz_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_warp_offsets[0]),
	     thrust::raw_pointer_cast(&eng.device_threads_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_row_offset_warp[0]),
	     thrust::raw_pointer_cast(&dx[0]),
	     thrust::raw_pointer_cast(&dy[0]));
	  measuretime += t.seconds_elapsed();
	} else {
	  cusp::detail::timer t;
	  t.start();
	  warpKernel2_noregister<false> <<< nblocks, blocksize, (blocksize+16)*sizeof(ValueType) >>>
	    ((int) kernel1.nznrows, (int) kernel1.nwarps,
	     thrust::raw_pointer_cast(&eng.device_values[0]),
	     thrust::raw_pointer_cast(&eng.device_colinds[0]),
	     thrust::raw_pointer_cast(&eng.device_row_map[0]),
	     thrust::raw_pointer_cast(&eng.device_max_nz_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_warp_offsets[0]),
	     thrust::raw_pointer_cast(&eng.device_threads_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_row_offset_warp[0]),
	     thrust::raw_pointer_cast(&dx[0]),
	     thrust::raw_pointer_cast(&dy[0]));
	  measuretime += t.seconds_elapsed();
	}
      }
      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify(y,ycheck,N)) {
	std::cout << "warpkernel2_noreg (" << nblocks << "," << blocksize <<") time = " 
		  << std::scientific << measuretime/ntests << std::endl;
	warpkernel::addData(datafile, "warpkernel2_noreg", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel2_noreg" << std::endl;
    }
    //

    // normal kernel with reordered x and columninds
    {
      kernel1.scan(nz, N, A, threshold);

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

      warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng(kernel1,
    									  &(A.values[0]),
    									  &(A.column_indices[0]));

      eng.device_colinds = reorder_cols;

      cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
      ValueType measuretime = 0;
      if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dreordered_x[0]));
      for (int t = 0; t < ntests; t++) {
	if (cache == true) {
	  cusp::detail::timer t;
	  t.start();
	  warpKernel2_noregister_noremap<true> <<< nblocks, blocksize, (blocksize+16)*sizeof(ValueType) >>>
	    ((int) kernel1.nznrows, (int) kernel1.nwarps,
	     thrust::raw_pointer_cast(&eng.device_values[0]),
	     thrust::raw_pointer_cast(&eng.device_colinds[0]),
	     thrust::raw_pointer_cast(&eng.device_max_nz_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_warp_offsets[0]),
	     thrust::raw_pointer_cast(&eng.device_threads_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_row_offset_warp[0]),
	     thrust::raw_pointer_cast(&dreordered_x[0]),
	     thrust::raw_pointer_cast(&dy[0]));
	  measuretime += t.seconds_elapsed();
	} else {
	  cusp::detail::timer t;
	  t.start();
	  warpKernel2_noregister_noremap<false> <<< nblocks, blocksize, (blocksize+16)*sizeof(ValueType) >>>
	    ((int) kernel1.nznrows, (int) kernel1.nwarps,
	     thrust::raw_pointer_cast(&eng.device_values[0]),
	     thrust::raw_pointer_cast(&eng.device_colinds[0]),
	     thrust::raw_pointer_cast(&eng.device_max_nz_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_warp_offsets[0]),
	     thrust::raw_pointer_cast(&eng.device_threads_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_row_offset_warp[0]),
	     thrust::raw_pointer_cast(&dreordered_x[0]),
	     thrust::raw_pointer_cast(&dy[0]));
	  measuretime += t.seconds_elapsed();
	}
      }
      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify_x(y,ycheck,N,&kernel1.row_map[0])) {
    	std::cout << "warpkernel2_noreg_rex(" << nblocks << "," << blocksize <<") time = " 
    		  << std::scientific << measuretime/ntests << std::endl;
    	warpkernel::addData(datafile, "warpkernel2_no_reg_rex", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel2_noreg_rex" << std::endl;

      eng.device_colinds = restore_col;
    }
    //

    // normal kernel with reordered x and columninds and A.values
    {
      kernel1.scan(nz, N, A, threshold);

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
      }

      warpkernel::engine<ValueType, IndexType, warpkernel::structure2> eng(kernel1,
    									  &(A_new_values[0]),
    									  &(reorder_cols[0]));


      cusp::array1d<ValueType, DeviceSpace> dy(N,0);  
      ValueType measuretime = 0;
      if (cache) cudaBindTexture(0, x_tex, thrust::raw_pointer_cast(&dreordered_x[0]));
      for (int t = 0; t < ntests; t++) {
	if (cache == true) {
	  cusp::detail::timer t;
	  t.start();
	  warpKernel2_noregister_noremap<true> <<< nblocks, blocksize, (blocksize+16)*sizeof(ValueType) >>>
	    ((int) kernel1.nznrows, (int) kernel1.nwarps,
	     thrust::raw_pointer_cast(&eng.device_values[0]),
	     thrust::raw_pointer_cast(&eng.device_colinds[0]),
	     thrust::raw_pointer_cast(&eng.device_max_nz_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_warp_offsets[0]),
	     thrust::raw_pointer_cast(&eng.device_threads_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_row_offset_warp[0]),
	     thrust::raw_pointer_cast(&dreordered_x[0]),
	     thrust::raw_pointer_cast(&dy[0]));
	  measuretime += t.seconds_elapsed();
	} else {
	  cusp::detail::timer t;
	  t.start();
	  warpKernel2_noregister_noremap<false> <<< nblocks, blocksize, (blocksize+16)*sizeof(ValueType) >>>
	    ((int) kernel1.nznrows, (int) kernel1.nwarps,
	     thrust::raw_pointer_cast(&eng.device_values[0]),
	     thrust::raw_pointer_cast(&eng.device_colinds[0]),
	     thrust::raw_pointer_cast(&eng.device_max_nz_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_warp_offsets[0]),
	     thrust::raw_pointer_cast(&eng.device_threads_per_row[0]),
	     thrust::raw_pointer_cast(&eng.device_row_offset_warp[0]),
	     thrust::raw_pointer_cast(&dreordered_x[0]),
	     thrust::raw_pointer_cast(&dy[0]));
	  measuretime += t.seconds_elapsed();
	}
      }
      cusp::array1d<ValueType, CPUSpace> ycheck = dy;
      if (verify_x(y,ycheck,N,&kernel1.row_map[0])) {
    	std::cout << "warpkernel2_noreg_rex_rowsort(" << nblocks << "," << blocksize <<") time = " 
    		  << std::scientific << measuretime/ntests << std::endl;
    	warpkernel::addData(datafile, "warpkernel2_no_reg_rex_rowosrt", measuretime/ntests, kernel1, blocksize);	      
      } else std::cout << "Failed warpkernel2_noreg_rex_rowsort" << std::endl;

      eng.device_colinds = restore_col;
    }
    //

  }
}
