/*
  Copyright (C) 2012,2018 Jonathan Wong

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once
#ifndef WARP_KERNEL_HPP
#define WARP_KERNEL_HPP

#define NDEBUG 

// Serialization Includes
// #include <boost/archive/text_iarchive.hpp>
// #include <boost/archive/text_oarchive.hpp>
// #include <boost/serialization/vector.hpp>
// #include <boost/serialization/base_object.hpp>

// // stats
// #include <boost/accumulators/accumulators.hpp>
// #include <boost/accumulators/statistics/stats.hpp>
// #include <boost/accumulators/statistics/min.hpp>
// #include <boost/accumulators/statistics/max.hpp>

// General
#include <fstream>
#include <algorithm>

// Cusp includes
#include <cusp/csr_matrix.h>
#include <cusp/system/cuda/arch.h>
#include <performance/timer.h>

// #define WARP_SIZE 32

// Texture cache fetching

// Define texture memory type for double
typedef texture<int2, 1, cudaReadModeElementType> VecType;
VecType x_tex;

using namespace cusp::system::cuda;

namespace warpkernel 
{

  typedef cusp::device_memory DeviceSpace;
  typedef cusp::host_memory CPUSpace;

  template <bool usecache>
  __inline__ __device__ double fetch_cache(const int& i, const double* x) {
    if (usecache) {
      int2 v = tex1Dfetch(x_tex,i);
      return __hiloint2double(v.y, v.x);
    } else {
      return x[i];
    }
  }

  // Define kernels

  template <bool usecache, typename ValueType, typename IndexType>
  __global__ void warpKernel(uint nrows, ValueType* A, IndexType *colinds, 
			     IndexType *rowmap, uint* maxrows, IndexType *warp_offset,
			     ValueType* x, ValueType* y) {

    const uint tid = threadIdx.x;
    const uint id = tid  + blockIdx.x * blockDim.x;
    const uint wid = tid & (WARP_SIZE-1);
    const uint warpid = id / WARP_SIZE;

    if (id < nrows) {

      IndexType toffset = warp_offset[warpid] + wid;
      uint maxnz = maxrows[warpid] * WARP_SIZE + toffset;

      ValueType sum = A[toffset] * fetch_cache<usecache> (colinds[toffset],x);
      for(toffset += WARP_SIZE; toffset < maxnz; toffset += WARP_SIZE) {
	sum += A[toffset] * fetch_cache<usecache> (colinds[toffset],x);
      }

      y[rowmap[id]] = sum;
    }
  }

  // specialized kernel for assembly
  template <bool usecache, typename ValueType, typename IndexType>
  __global__ void warpKernel_assembly(uint nrows, ValueType* A, 
				      IndexType *rowmap, uint* maxrows, IndexType *warp_offset,
				      ValueType* y) {

    const uint tid = threadIdx.x;
    const uint id = tid  + blockIdx.x * blockDim.x;
    const uint wid = tid & (WARP_SIZE-1);
    const uint warpid = id / WARP_SIZE;

    if (id < nrows) {

      IndexType toffset = warp_offset[warpid] + wid;
      uint maxnz = maxrows[warpid] * WARP_SIZE + toffset;

      ValueType sum = A[toffset];
      for(toffset += WARP_SIZE; toffset < maxnz; toffset += WARP_SIZE) {
	sum += A[toffset];
      }

      y[rowmap[id]] = sum;
    }
  }


  template < bool usecache, typename ValueType, typename IndexType>
  __global__ void warpKernel_reorderx(IndexType nrows, ValueType* A, IndexType *colinds, 
				      uint* maxrows, IndexType *warp_offset,
				      ValueType* x, ValueType* y) {

    const uint tid = threadIdx.x;
    const uint id = tid  + blockIdx.x * blockDim.x;
    const uint wid = tid & (WARP_SIZE-1);
    const uint warpid = id / WARP_SIZE;

    if (id < nrows) {

      IndexType toffset = warp_offset[warpid] + wid;
      uint maxnz = maxrows[warpid] * WARP_SIZE + toffset;

      ValueType sum = A[toffset] * fetch_cache<usecache> (colinds[toffset],x);
      for(toffset += WARP_SIZE; toffset < maxnz; toffset += WARP_SIZE) {
	sum += A[toffset] * fetch_cache<usecache> (colinds[toffset],x);
      }

      y[id] = sum;
    }
  }

  template <bool usecache, typename ValueType, typename IndexType>
  __global__ void warpKernel2(IndexType nrows, int nwarps, ValueType* A, IndexType *colinds, 
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
	  sum += sumvalues[tid+i];
	  sumvalues[tid] = sum;
	}
      }

      if ((wid & (offsets-1)) == 0) {
	y[rowmap[rowid]] = sum; 
      }
    }
  }

  template <bool usecache, typename ValueType, typename IndexType>
  __global__ void warpKernel2_reorderx(IndexType nrows, int nwarps, ValueType* A, IndexType *colinds, 
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
	  sum += sumvalues[tid+i];
	  sumvalues[tid] = sum;
	}
      }

      if ((wid & (offsets-1)) == 0) {
	y[rowid] = sum; 
      }
    }
  }

  // Define data structures

  class structure
  {
    //  friend class boost::serialization::access;
    std::string driver_name;

  protected:
    // reverse sort : longest to shortest, then lower row number first
    static bool checkPairOrder(std::pair<uint,uint> pair1, std::pair<uint,uint> pair2) {
      return pair1.second > pair2.second || 
	(pair1.second == pair2.second && pair1.first < pair2.first);
    }

    // inorder sort : lower to higher column number
    static bool checkPairInOrder(std::pair<uint,uint> pair1, std::pair<uint,uint> pair2) {
      return pair1.second < pair2.second ;
    }

    template<typename ValueType>
    void sort(ValueType *rows, std::vector<std::pair<uint,uint> > & nnz_r) {

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

  private:
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      if (version > 0)
	ar & driver_name;
      ar & nrows;
      ar & nznrows;
      ar & nz;
      ar & nwarps;
      ar & allocate_nz;
      ar & row_map;
      ar & row_map_inv;
      ar & reorder_rows;
      ar & warp_offsets;
      ar & max_nz_rows;
      ar & reorder_A_rows;
    }

  public:
    structure(){}
    structure(const uint & nrows_, const uint & nz_, const uint & alloc_) :
      nrows(nrows_), nz(nz_), allocate_nz(alloc_) { }

    std::vector<int> row_map; // y[row_map[i]] = y'[i]
    std::vector<int> row_map_inv; // new_colinds[n] = row_map_inv[colinds[n]]
    std::vector<uint> reorder_rows; // new_values[reorder_rows[i]] = A[i]
    std::vector<int> reorder_A_rows; // only used for reordering A
    std::vector<int> warp_offsets;
    std::vector<uint> max_nz_rows;
    uint nrows;
    uint nznrows;
    uint nz;
    uint allocate_nz;
    uint nwarps;

    virtual void storemap(std::vector<int> & row_map_) {
      row_map.resize(row_map_.size());
      row_map = row_map_;
      nrows = row_map.size();
      row_map_inv.resize(nrows);
      for(int i=0;i<nrows;i++) {
	row_map_inv[row_map[i]] = i;
      }
    }

    void storereorder(std::vector<uint> & reorder_rows_) {
      reorder_rows = reorder_rows_;
      nz = reorder_rows.size();
      if (allocate_nz < nz) allocate_nz = nz;
    }

    void store(std::vector<int> & row_map_, std::vector<uint> & reorder_rows_) {
      storemap(row_map_);
      storereorder(reorder_rows_);
    }

    // void save(std::ofstream & ofs) {
    //   boost::archive::text_oarchive oa(ofs);
    //   oa << *this;
    // }

    // void save(char filename[]) {
    //   std::ofstream ofs(filename);
    //   save(ofs);
    // }

    // void load(std::ifstream & ifs) {
    //   boost::archive::text_iarchive ia(ifs);
    //   ia >> *this;
    // }

    // void load(char filename[]) {
    //   std::ifstream ifs(filename);
    //   load(ifs);
    // }

    template <typename ValueType, typename IndexType>
    void scan(uint & nz_, uint & nrows_, 
	      ValueType * A, IndexType * rows, IndexType *colinds) {

      nrows = nrows_; nz = nz_ ;

      std::vector<std::pair<uint,uint> > nnz_r; // non-zeros per row
      sort(rows, nnz_r);

      std::vector<int> map(nrows,0);
      for(int r = 0; r < nrows; r++)
	map[r] = nnz_r[r].first;

      storemap(map); // sets row_map and row_map_inv

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



    template <typename ValueType, typename IndexType>
    void scan(uint & nz_, uint & nrows_, 
	      cusp::csr_matrix<IndexType, ValueType, CPUSpace> &A) {

      scan(nz_, nrows_, thrust::raw_pointer_cast(&A.values[0]),
	   thrust::raw_pointer_cast(&A.row_offsets[0]),
	   thrust::raw_pointer_cast(&A.column_indices[0]));

    }

    // reorder for coalesced access pattern
    template<typename IndexType, typename ValueType>
    void reorder(ValueType *A, IndexType * colinds,
		 cusp::array1d<ValueType, DeviceSpace> &device_values, // allocate nz
		 cusp::array1d<IndexType, DeviceSpace> &device_colinds) // allocate nz
    {
      cusp::array1d<ValueType, CPUSpace> new_values(allocate_nz,0);
      cusp::array1d<IndexType, CPUSpace> new_colinds(allocate_nz,0);
    
      device_values.resize(allocate_nz,0);
      device_colinds.resize(allocate_nz,0);

      for(int i=0; i< nz; i++) {
	new_values[reorder_rows[i]] = A[i];
	new_colinds[reorder_rows[i]] = colinds[i];
      }

      device_values = new_values;
      device_colinds = new_colinds;
    }

    // reorder column indices of x [CPU]
    template<typename IndexType>
    void reorder_columns(cusp::array1d<IndexType, CPUSpace> & colinds)
    {

      // new column indices
      cusp::array1d<IndexType, CPUSpace>new_colinds(nz);
      for(int n=0; n<nz; n++){
	new_colinds[n] = row_map_inv[colinds[n]];
      }
      colinds = new_colinds;
    
    }

    // reorder column indices of x , sort per row [CPU]
    template<typename IndexType>
    void reorder_columns_rowsort(cusp::array1d<IndexType, CPUSpace> & colinds,
				 cusp::array1d<IndexType, CPUSpace> & rows)
    {

      // new column indices
      cusp::array1d<IndexType, CPUSpace>new_colinds(nz);

      reorder_A_rows.resize(nz);

      for(int r=0; r < nrows; r++ ) {
	std::vector<std::pair<IndexType,IndexType> > temp(rows[r+1] - rows[r]);
	for(int n=rows[r]; n < rows[r+1]; n++){
	  temp[n-rows[r] ] = std::make_pair( n-rows[r], row_map_inv[colinds[n]] );
	}
	// sort within a row
	std::sort(temp.begin(), temp.end(), checkPairInOrder);
      
	for (int i = 0; i < rows[r+1] - rows[r]; i++) {
	  // add back to new_colinds
	  int n = rows[r] + i;
	  new_colinds[n] = temp[i].second;
	
	  // save entries in new mapping
	  reorder_A_rows[n] = temp[i].first + rows[r];
	}
      }
      colinds = new_colinds;
    
    }

    // reorder column indicies and then reorder [CPU]
    template<typename IndexType>
    void reorder_columns_coalesced(cusp::array1d<IndexType, CPUSpace> & colinds)
    {

      cusp::array1d<IndexType, CPUSpace> new_colinds(allocate_nz, 0);
    
      for(int i=0; i<nz ;i++) {
	new_colinds[reorder_rows[i]] = row_map_inv[colinds[i]];
      }
      colinds = new_colinds;
    }

    template<typename IndexType>
    void reorder_columns_coalesced_GPU(cusp::array1d<IndexType, DeviceSpace> & values,
				       cusp::array1d<IndexType, DeviceSpace> & dvalues,
				       cusp::array1d<uint, DeviceSpace> & reorder_rows){

      thrust::scatter(values.begin(), values.end(),
		      reorder_rows.begin(),
		      dvalues.begin());
    }


    template<typename ValueType, typename Array>
    void reorder_values_coalesced(Array & values,
				  cusp::array1d<ValueType, DeviceSpace> & dvalues){

      cusp::array1d<ValueType, CPUSpace> new_values(allocate_nz,0);
      dvalues.resize(allocate_nz);
      for(int i=0;i<nz;i++) {
	new_values[reorder_rows[i]] = values[i];
      }
      dvalues = new_values;
    }

    template<typename ValueType>
    void reorder_values_coalesced(cusp::array1d<ValueType, CPUSpace> & values,
				  cusp::array1d<ValueType, DeviceSpace> & dvalues){

      cusp::array1d<ValueType, CPUSpace> new_values(allocate_nz,0);
      dvalues.resize(allocate_nz);
      for(int i=0;i<nz;i++) {
	new_values[reorder_rows[i]] = values[i];
      }
      dvalues = new_values;
    }

    template<typename ValueType>
    void reorder_values_coalesced_GPU(cusp::array1d<ValueType, DeviceSpace> & values,
				      cusp::array1d<ValueType, DeviceSpace> & dvalues,
				      cusp::array1d<uint, DeviceSpace> & reorder_rows){

      thrust::scatter(values.begin(), values.end(),
		      reorder_rows.begin(),
		      dvalues.begin());
    }


    template<typename ValueType>
    void reorder_values_coalesced_rowsort(cusp::array1d<ValueType, CPUSpace> & values,
					  cusp::array1d<ValueType, DeviceSpace> & dvalues){

      cusp::array1d<ValueType, CPUSpace> new_values(allocate_nz,0);
      dvalues.resize(allocate_nz);
      for(int i=0;i<nz;i++) {
	new_values[reorder_rows[i]] = values[reorder_A_rows[i]];
      }
      dvalues = new_values;
    }
  
    // [CPU}, [GPU]
    template<typename ValueType> 
    void reorder_x(cusp::array1d<ValueType, CPUSpace> &x,
		   cusp::array1d<ValueType, DeviceSpace> &device_x)
    {

      // copy of x
      cusp::array1d<ValueType, CPUSpace> xcopy(x.size());
      for(int r =0; r< nrows; r++) {
	xcopy[row_map_inv[r]] = x[r];
      }
      device_x = xcopy;

    }
  
    // preform the reordering
    template<typename IndexType, typename ValueType>
    void process(ValueType *A, IndexType * colinds,
		 cusp::array1d<ValueType, DeviceSpace> &device_values, // nz
		 cusp::array1d<IndexType, DeviceSpace> &device_colinds, // nz
		 cusp::array1d<IndexType, DeviceSpace> &device_row_map, // nrows
		 cusp::array1d<uint, DeviceSpace> &device_max_nz_per_row, // nwarps
		 cusp::array1d<IndexType, DeviceSpace> &device_warp_offsets) // nwarps
    {

      reorder(A,colinds, device_values, device_colinds);

      device_row_map = row_map;
      device_max_nz_per_row = max_nz_rows;
      device_warp_offsets = warp_offsets;

    }


  };

  class structure2 : public structure
  {
    //  friend class boost::serialization::access;
  // private:
  //   template<class Archive>
  //   void serialize(Archive &ar, const unsigned int version)
  //   {
  //     ar & boost::serialization::base_object<structure>(*this);
  //     ar & nmax_per_thread;
  //     ar & row_offset_warp;
  //     ar & threads_per_row;
  //     ar & min_nz;
  //     ar & max_nz;
  //   }
  public:
    structure2() {}
    structure2(const uint &nrows_, const uint &nz_, const uint & alloc_, const uint & threshold_):
      structure(nrows_, nz_, alloc_), nmax_per_thread(threshold_)
    {}

    std::vector<uint> row_offset_warp;
    std::vector<uint> threads_per_row;
    uint nmax_per_thread; // nz threshold per thread
    uint min_nz;
    uint max_nz;

    void store_param(std::vector<uint> & row_offset_warp_, std::vector<uint> & tpr_) {
      row_offset_warp = row_offset_warp_;
      threads_per_row = tpr_;
    }

    void store(std::vector<int> & row_map_, std::vector<uint> & reorder_rows_,
	       std::vector<uint> & row_offsets_, std::vector<uint> & tpr_) {
      ((structure *)this)->store(row_map_,reorder_rows_);
      store_param(row_offsets_,tpr_);
    }

    // // overload serialization functions to serialize correctly
    // void save(std::ofstream & ofs) {
    //   boost::archive::text_oarchive oa(ofs);
    //   oa << *this;
    // }

    // void save(char * filename) {
    //   std::ofstream ofs(filename);
    //   save(ofs);
    // }

    // void load(std::ifstream & ifs) {
    //   boost::archive::text_iarchive ia(ifs);
    //   ia >> *this;
    // }

    // void load(char * filename) {
    //   std::ifstream ifs(filename);
    //   load(ifs);
    // }

    // overload struct1 scan
    template <typename ValueType, typename IndexType>
    void scan(uint & nz_, uint & nrows_, 
	      ValueType * A, IndexType * rows, IndexType *colinds, 
	      IndexType threshold) {

      nmax_per_thread = threshold;

      nrows = nrows_; nz = nz_ ;

      std::vector<std::pair<uint,uint> > nnz_r; // non-zeros per row
      sort(rows, nnz_r);

      std::vector<int> map(nrows);
      for(int r = 0; r < nrows; r++)
	map[r] = nnz_r[r].first;

      storemap(map); // sets row_map and row_map_inv

      nwarps = (nznrows + WARP_SIZE - 1) / WARP_SIZE;

      std::vector<uint> A_w; // max non-zeros per row
      std::vector<uint> nnz_imin(nwarps,nrows); // minimum non-zeros per row
      std::vector<uint> nnz_imax(nwarps); // maximum non-zeros per row
      std::vector<uint> rows_per_warp;
      std::vector<uint> rows_w(nwarps); // actual rows per warp

      // find global minimum and maximum
      // boost::accumulators::accumulator_set<uint, boost::accumulators::stats<boost::accumulators::tag::min, boost::accumulators::tag::max> > acc;

      // Use sorted row-sizes to calculate A_w, nnz_w, etc.
      for (int w = 0; w < nwarps; w++) {
	for (int r = WARP_SIZE * w; r < nznrows && r < WARP_SIZE*(w+1); r++) {
	  uint rowsize = nnz_r[r].second;
	  assert(rowsize < nrows);
	  if (rowsize < nnz_imin[w]) nnz_imin[w] = rowsize; // min
	  if (rowsize > nnz_imax[w]) nnz_imax[w] = rowsize; // max
	  rows_w[w] += 1;
	  // if (rowsize > 0)
	    //	    acc(rowsize);
	}

	assert(nnz_imax[w] < nrows);
	A_w.push_back( nnz_imax[w]);
	rows_per_warp.push_back(WARP_SIZE);
      }

      // min_nz = boost::accumulators::min(acc);
      // max_nz = boost::accumulators::max(acc);

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

    template <typename ValueType, typename IndexType>
    void scan(uint & nz_, uint & nrows_, 
	      cusp::csr_matrix<IndexType, ValueType, CPUSpace> &A, IndexType threshold) {

      scan(nz_, nrows_, thrust::raw_pointer_cast(&A.values[0]),
	   thrust::raw_pointer_cast(&A.row_offsets[0]),
	   thrust::raw_pointer_cast(&A.column_indices[0]),
	   threshold);

    }

    // preform the reordering
    template<typename IndexType, typename ValueType>
    void process(ValueType *A, IndexType * colinds,
		 cusp::array1d<ValueType, DeviceSpace> &device_values, // nz
		 cusp::array1d<IndexType, DeviceSpace> &device_colinds, // nz
		 cusp::array1d<IndexType, DeviceSpace> &device_row_map, // nrows
		 cusp::array1d<uint, DeviceSpace> &device_max_nz_per_row, // nwarps
		 cusp::array1d<IndexType, DeviceSpace> &device_warp_offsets, // nwarps
		 cusp::array1d<uint, DeviceSpace> &device_threads_per_row, // offsets - nwarps
		 cusp::array1d<uint, DeviceSpace> &device_row_offset_warp) // rows - nwarps
    {

      reorder(A,colinds, device_values, device_colinds);

      device_row_map = row_map;
      device_max_nz_per_row = max_nz_rows;
      device_warp_offsets = warp_offsets;
      device_threads_per_row = threads_per_row;
      device_row_offset_warp = row_offset_warp;

    }

  };

  // purpose of engine is to launch kernel calls and keep track of all the busy work
  template<typename ValueType, typename IndexType>
  class engine
  {

    structure *kernel;

  public:
    // allocate arrays
    cusp::array1d<ValueType, DeviceSpace> device_values; // nz
    cusp::array1d<IndexType, DeviceSpace> device_colinds; // nz
    cusp::array1d<IndexType, DeviceSpace> device_row_map; // nrows
    cusp::array1d<uint, DeviceSpace> device_max_nz_per_row; // nwarps
    cusp::array1d<IndexType, DeviceSpace> device_warp_offsets; // nwarps
    cusp::array1d<uint, DeviceSpace> device_threads_per_row; // offsets - nwarps
    cusp::array1d<uint, DeviceSpace> device_row_offset_warp; // rows - nwarps

    engine(structure & k) {
      kernel = &k;
    }

    engine(structure2 & k, ValueType *A, IndexType *colinds) {
      k.process(A, colinds,
		device_values, device_colinds,
		device_row_map, device_max_nz_per_row,
		device_warp_offsets, 
		device_threads_per_row, device_row_offset_warp);
      kernel = &k;
    }

    engine(structure & k, ValueType *A, IndexType *colinds) {
      k.process(A, colinds,
		device_values, device_colinds,
		device_row_map, device_max_nz_per_row,
		device_warp_offsets);
      kernel = &k;
    }

    template<bool cache>
    float run(int nblocks, int blocksize, ValueType *dx, ValueType *dy) {
      // bind texture to dx
      if (cache) cudaBindTexture(0, x_tex, dx);

      dim3 dimGrid;

      if (nblocks > 65535) {
	dimGrid.x = 65535;
	dimGrid.y = (nblocks + 65534)/65535;
	dimGrid.z = 1;
      } else {
	dimGrid.x = nblocks;
	dimGrid.y = 1;
	dimGrid.z = 1;
      }

      if (dynamic_cast<structure2 *> (kernel) ) {
	timer t;
	warpkernel::warpKernel2<cache> <<< dimGrid, blocksize, (blocksize+16)*sizeof(ValueType) >>>
	  ((int) (*kernel).nznrows, (int) (*kernel).nwarps,
	   thrust::raw_pointer_cast(&device_values[0]),
	   thrust::raw_pointer_cast(&device_colinds[0]),
	   thrust::raw_pointer_cast(&device_row_map[0]),
	   thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
	   thrust::raw_pointer_cast(&device_warp_offsets[0]),
	   thrust::raw_pointer_cast(&device_threads_per_row[0]),
	   thrust::raw_pointer_cast(&device_row_offset_warp[0]),
	   thrust::raw_pointer_cast(&dx[0]),
	   thrust::raw_pointer_cast(&dy[0]));
	cudaDeviceSynchronize();
	ValueType measure = t.seconds_elapsed();
	return measure;
      } else
	if (dynamic_cast<structure *> (kernel) ) {
	  timer t;
	  warpkernel::warpKernel<cache> <<< dimGrid, blocksize >>>
	    ((*kernel).nznrows,
	     thrust::raw_pointer_cast(&device_values[0]),
	     thrust::raw_pointer_cast(&device_colinds[0]),
	     thrust::raw_pointer_cast(&device_row_map[0]),
	     thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
	     thrust::raw_pointer_cast(&device_warp_offsets[0]),
	     dx,
	     dy);
	  cudaDeviceSynchronize();
	  return t.seconds_elapsed();
	}  
      return -1;
    }

    template<bool cache>
    float run_x(int nblocks, int blocksize, ValueType *dx, ValueType *dy) {
      // bind texture to dx
      if (cache) cudaBindTexture(0, x_tex, dx);

      if (dynamic_cast<structure2 *> (kernel) ) {
	timer t;
	warpkernel::warpKernel2_reorderx<cache> <<< nblocks, blocksize, (blocksize+16)*sizeof(ValueType) >>>
	  ((int) (*kernel).nznrows, (int) (*kernel).nwarps,
	   thrust::raw_pointer_cast(&device_values[0]),
	   thrust::raw_pointer_cast(&device_colinds[0]),
	   thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
	   thrust::raw_pointer_cast(&device_warp_offsets[0]),
	   thrust::raw_pointer_cast(&device_threads_per_row[0]),
	   thrust::raw_pointer_cast(&device_row_offset_warp[0]),
	   thrust::raw_pointer_cast(&dx[0]),
	   thrust::raw_pointer_cast(&dy[0]));
		  cudaDeviceSynchronize();
	return t.seconds_elapsed();
      } else
	if (dynamic_cast<structure *> (kernel)) {
	  timer t;
	  warpkernel::warpKernel_reorderx<cache> <<< nblocks, blocksize >>>
	    ((int) (*kernel).nznrows,
	     thrust::raw_pointer_cast(&device_values[0]),
	     thrust::raw_pointer_cast(&device_colinds[0]),
	     thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
	     thrust::raw_pointer_cast(&device_warp_offsets[0]),
	     dx,
	     dy);
	  	  cudaDeviceSynchronize();
	  return t.seconds_elapsed();
	}  
      return -1;
    }

    template<bool cache>
    float run_assembly(int nblocks, int blocksize, ValueType *dy) {
      // bind texture to dx
      // if (typeid(*kernel) == typeid(structure2)) {
      //   timer t;
      //   ();
      //   warpkernel::warpKernel2_reorderx<cache> <<< nblocks, blocksize, (blocksize+16)*sizeof(ValueType) >>>
      // 	((int) (*kernel).nznrows, (int) (*kernel).nwarps,
      // 	 thrust::raw_pointer_cast(&device_values[0]),
      // 	 thrust::raw_pointer_cast(&device_colinds[0]),
      // 	 thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
      // 	 thrust::raw_pointer_cast(&device_warp_offsets[0]),
      // 	 thrust::raw_pointer_cast(&device_threads_per_row[0]),
      // 	 thrust::raw_pointer_cast(&device_row_offset_warp[0]),
      // 	 thrust::raw_pointer_cast(&dx[0]),
      // 	 thrust::raw_pointer_cast(&dy[0]));
      //   return t.seconds_elapsed();
      // } else
      if (dynamic_cast<structure *> (kernel)) {
	timer t;
	warpkernel::warpKernel_assembly<cache> <<< nblocks, blocksize >>>
	  ((int) (*kernel).nznrows,
	   thrust::raw_pointer_cast(&device_values[0]),
	   //	   thrust::raw_pointer_cast(&device_colinds[0]),
	   thrust::raw_pointer_cast(&device_max_nz_per_row[0]),
	   thrust::raw_pointer_cast(&device_warp_offsets[0]),
	   //	   dx,
	   dy);
	return t.seconds_elapsed();
      }  
      return -1;
    }


    template<typename R>
    bool verify(R orig, R comp) {
      bool check = true;
      for (int i=0; i< (*kernel).nrows; i++) {
	if ((abs(orig[i]) == 0 && comp[i]!=0) || abs((orig[i]-comp[i])/orig[i]) > 1E-5) {
	  std::cout << orig[i] << "\t" << comp[i] << "\t" << i << std::endl;
	  check= false;
	  return check;
	}
      }
      return check;
    }

    template<typename R>
    bool verify_x(R orig, R comp) {
      bool check = true;
      for (int i=0; i< (*kernel).nrows; i++) {
	if ((abs(orig[(*kernel).row_map[i]]) == 0 && comp[i]!=0) 
	    || abs((orig[(*kernel).row_map[i]]-comp[i])/orig[(*kernel).row_map[i]]) > 1E-5) {
	  std::cout << orig[(*kernel).row_map[i]] << "\t" << comp[i] << "\t" << i << std::endl;
	  check = false;
	  return check;
	}
      }
      return check;
    }

  };

  // helper functions

  // void startDatafile(std::ofstream & ofs, uint nz, uint nrows, uint ntests) {

  //   ofs << "nz\t" << nz << std::endl;
  //   ofs << "nrows\t" << nrows << std::endl;
  //   ofs << "bytes\t" << nz * 20 << std::endl;
  //   ofs << "ntests\t" << ntests << std::endl; // compatibility with past results files
  //   ofs << std::endl;
  //   ofs << std::endl;
  //   ofs << "Kernel name\t"
  //       << "padded_nz\t"
  //       << "time [s]\t"
  //       << "nwarps\t" 
  //       << "nblocks\t"
  //       << "blocksize" << std::endl;

  // }

  // void addData(std::ofstream &ofs, char * kernelname, float measuredtime, int padded_nz,
  // 	     int nwarps, int nblocks, int blocksize)
  // {
  //   ofs << kernelname << "\t";
  //   ofs << padded_nz << "\t";
  //   ofs << std::scientific << measuredtime << "\t";
  //   ofs << nwarps << "\t";
  //   ofs << nblocks  << "\t";
  //   ofs << blocksize << std::endl;
  // }



  // void addData(std::ofstream &ofs, char * kernelname, float measuredtime, 
  // 		      structure &s, uint blocksize) 
  // {
  //   addData(ofs, 
  // 	  kernelname, 
  // 	  measuredtime, 
  // 	  s.allocate_nz, 
  // 	  s.nwarps, 
  // 	  (s.nwarps + (blocksize/WARP_SIZE)-1)/(blocksize/WARP_SIZE),
  // 	  blocksize);
  // }

}
//BOOST_CLASS_VERSION(warpkernel::structure,1)
//BOOST_CLASS_VERSION(warpkernel::structure2,1)

#endif
