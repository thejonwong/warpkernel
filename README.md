# GPU SPMV Warpkernels

Original copy of the GPU SPMV kernel. Since they were written using CUDA 4.0, minor modifications to handle the change related to synchrnouse behavior of warps needs to be adjusted.

## Citation

If you use **WarpKernel** in your research, please cite the following paper:

### Journal Version (International Journal for Numerical Methods in Engineering)

```bibtex
@article{WongKuhlDarve2015,
  author    = {Wong, Jonathan and Kuhl, Ellen and Darve, Eric},
  title     = {A new sparse matrix vector multiplication graphics processing unit algorithm designed for finite element problems},
  journal   = {International Journal for Numerical Methods in Engineering},
  volume    = {102},
  number    = {12},
  pages     = {1784--1814},
  year      = {2015},
  doi       = {10.1002/nme.4865}
}
```

### Arxiv version
```bibtex
@article{DBLP:journals/corr/WongKD15,
  author       = {Jonathan Wong and Ellen Kuhl and Eric Darve},
  title        = {A New Sparse Matrix Vector Multiplication {GPU} Algorithm Designed
                  for Finite Element Problems},
  journal      = {CoRR},
  volume       = {abs/1501.00324},
  year         = {2015},
  url          = {http://arxiv.org/abs/1501.00324},
  eprinttype   = {arXiv},
  eprint       = {1501.00324},
  biburl       = {https://dblp.org/rec/journals/corr/WongKD15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```

Requirements for warpkernel.hpp:
1. CUDA 4.0
2. Thrust (http://docs.nvidia.com/cuda/thrust/index.html)

Optional Requirements for test codes. Packages listed below are used for comparison of results:
3. CUSP (http://code.google.com/p/cusp-library/)
4. ModernGPU sparse 

Installation:
Run "make all"

Warpkernel Library Usage:
1. scan
Prototype:
```C
  void scan(uint & nz_, uint & nrows_, ValueType * A, IndexType * rows, IndexType *colinds, [IndexType threshold])
```
This scans the matrix A and analyzes its structure to generate the propper mappings for the warpkernels. For warpkernel::structure (K1) and warpkernel::structure2 (K2) have similar but separate scan calls that convert a matrix into the WPK1/2 formats repsectively.

2. Process
Performs the re-ordering of values and columns into the corresponding WPK format. warpkernel::engine instatiation will call this by default.

3. Run
`warpkernel::engine::run` will perform the SPMV multiplication on arrays `x` and `y`. On the other hand `warpkernel::engine::run_x` will perform the multiplication with the entries of X reordered (Kr). This means that for each entry of the reordered `x` array `x'`, the result from run_x yields the proper corresponding entry in the original solution array `y`, `y'`. run computes `A*x=y` and run_x computes `A*x'=y'`.

4. Verify
`warpkernel::engine::verify` compare the computed vs another GPU SPMV computed result. `verify_x` remaps the compared result such that the ordering matches the ordering for the result vector for hte computed result if Kr SPMV algorithms are used.
