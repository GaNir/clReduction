# clReduction
Vector sum using reduction on OpenCL

inspired by:
http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

Tested on Nvidia GeForce GT 710B 

- Simple summing on the gpu => 2.8  GB/s
- Optimized Reduction       => 11.0 GB/s 

Capable of reaching about 0.8! of the gpu memory bandwidth limit (14.4 GB/s). Following clReduction.cpp commits, the kernel changes and corresponding runtime speeds can be tracked.

