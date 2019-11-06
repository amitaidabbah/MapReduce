 # MapReduce
  Multi-threaded Programming
The implementation of this design is split into two parts:
1) Implementing the functions map and reduce. This will be different for every task. We call this part the client. 
2) Implementing everything else – the partition into phases, distribution of work between threads, synchronisation etc. This will be identical for different tasks. This is called the framework

The framework will support running a MapReduce operations as an asynchrony job, together with ability to query the current state of a job while it is running. A header MapReduceFramework.h is provided.
