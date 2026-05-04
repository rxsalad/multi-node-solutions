# Multi-Node RCCL Testing for MI350x8 GPU Droplets

Follow [the guide](mi350x8-mn-environment.MD) to prepare the environment before proceeding.

This guide provides a basic introduction and step-by-step instructions for manually performing RCCL testing in a multi-node MI350x8 GPU droplet environment.

## Test Environment

```
GPU Droplet Image ID: 221919586
Guest OS: Ubuntu 24.04
Linux Kernel: 6.8.0-106-generic
ROCm: 7.0.2 
GPU Driver: 6.14.14
GPU Firmware: 01.25.17.07
```

| GPU Droplet    | rs-mi350x8-mn-test1  | rs-mi350x8-mn-test2  | 
|----------------|----------------|-----------------|
| eth0-internet  | 143.110.201.31 | 178.128.11.166 |
| eth1-vpc       | 10.120.0.9/20  | 10.120.0.10/20 | 
| eth2-roce      | 192.168.52.101/24 | 192.168.52.102/24  | 
| eth3-roce      | 192.168.53.101/24 | 192.168.53.102/24  | 
| eth4-roce      | 192.168.54.101/24 | 192.168.54.102/24  | 
| eth5-roce      | 192.168.55.101/24 | 192.168.55.102/24  |
| eth6-roce      | 192.168.56.101/24 | 192.168.56.102/24  | 
| eth7-roce      | 192.168.57.101/24 | 192.168.57.102/24  | 
| eth8-roce      | 192.168.58.101/24 | 192.168.58.102/24  | 
| eth9-roce      | 192.168.59.101/24 | 192.168.59.102/24  | 


## OpenMPI Installation and Configuration

[OpenMPI](https://www.open-mpi.org/) is an implementation of [the MPI standard](https://www.mpi-forum.org/) that provides runtime tools (such as mpirun) for launching distributed applications, as well as communication libraries for coordinating processes across multiple nodes. Although it supports CPU-based collective communication, for RCCL testing, OpenMPI is used strictly as a launch and coordination layer across nodes, while all GPU-to-GPU communication is handled entirely by RCCL.

Install OpenMPI on each GPU droplet:
```
# openmpi-bin: runtime tools for executing MPI applications
# libopenmpi-dev: development libraries and headers for building MPI programs

apt install -y openmpi-bin libopenmpi-dev
```

OpenMPI relies on SSH for launching processes across nodes, so we need to configure SSH key-based authentication:

|                | rs-mi350x8-mn-test1  | rs-mi350x8-mn-test2  | 
|----------------|----------------|-----------------|
| IP Address (eth1-vpc)        | 10.120.0.9/20  | 10.120.0.10/20 | 
| Key Pair   | `ssh-keygen -b 4096 -t rsa` |  | 
| SSH client configuration    | `nano ~/.ssh/config` <br> Host * <br> &nbsp;&nbsp; User root <br> &nbsp;&nbsp; IdentityFile ~/.ssh/id_rsa |  | 
| Copy the public key<br> (~/.ssh/id_rsa.pub) <br> to each node   | `cat ~/.ssh/id_rsa.pub` <br> `nano ~/.ssh/authorized_keys`  |  <br> `nano ~/.ssh/authorized_keys` | 
| Test SSH   | `ssh 10.120.0.9` <br> `ssh 10.120.0.10`  |  | 

Use OpenMPI to run a simple distributed test across two droplets from `rs-mi350x8-mn-test1`:

```
root@rs-mi350x8-mn-test1:~# mpirun --version
mpirun (Open MPI) 4.1.6

# OpenMPI will use SSH to connect to both nodes, launch an MPI process on each node, each running the hostname command

root@rs-mi350x8-mn-test1:~# mpirun -H 10.120.0.9,10.120.0.10 --allow-run-as-root hostname

rs-mi350x8-mn-test1
rs-mi350x8-mn-test2
```

## Basic Distributed Test with mpi4py

Install `mpi4py` on each GPU droplet, a Python library that provides bindings to MPI (the underlying message-passing communication library), enabling Python applications to perform distributed parallel computing across multiple processes and nodes.

```
pip install mpi4py --break-system-packages
```

Write a simple MPI program in Python and copy it to each droplet (/root/hello_mpi.py):

``` python 
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

hostname = MPI.Get_processor_name()

print(f"Hello from rank {rank} out of {size} on {hostname}")

if size >= 2:
    if rank == 0:
        msg = "Hello from rank 0"
        comm.send(msg, dest=1, tag=0)
        print("Rank 0 sent message to rank 1")

    elif rank == 1:
        msg = comm.recv(source=0, tag=0)
        print(f"Rank 1 received message: {msg}")
```

Use OpenMPI to launch the application across the 2 droplets from `rs-mi350x8-mn-test1`:

```
# OpenMPI uses SSH to connect to both nodes, launches an MPI process on each node, assigns ranks to each process, and enables reliable inter-process communication (send/receive) between them as if they were part of a single distributed application.

root@rs-mi350x8-mn-test1:~# mpirun -H 10.120.0.9,10.120.0.10 --allow-run-as-root python3 /root/hello_mpi.py

Hello from rank 1 out of 2 on rs-mi350x8-mn-test2
Hello from rank 0 out of 2 on rs-mi350x8-mn-test1
Rank 0 sent message to rank 1
Rank 1 received message: Hello from rank 0
```

## RCCL Test Setup

[RCCL](https://github.com/ROCm/rccl) (ROCm Communication Collectives Library) is part of the ROCm software stack and provides collective communication primitives (e.g., all-reduce, broadcast, reduce-scatter) for multi-GPU and multi-node workloads. [RCCL testing](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/common/rccl-benchmarking.html#rccl-benchmarking) can ensure correctness, connectivity, and performance of multi-GPU and multi-node communication in a new environment before running real workloads.

Check the RCCL and OpenMPI versions and installation locations on each GPU droplet:

```
# When PyTorch (ROCm build) runs, it relies on a compatible ROCm stack provided by the system or container environment at runtime via dynamic linking.

cat /opt/rocm/.info/version

dpkg -l | grep rccl
ls /opt/rocm/lib | grep rccl

dpkg -l | grep openmpi
mpicc -show
ls /usr/lib/x86_64-linux-gnu/openmpi/lib
```

To run the test, install AMD’s official RCCL test tool ([rccl-tests](https://github.com/ROCm/rccl-tests)) on each GPU droplet:

```
git clone https://github.com/rocm/rccl-tests.git

cd rccl-tests/

# Build the program using all CPU cores, with MPI support enabled, and explicitly using OpenMPI installed at /usr/lib/x86_64-linux-gnu/openmpi. 
# It assumes that librccl.so is installed in /opt/rocm/lib/.

make -j$(nproc) MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
```

## All-Reduce Concept

All-reduce is a collective communication operation commonly used in distributed training (and sometimes inference) to keep multiple GPUs synchronized.

Each GPU starts with its own local data, and all GPUs collaboratively compute a global aggregate (usually a sum or average). The final result is then shared back to every GPU, ensuring all replicas stay identical.

### What It Does 

Example: 4 GPUs, each holding a vector

- GPU1: (a1, b1, c1, d1)
- GPU2: (a2, b2, c2, d2)
- GPU3: (a3, b3, c3, d3)
- GPU4: (a4, b4, c4, d4)

Goal - Compute an element-wise aggregation (sum or average) across all GPUs, and ensure each GPU receives the same aggregated vector:

- GPU1: (a1+a2+a3+a4, b1+b2+b3+b4, c1+c2+c3+c4, d1+d2+d3+d4)
- GPU2: (a1+a2+a3+a4, b1+b2+b3+b4, c1+c2+c3+c4, d1+d2+d3+d4)
- GPU3: (a1+a2+a3+a4, b1+b2+b3+b4, c1+c2+c3+c4, d1+d2+d3+d4)
- GPU4: (a1+a2+a3+a4, b1+b2+b3+b4, c1+c2+c3+c4, d1+d2+d3+d4)

### How It Works 

Instead of sending all data to a single central node (which becomes a bottleneck), all-reduce is performed in a distributed manner using efficient network patterns such as ring or tree algorithms. It typically consists of two phases:

#### Reduction Phase 

All GPUs collaborate by exchanging and combining their corresponding elements. Through these exchanges, each GPU progressively accumulates partial results until the full reduced values are computed. 

At the end of this phase, each GPU holds one portion of the reduced result. For example:

- GPU1: a1+a2+a3+a4 
- GPU2: b1+b2+b3+b4
- GPU3: c1+c2+c3+c4
- GPU4: d1+d2+d3+d4

#### Distribution Phase 

The partial reduced results are then exchanged so that every GPU receives all segments of the final vector. After this step, all GPUs hold the complete aggregated result and are fully synchronized.

### Communication Cost

For N GPUs, each holding M data elements, the per-GPU communication cost of ring all-reduce is:

Reduction phase:  M x (N − 1) / N   
Broadcast phase:  M x (N − 1) / N   
Total per GPU: 2 x M x (N − 1) / N   

Although each GPU only communicates with its immediate neighbors in the ring at each step, the algorithm still achieves global aggregation by progressively relaying and combining data across multiple rounds, effectively circulating information through the entire ring without requiring a central coordinator.

The communication is typically handled by RCCL for AMD GPUs and carried over:
- XGMI for high-bandwidth intra-node GPU communication
- RoCE (RDMA over Converged Ethernet) for low-latency inter-node GPU communication

## Single Node RCCL Testing

This test evaluates the performance of 8 AMD MI350X GPUs within `rs-mi350x8-mn-test1` executing an all-reduce operation on a 16 GiB tensor per GPU using RCCL:

- Out-of-place means the operation uses separate input and output buffers, so the original data is preserved, while in-place overwrites the input buffer with the result to save memory. 

- [algbw](https://github.com/ROCm/rccl-tests/blob/develop/doc/PERFORMANCE.md#algorithm-bandwidth) (algorithmic bandwidth) measures the effective per-GPU throughput of the all-reduce operation based on the input data size and total execution time (17.179869184 GB / 0.076026 s = 225.97 GB/s).  

- [busbw](https://github.com/ROCm/rccl-tests/blob/develop/doc/PERFORMANCE.md#algorithm-bandwidth) (bus bandwidth) measures the effective per-GPU load on the interconnect (e.g., XGMI and RoCE), accounting for all intermediate data transfers during the collective operation. It is higher than algbw because all-reduce requires multiple transmissions of each data element across the ring (17.179869184 * 2 * 7/8 / 0.076026 s = 395.45 GB/s).

```
root@rs-mi350x8-mn-test1:~/rccl-tests# ./build/all_reduce_perf \
  -b 16G \
  -e 16G \
  -g 8
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 8 minBytes 17179869184 maxBytes 17179869184 step: 1048576(bytes) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version develop_deprecated:40b1b17
# Using devices
#  Rank  0 Group  0 Pid 370127 on rs-mi350x8-mn-test1 device  0 [0000:83:00] AMD Instinct MI350X VF
#  Rank  1 Group  0 Pid 370127 on rs-mi350x8-mn-test1 device  1 [0000:8b:00] AMD Instinct MI350X VF
#  Rank  2 Group  0 Pid 370127 on rs-mi350x8-mn-test1 device  2 [0000:93:00] AMD Instinct MI350X VF
#  Rank  3 Group  0 Pid 370127 on rs-mi350x8-mn-test1 device  3 [0000:9b:00] AMD Instinct MI350X VF
#  Rank  4 Group  0 Pid 370127 on rs-mi350x8-mn-test1 device  4 [0000:a3:00] AMD Instinct MI350X VF
#  Rank  5 Group  0 Pid 370127 on rs-mi350x8-mn-test1 device  5 [0000:ab:00] AMD Instinct MI350X VF
#  Rank  6 Group  0 Pid 370127 on rs-mi350x8-mn-test1 device  6 [0000:b3:00] AMD Instinct MI350X VF
#  Rank  7 Group  0 Pid 370127 on rs-mi350x8-mn-test1 device  7 [0000:bb:00] AMD Instinct MI350X VF
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong                               
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)                                      
 17179869184    4294967296     float     sum      -1    76026  225.97  395.45      0    76009  226.02  395.54      0
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 395.497 
#
# Collective test concluded: all_reduce_perf
```

The same test is repeated on `rs-mi350x8-mn-test2`, yielding nearly identical results. This consistency indicates stable and well-balanced intra-node communication over the XGMI interconnect.

```
root@rs-mi350x8-mn-test2:~/rccl-tests# ./build/all_reduce_perf \
  -b 16G \
  -e 16G \
  -g 8
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 8 minBytes 17179869184 maxBytes 17179869184 step: 1048576(bytes) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version develop_deprecated:40b1b17
# Using devices
#  Rank  0 Group  0 Pid 328125 on rs-mi350x8-mn-test2 device  0 [0000:83:00] AMD Instinct MI350X VF
#  Rank  1 Group  0 Pid 328125 on rs-mi350x8-mn-test2 device  1 [0000:8b:00] AMD Instinct MI350X VF
#  Rank  2 Group  0 Pid 328125 on rs-mi350x8-mn-test2 device  2 [0000:93:00] AMD Instinct MI350X VF
#  Rank  3 Group  0 Pid 328125 on rs-mi350x8-mn-test2 device  3 [0000:9b:00] AMD Instinct MI350X VF
#  Rank  4 Group  0 Pid 328125 on rs-mi350x8-mn-test2 device  4 [0000:a3:00] AMD Instinct MI350X VF
#  Rank  5 Group  0 Pid 328125 on rs-mi350x8-mn-test2 device  5 [0000:ab:00] AMD Instinct MI350X VF
#  Rank  6 Group  0 Pid 328125 on rs-mi350x8-mn-test2 device  6 [0000:b3:00] AMD Instinct MI350X VF
#  Rank  7 Group  0 Pid 328125 on rs-mi350x8-mn-test2 device  7 [0000:bb:00] AMD Instinct MI350X VF
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong                               
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)                                      
 17179869184    4294967296     float     sum      -1    75981  226.11  395.69      0    76226  225.38  394.42      0
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 395.053 
#
# Collective test concluded: all_reduce_perf
```

## Multi Node RCCL Testing

This test from `rs-mi350x8-mn-test1` evaluates a 16-GPU distributed all-reduce across two GPU droplets, each containing 8 AMD MI350X GPUs.

- Process orchestration: OpenMPI launches 16 ranks (1 rank per GPU) across the two nodes (8 ranks per node). OpenMPI is used strictly for process launch, environment propagation, and initial rank coordination. It communicates over TCP via the eth1 interface for bootstrap and rank placement only.
- Collective communication layer: All GPU-to-GPU collective operations are performed by RCCL and are independent of MPI once launched.
- Intra-node communication: Within each node, RCCL uses AMD XGMI for high-bandwidth GPU-to-GPU transfers, forming a fully connected 8-GPU local topology.
- Inter-node communication: The 16-GPU logical ring is stitched across nodes with exactly two inter-node ring edges. These cross-node hops use RDMA/RoCE over 8×400GbE NICs per node, enabling bandwidth aggregation through striping across multiple NICs. TCP is not used in the data path unless RDMA is unavailable.
- The two-node configuration shows a clear performance drop compared to the single-node case (algbw, 17.179869184 GB / 0.088766 s = 193.54 GB/s; busbw, 17.179869184 GB * 2 * 15/16 / 0.088766 s = 362.89 GB/s ). This degradation is primarily attributed to the added latency and lower efficiency of the RDMA-based inter-node hops relative to XGMI, despite NIC-level parallelism across 8×400GbE links.

```
root@rs-mi350x8-mn-test1:~/rccl-tests# mpirun --allow-run-as-root \
 --np 16 -H 10.120.0.9:8,10.120.0.10:8 \
  --bind-to none --mca btl self,tcp \
  --mca btl_tcp_if_include eth1 /root/rccl-tests/build/all_reduce_perf -b 16G -e 16G -g 1
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 17179869184 maxBytes 17179869184 step: 1048576(bytes) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version develop_deprecated:40b1b17
# Using devices
#  Rank  0 Group  0 Pid 371663 on rs-mi350x8-mn-test1 device  0 [0000:83:00] AMD Instinct MI350X VF
#  Rank  1 Group  0 Pid 371664 on rs-mi350x8-mn-test1 device  1 [0000:8b:00] AMD Instinct MI350X VF
#  Rank  2 Group  0 Pid 371665 on rs-mi350x8-mn-test1 device  2 [0000:93:00] AMD Instinct MI350X VF
#  Rank  3 Group  0 Pid 371666 on rs-mi350x8-mn-test1 device  3 [0000:9b:00] AMD Instinct MI350X VF
#  Rank  4 Group  0 Pid 371667 on rs-mi350x8-mn-test1 device  4 [0000:a3:00] AMD Instinct MI350X VF
#  Rank  5 Group  0 Pid 371668 on rs-mi350x8-mn-test1 device  5 [0000:ab:00] AMD Instinct MI350X VF
#  Rank  6 Group  0 Pid 371669 on rs-mi350x8-mn-test1 device  6 [0000:b3:00] AMD Instinct MI350X VF
#  Rank  7 Group  0 Pid 371671 on rs-mi350x8-mn-test1 device  7 [0000:bb:00] AMD Instinct MI350X VF
#  Rank  8 Group  0 Pid 330598 on rs-mi350x8-mn-test2 device  0 [0000:83:00] AMD Instinct MI350X VF
#  Rank  9 Group  0 Pid 330599 on rs-mi350x8-mn-test2 device  1 [0000:8b:00] AMD Instinct MI350X VF
#  Rank 10 Group  0 Pid 330600 on rs-mi350x8-mn-test2 device  2 [0000:93:00] AMD Instinct MI350X VF
#  Rank 11 Group  0 Pid 330601 on rs-mi350x8-mn-test2 device  3 [0000:9b:00] AMD Instinct MI350X VF
#  Rank 12 Group  0 Pid 330602 on rs-mi350x8-mn-test2 device  4 [0000:a3:00] AMD Instinct MI350X VF
#  Rank 13 Group  0 Pid 330603 on rs-mi350x8-mn-test2 device  5 [0000:ab:00] AMD Instinct MI350X VF
#  Rank 14 Group  0 Pid 330604 on rs-mi350x8-mn-test2 device  6 [0000:b3:00] AMD Instinct MI350X VF
#  Rank 15 Group  0 Pid 330605 on rs-mi350x8-mn-test2 device  7 [0000:bb:00] AMD Instinct MI350X VF
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong                               
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)                                      
 17179869184    4294967296     float     sum      -1    88766  193.54  362.89      0    88756  193.56  362.93      0
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 362.909 
#
# Collective test concluded: all_reduce_perf
```

This test from `rs-mi350x8-mn-test1` shows how RCCL all-reduce performance scales with message size on a 16-GPU, two-node setup. At small message sizes (e.g., 1–16 MB), performance is low and primarily limited by latency from kernel launches and inter-node communication. As message size increases, bandwidth improves rapidly, reflecting better utilization of the GPUs, XGMI, and RDMA interconnects. 

```
root@rs-mi350x8-mn-test1:~/rccl-tests#  mpirun --allow-run-as-root \
 --np 16 -H 10.120.0.9:8,10.120.0.10:8 \
  --bind-to none --mca btl self,tcp \
  --mca btl_tcp_if_include eth1 /root/rccl-tests/build/all_reduce_perf -b 1M -e 16G -f 2 -g 1
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 1048576 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version develop_deprecated:40b1b17
# Using devices
#  Rank  0 Group  0 Pid 374939 on rs-mi350x8-mn-test1 device  0 [0000:83:00] AMD Instinct MI350X VF
#  Rank  1 Group  0 Pid 374940 on rs-mi350x8-mn-test1 device  1 [0000:8b:00] AMD Instinct MI350X VF
#  Rank  2 Group  0 Pid 374941 on rs-mi350x8-mn-test1 device  2 [0000:93:00] AMD Instinct MI350X VF
#  Rank  3 Group  0 Pid 374942 on rs-mi350x8-mn-test1 device  3 [0000:9b:00] AMD Instinct MI350X VF
#  Rank  4 Group  0 Pid 374943 on rs-mi350x8-mn-test1 device  4 [0000:a3:00] AMD Instinct MI350X VF
#  Rank  5 Group  0 Pid 374944 on rs-mi350x8-mn-test1 device  5 [0000:ab:00] AMD Instinct MI350X VF
#  Rank  6 Group  0 Pid 374945 on rs-mi350x8-mn-test1 device  6 [0000:b3:00] AMD Instinct MI350X VF
#  Rank  7 Group  0 Pid 374946 on rs-mi350x8-mn-test1 device  7 [0000:bb:00] AMD Instinct MI350X VF
#  Rank  8 Group  0 Pid 333854 on rs-mi350x8-mn-test2 device  0 [0000:83:00] AMD Instinct MI350X VF
#  Rank  9 Group  0 Pid 333855 on rs-mi350x8-mn-test2 device  1 [0000:8b:00] AMD Instinct MI350X VF
#  Rank 10 Group  0 Pid 333856 on rs-mi350x8-mn-test2 device  2 [0000:93:00] AMD Instinct MI350X VF
#  Rank 11 Group  0 Pid 333857 on rs-mi350x8-mn-test2 device  3 [0000:9b:00] AMD Instinct MI350X VF
#  Rank 12 Group  0 Pid 333858 on rs-mi350x8-mn-test2 device  4 [0000:a3:00] AMD Instinct MI350X VF
#  Rank 13 Group  0 Pid 333859 on rs-mi350x8-mn-test2 device  5 [0000:ab:00] AMD Instinct MI350X VF
#  Rank 14 Group  0 Pid 333860 on rs-mi350x8-mn-test2 device  6 [0000:b3:00] AMD Instinct MI350X VF
#  Rank 15 Group  0 Pid 333861 on rs-mi350x8-mn-test2 device  7 [0000:bb:00] AMD Instinct MI350X VF
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong                               
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)                                      
     1048576        262144     float     sum      -1    81.69   12.84   24.07      0    80.37   13.05   24.46      0
     2097152        524288     float     sum      -1    99.36   21.11   39.57      0    99.03   21.18   39.71      0
     4194304       1048576     float     sum      -1    119.9   34.98   65.59      0    119.1   35.21   66.02      0
     8388608       2097152     float     sum      -1    167.4   50.10   93.93      0    167.8   49.99   93.74      0
    16777216       4194304     float     sum      -1    267.9   62.63  117.43      0    265.8   63.13  118.37      0
    33554432       8388608     float     sum      -1    405.6   82.73  155.11      0    410.4   81.77  153.31      0
    67108864      16777216     float     sum      -1    609.1  110.17  206.57      0    608.3  110.33  206.86      0
   134217728      33554432     float     sum      -1   1078.8  124.42  233.28      0   1067.7  125.71  235.71      0
   268435456      67108864     float     sum      -1   1468.5  182.79  342.73      0   1469.9  182.63  342.43      0
   536870912     134217728     float     sum      -1   2836.8  189.25  354.85      0   2837.5  189.21  354.76      0
  1073741824     268435456     float     sum      -1   5597.9  191.81  359.65      0   5594.8  191.92  359.85      0
  2147483648     536870912     float     sum      -1    11111  193.27  362.38      0    11112  193.25  362.35      0
  4294967296    1073741824     float     sum      -1    22150  193.90  363.56      0    22162  193.80  363.38      0
  8589934592    2147483648     float     sum      -1    44344  193.71  363.21      0    44355  193.66  363.12      0
 17179869184    4294967296     float     sum      -1    88655  193.78  363.34      0    88650  193.79  363.36      0
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 229.757 
#
# Collective test concluded: all_reduce_perf
```






