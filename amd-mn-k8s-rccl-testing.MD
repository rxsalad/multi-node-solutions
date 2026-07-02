# Multi-Node RCCL Testing for DOKS with AMD GPU Nodes

Follow [amd-mn-k8s-environment.MD](amd-mn-k8s-environment.MD) and [amd-mn-k8s-rdma-sharing.MD](amd-mn-k8s-rdma-sharing.MD) to prepare the environment before proceeding. The required networking components—including the Multus CNI and a secondary CNI (either [host-device CNI](amd-mn-k8s-environment.MD#summary--pod-gpu-rdma-and-network-setup-flow) or [ipvlan CNI](amd-mn-k8s-rdma-sharing.MD#summary--pod-gpu-rdma-and-network-setup-flow)), and the corresponding NetworkAttachmentDefinitions (NADs)—should already be installed and configured. Also, familiarize yourself with multi-node RCCL testing on GPU Droplets by reviewing [mi350x8-mn-rccl-testing.MD](mi350x8-mn-rccl-testing.MD).

This guide provides a step-by-step introduction to manually performing multi-node RCCL testing on DOKS clusters with AMD GPU nodes, covering both MI325x8 and MI350x8 platforms.

## Test Environment

DOKS Cluster – MI350x8:

``` bash
# kubectl get nodes -o wide
NAME                        STATUS   ROLES    AGE   VERSION   INTERNAL-IP   EXTERNAL-IP     OS-IMAGE                       KERNEL-VERSION        CONTAINER-RUNTIME
rs-mn-cpu-pool-381iam       Ready    <none>   19d   v1.34.1   10.120.0.6    167.71.156.33   Debian GNU/Linux 13 (trixie)   6.12.48+deb13-amd64   containerd://1.7.28
rs-mn-mi350x8-pool-3cmdp6   Ready    <none>   21h   v1.34.1   10.120.0.7    178.128.69.26   Debian GNU/Linux 13 (trixie)   6.12.48+deb13-amd64   containerd://1.7.28
rs-mn-mi350x8-pool-3cmdpa   Ready    <none>   21h   v1.34.1   10.120.0.8    104.248.71.70   Debian GNU/Linux 13 (trixie)   6.12.48+deb13-amd64   containerd://1.7.28
```

DOKS Cluster – MI325x8:

``` bash
# kubectl get nodes -o wide
NAME                        STATUS   ROLES    AGE     VERSION   INTERNAL-IP   EXTERNAL-IP       OS-IMAGE                       KERNEL-VERSION        CONTAINER-RUNTIME
rs-mn-cpu-pool-3c35j6       Ready    <none>   5d18h   v1.34.1   10.118.0.9    138.197.131.150   Debian GNU/Linux 13 (trixie)   6.12.48+deb13-amd64   containerd://1.7.28
rs-mn-mi325x8-pool-3c35jt   Ready    <none>   5d18h   v1.34.1   10.118.0.6    138.197.135.192   Debian GNU/Linux 13 (trixie)   6.12.48+deb13-amd64   containerd://1.7.28
rs-mn-mi325x8-pool-3cmiju   Ready    <none>   2d4h    v1.34.1   10.118.0.7    167.99.176.197    Debian GNU/Linux 13 (trixie)   6.12.48+deb13-amd64   containerd://1.7.28
```

## Multi Node RCCL Testing  

### Overview

To run RCCL tests in a multi-node GPU droplet environment, each node must have **ROCm/RCCL**, **rccl-tests**, **OpenMPI** and **SSH key-based authentication** configured.

OpenMPI uses SSH to launch the [all_reduce_perf](mi350x8-mn-rccl-testing.MD#all-reduce-concept) benchmark, running 8 ranks per node. Its role is limited to process startup, environment propagation, and initial rank coordination over the **eth1** interface (vpc) via TCP.

After initialization, all collective GPU communication is handled by RCCL:

- Intra-node communication: RCCL uses **AMD XGMI** to provide high-bandwidth GPU-to-GPU communication across the 8-GPU topology.
- Inter-node communication: RCCL uses **RDMA/RoCE** over 8x400 GbE NICs, leveraging multi-rail networking to distribute traffic across multiple interfaces and maximize bandwidth.

TCP is used only for OpenMPI bootstrap and is not part of the collective data path unless RDMA is unavailable.

Refer to [mi350x8-mn-rccl-testing.MD](mi350x8-mn-rccl-testing.MD) for additional details on running RCCL tests in the multi-node GPU Droplet environment.

To run RCCL tests on Kubernetes, workloads are typically executed inside pods (containers) rather than directly on the host. Each pod must be granted access to GPUs, RDMA devices, and the associated network interfaces, and the container image must include the required runtime components, such as ROCm/RCCL, rccl-tests, OpenMPI, and SSH key-based authentication for MPI process launch.

OpenMPI uses the pod's `eth0` interface for initial rank discovery, coordination, and process startup across pods. Once the MPI job is established, the underlying GPU communication path remains unchanged: intra-node communication uses AMD XGMI, while inter-node communication uses RDMA over RoCE.

In a Kubernetes environment, additional tooling such as the MPI Operator is commonly used to simplify the deployment, orchestration, and lifecycle management of multi-node MPI workloads. The MPI Operator automates tasks such as pod creation, hostfile generation, SSH configuration, and MPI job execution, making it easier to run distributed RCCL tests at scale.

### Container Image

Refer to [amd-mn-doks/Dockerfile.simple](amd-mn-doks/Dockerfile.simple) for a reference implementations of the container image, which includes the following components:

| Component                       | Version                   | Purpose                                                   |
| ------------------------------- | ------------------------- | --------------------------------------------------------- |
| ROCm Base Image                 | 7.2.1                     | AMD GPU runtime and development environment               |
| RCCL                            | 2.27.7 (via `rocm-7.2.1`) | AMD collective communication library                      |
| RCCL-Tests                      | `therock-7.11`            | Benchmark suite for RCCL collectives                      |
| Open MPI                        | 4.1.6                     | MPI implementation                                        |
| CMake                           | 3.28.0                    | Build system                                              |
| Pollara Userspace RDMA Provider | 1.117.1-a-63              | Required userspace RDMA provider for AMD Pensando Pollara NICs on MI350x8 |
|  AMD GPU Targets | gfx950, gfx942             | RCCL and RCCL-Tests are compiled only for gfx950 (MI350x8) and gfx942 (MI325x8) |



Build and push the container images:

``` bash
docker image build -t docker.io/richardxgf/amd:rccl-test-simple -f Dockerfile.simple . 
docker push docker.io/richardxgf/amd:rccl-test-simple
```

###  MPI Operator

A [Kubernetes Operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) is an extension of the Kubernetes control plane that builds on the controller pattern to automate the full lifecycle management of complex, stateful applications. It introduces domain-specific APIs through Custom Resource Definitions (CRDs) and is implemented as a custom [controller](https://kubernetes.io/docs/concepts/architecture/controller/) that runs a reconciliation loop, continuously comparing the desired state defined by the user with the actual state in the cluster. When drift is detected, the controller takes corrective actions to restore the desired state. By embedding operational knowledge into software, Operators make it easier to manage distributed systems such as databases, AI training workloads, and HPC applications.

The [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) is a specialized Kubernetes Operator designed to run MPI workloads in a cloud-native way. It exposes a custom resource called MPIJob, allowing users to declaratively define distributed jobs. The operator then automates the creation, configuration, and coordination of all required Kubernetes resources needed for multi-node execution, making it well-suited for large-scale HPC workloads and AI training scenarios.

Installing the MPI Operator deploys a controller that manages distributed MPI workloads. These typically include:

- A dedicated namespace for resource isolation
- A MPIJob CRD that defines MPI workloads declaratively
- Service accounts and RBAC rules that enable the controller to manage cluster resources securely
- A controller deployment that watches MPIJob resources and reconciles them by creating and managing underlying Kubernetes objects

Together, these components extend Kubernetes so that MPI applications can be launched and managed without manually creating pods, configuring networking, or setting up SSH-based clusters.

```
# kubectl apply -f https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.4.0/deploy/v2beta1/mpi-operator.yaml
namespace/mpi-operator created
customresourcedefinition.apiextensions.k8s.io/mpijobs.kubeflow.org created
serviceaccount/mpi-operator created
clusterrole.rbac.authorization.k8s.io/kubeflow-mpijobs-admin created
clusterrole.rbac.authorization.k8s.io/kubeflow-mpijobs-edit created
clusterrole.rbac.authorization.k8s.io/kubeflow-mpijobs-view created
clusterrole.rbac.authorization.k8s.io/mpi-operator created
clusterrolebinding.rbac.authorization.k8s.io/mpi-operator created
deployment.apps/mpi-operator created

# kubectl get pods -A | grep mpi
mpi-operator   mpi-operator-567c4f8987-227xs                     1/1     Running   0          65m
```

At runtime, the MPI Operator translates a single MPIJob specification into a coordinated distributed execution environment. When a job is submitted, the operator creates **a launcher pod** and **the required number of worker pods**. It automatically configures networking and DNS resolution between pods, generates SSH keys for secure inter-pod communication, and constructs an MPI hostfile describing all participating nodes.

The launcher pod then invokes mpirun, which uses this generated environment to start and coordinate MPI ranks across all worker pods. After execution begins, the operator shifts into a supervisory role—monitoring job progress, tracking completion status, handling failures if needed, and cleaning up all associated resources once the job finishes.

### MPIJob Manifest

If your DOKS cluster is configured with **Multus CNI + host-device CNI**, refer to [amd-mn-doks/rccl-test-2-nodes-mi350x8.yaml](amd-mn-doks/rccl-test-2-nodes-mi350x8.yaml) and [amd-mn-doks/rccl-test-2-nodes-mi325x8.yaml](amd-mn-doks/rccl-test-2-nodes-mi325x8.yaml) for example MPIJob manifests.

if the cluster is configured with **Multus CNI + ipvlan CNI**, refer to [rdma-sharing/rccl-test-2-nodes-mi350x8-ipvlan.yaml](rdma-sharing/rccl-test-2-nodes-mi350x8-ipvlan.yaml) and [rdma-sharing/rccl-test-2-nodes-mi325x8-ipvlan.yaml](rdma-sharing/rccl-test-2-nodes-mi325x8-ipvlan.yaml) for the corresponding MPIJob manifests.

The primary differences between these manifests are **the Multus CNI network attachment annotations** and **the value of the NCCL_IB_GID_INDEX environment variable**. The appropriate NCCL_IB_GID_INDEX value depends on the underlying RDMA/RoCE NIC hardware, driver implementation, and network CNI configuration.

Instead of explicitly specifying a GID index, setting **NCCL_IB_GID_INDEX=-1** allows NCCL/RCCL to automatically select an appropriate available GID. This auto-selection capability is supported by NCCL/RCCL, and its reliability and compatibility across different NIC drivers have improved in newer releases. For DOKS clusters, NCCL_IB_GID_INDEX=-1 is generally the recommended default, while a fixed GID index should be used only for troubleshooting or when required by a specific RDMA/RoCE environment.

These MPIJobs runs the RCCL `all_reduce_perf` benchmark across two worker nodes (MI325x8 or MI350x8), each with 8 GPUs, for a total of 16 GPUs. The job is managed by the Kubeflow MPI Operator and validates distributed GPU communication over XGMI (intra-node) and RoCE RDMA (inter-node).

Notes: In an MPIJob, **slotsPerWorker**, **worker replicas**, and **mpirun -np** must be aligned to ensure correct GPU mapping and full utilization. Since each node has 8 GPUs, each worker Pod exposes 8 slots, enabling a one-to-one mapping between MPI ranks and GPUs. Therefore, -np should equal worker replicas × 8, ensuring each rank is bound to a dedicated GPU across all Pods. This alignment provides a clean one-rank-per-GPU execution model for predictable RCCL performance, while misconfiguration can lead to underutilization or slot mismatch errors.

**Key components:**

- Launcher pod: Runs `mpirun` and orchestrates 16 MPI ranks across the worker nodes. It uses OpenMPI’s SSH-based launcher to remotely start processes on workers.
- Worker pods: Each requests 8 GPUs and 8 RDMA interfaces, and runs an sshd daemon (port 2222) that enables remote process execution from the Launcher via SSH.
- Multus network attachments: Attach fabric0–fabric7 RoCE interfaces to each worker pod, enabling multi-rail RDMA communication for RCCL traffic.
- RCCL configuration: Uses NCCL/RCCL environment variables to tune communication behavior over RoCE, ensuring correct NIC selection, transport behavior, and performance optimization.
- Shared memory volume: Provides a 16 GiB /dev/shm mount to support RCCL buffers and high-throughput intra-node communication.
- Health probes: Validate SSH service readiness (port 2222) before MPI execution begins.

OpenMPI’s mpirun uses SSH as the remote execution mechanism, connecting from the Launcher pod to each Worker pod’s sshd service on port 2222 over Kubernetes pod networking. Authentication is handled via SSH key pairs injected by the MPI Operator, enabling passwordless trust. Once connected, mpirun uses these SSH sessions to launch MPI ranks on the Worker pods, which then execute the RCCL all_reduce_perf benchmark across GPUs distributed over XGMI and RoCE fabrics.

### Test Execution 

<details>
<summary>MI350x8</summary>

``` bash
# Submit MPIJob (RCCL multi-node test workload)
# kubectl apply -f amd-mn-doks/rccl-test-2-nodes-mi350x8.yaml
mpijob.kubeflow.org/mpi-multus-gfx950 created

# Check the MPI operator's logs
# kubectl -n mpi-operator logs deployment/mpi-operator

# Verify pods created by MPI Operator: 1 launcher pod = MPI master, (N) worker pods = MPI ranks
# kubectl get pods -o wide
NAME                               READY   STATUS      RESTARTS   AGE   IP             NODE                        NOMINATED NODE   READINESS GATES
mpi-multus-gfx950-launcher-fpbmh   1/1     Running.    0          51s   10.121.0.98    rs-mn-cpu-pool-381iam       <none>           <none>
mpi-multus-gfx950-worker-0         1/1     Running     0          51s   10.121.0.242   rs-mn-mi350x8-pool-3cmdp6   <none>           <none>
mpi-multus-gfx950-worker-1         1/1     Running     0          51s   10.121.1.57    rs-mn-mi350x8-pool-3cmdpa   <none>           <none>

# Check MPIJob status
# kubectl get mpijob
NAME                AGE
mpi-multus-gfx950   69s

# Detailed MPIJob inspection (recommended for debugging)

# kubectl describe mpijob mpi-multus-gfx950

# kubectl logs mpi-multus-gfx950-worker-0 -c mpi-worker
# kubectl logs mpi-multus-gfx950-worker-1 -c mpi-worker

# kubectl exec -it mpi-multus-gfx950-worker-0 -- /bin/bash
# kubectl exec -it mpi-multus-gfx950-worker-1 -- /bin/bash
```

``` bash
# The MPI launcher has finished execution
# kubectl get pods -o wide
NAME                               READY   STATUS      RESTARTS   AGE   IP             NODE                        NOMINATED NODE   READINESS GATES
mpi-multus-gfx950-launcher-fpbmh   0/1     Completed   0          15m   10.121.0.98    rs-mn-cpu-pool-381iam       <none>           <none>
mpi-multus-gfx950-worker-0         1/1     Running     0          15m   10.121.0.242   rs-mn-mi350x8-pool-3cmdp6   <none>           <none>
mpi-multus-gfx950-worker-1         1/1     Running     0          15m   10.121.1.57    rs-mn-mi350x8-pool-3cmdpa   <none>           <none>

# kubectl logs -f mpi-multus-gfx950-launcher-fpbmh 
Warning: Permanently added '[mpi-multus-gfx950-worker-0.mpi-multus-gfx950.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-gfx950-worker-1.mpi-multus-gfx950.default.svc]:2222' (ED25519) to the list of known hosts.
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 100 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version HEAD:5272cd1
# Using devices
#  Rank  0 Group  0 Pid    251 on mpi-multus-gfx950-worker-0 device  0 [0000:83:00] 
#  Rank  1 Group  0 Pid    252 on mpi-multus-gfx950-worker-0 device  1 [0000:8b:00] 
#  Rank  2 Group  0 Pid    253 on mpi-multus-gfx950-worker-0 device  2 [0000:93:00] 
#  Rank  3 Group  0 Pid    254 on mpi-multus-gfx950-worker-0 device  3 [0000:9b:00] 
#  Rank  4 Group  0 Pid    255 on mpi-multus-gfx950-worker-0 device  4 [0000:a3:00] 
#  Rank  5 Group  0 Pid    256 on mpi-multus-gfx950-worker-0 device  5 [0000:ab:00] 
#  Rank  6 Group  0 Pid    257 on mpi-multus-gfx950-worker-0 device  6 [0000:b3:00] 
#  Rank  7 Group  0 Pid    258 on mpi-multus-gfx950-worker-0 device  7 [0000:bb:00] 
#  Rank  8 Group  0 Pid    251 on mpi-multus-gfx950-worker-1 device  0 [0000:83:00] 
#  Rank  9 Group  0 Pid    252 on mpi-multus-gfx950-worker-1 device  1 [0000:8b:00] 
#  Rank 10 Group  0 Pid    253 on mpi-multus-gfx950-worker-1 device  2 [0000:93:00] 
#  Rank 11 Group  0 Pid    254 on mpi-multus-gfx950-worker-1 device  3 [0000:9b:00] 
#  Rank 12 Group  0 Pid    255 on mpi-multus-gfx950-worker-1 device  4 [0000:a3:00] 
#  Rank 13 Group  0 Pid    256 on mpi-multus-gfx950-worker-1 device  5 [0000:ab:00] 
#  Rank 14 Group  0 Pid    257 on mpi-multus-gfx950-worker-1 device  6 [0000:b3:00] 
#  Rank 15 Group  0 Pid    258 on mpi-multus-gfx950-worker-1 device  7 [0000:bb:00] 
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong                               
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)                                      
           8             2     float     sum      -1    31.40    0.00    0.00      0    31.11    0.00    0.00      0
          16             4     float     sum      -1    31.02    0.00    0.00      0    31.08    0.00    0.00      0
          32             8     float     sum      -1    31.97    0.00    0.00      0    30.97    0.00    0.00      0
          64            16     float     sum      -1    31.26    0.00    0.00      0    31.37    0.00    0.00      0
         128            32     float     sum      -1    31.41    0.00    0.01      0    32.73    0.00    0.01      0
         256            64     float     sum      -1    32.04    0.01    0.01      0    31.66    0.01    0.02      0
         512           128     float     sum      -1    32.49    0.02    0.03      0    32.12    0.02    0.03      0
        1024           256     float     sum      -1    32.88    0.03    0.06      0    32.96    0.03    0.06      0
        2048           512     float     sum      -1    35.64    0.06    0.11      0    35.72    0.06    0.11      0
        4096          1024     float     sum      -1    36.57    0.11    0.21      0    36.28    0.11    0.21      0
        8192          2048     float     sum      -1    36.09    0.23    0.43      0    36.22    0.23    0.42      0
       16384          4096     float     sum      -1    37.22    0.44    0.83      0    37.12    0.44    0.83      0
       32768          8192     float     sum      -1    39.08    0.84    1.57      0    39.05    0.84    1.57      0
       65536         16384     float     sum      -1    40.80    1.61    3.01      0    41.35    1.59    2.97      0
      131072         32768     float     sum      -1    44.19    2.97    5.56      0    43.73    3.00    5.62      0
      262144         65536     float     sum      -1    51.71    5.07    9.51      0    51.53    5.09    9.54      0
      524288        131072     float     sum      -1    55.51    9.45   17.71      0    55.37    9.47   17.75      0
     1048576        262144     float     sum      -1    60.96   17.20   32.25      0    61.29   17.11   32.08      0
     2097152        524288     float     sum      -1    69.29   30.27   56.75      0    69.99   29.96   56.18      0
     4194304       1048576     float     sum      -1    80.93   51.82   97.17      0    81.24   51.63   96.80      0
     8388608       2097152     float     sum      -1    125.5   66.83  125.31      0    104.0   80.66  151.23      0
    16777216       4194304     float     sum      -1    157.5  106.54  199.76      0    156.7  107.07  200.75      0
    33554432       8388608     float     sum      -1    261.1  128.52  240.98      0    261.7  128.21  240.39      0
    67108864      16777216     float     sum      -1    420.1  159.76  299.55      0    420.3  159.65  299.35      0
   134217728      33554432     float     sum      -1   1025.8  130.84  245.32      0   1026.8  130.72  245.09      0
   268435456      67108864     float     sum      -1   1366.9  196.38  368.22      0   1366.6  196.42  368.29      0
   536870912     134217728     float     sum      -1   2672.3  200.90  376.69      0   2669.7  201.10  377.05      0
  1073741824     268435456     float     sum      -1   5323.3  201.71  378.20      0   5310.5  202.19  379.11      0
  2147483648     536870912     float     sum      -1    10658  201.49  377.79      0    10655  201.54  377.89      0
  4294967296    1073741824     float     sum      -1    21180  202.78  380.22      0    21202  202.57  379.82      0
  8589934592    2147483648     float     sum      -1    42425  202.47  379.64      0    42358  202.79  380.24      0
 17179869184    4294967296     float     sum      -1    84728  202.76  380.18      0    84690  202.86  380.36      0
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 124.701 
#
# Collective test concluded: all_reduce_perf

# Delete MPIJob (cleanup) - This removes all resources created by the MPI Operator:
# - Launcher pod (MPI master)
# - Worker pods (MPI ranks)
# - Associated MPI runtime resources

# kubectl delete mpijob mpi-multus-gfx950
mpijob.kubeflow.org "mpi-multus-gfx950" deleted from default namespace
```

</details>
</br>

<details>
<summary>MI325x8</summary>

``` bash
# Submit MPIJob (RCCL multi-node test workload)
# kubectl apply -f amd-mn-doks/rccl-test-2-nodes-mi325x8.yaml
mpijob.kubeflow.org/mpi-multus-gfx942 created

# Check the MPI operator's logs
# kubectl -n mpi-operator logs deployment/mpi-operator

# Verify pods created by MPI Operator: 1 launcher pod = MPI master, (N) worker pods = MPI ranks
# kubectl get pods -o wide
NAME                               READY   STATUS    RESTARTS   AGE   IP             NODE                        NOMINATED NODE   READINESS GATES
mpi-multus-gfx942-launcher-fdxv9   1/1     Running   0          27s   10.119.0.14    rs-mn-cpu-pool-3c35j6       <none>           <none>
mpi-multus-gfx942-worker-0         1/1     Running   0          27s   10.119.1.56    rs-mn-mi325x8-pool-3cmiju   <none>           <none>
mpi-multus-gfx942-worker-1         1/1     Running   0          27s   10.119.0.235   rs-mn-mi325x8-pool-3c35jt   <none>           <none>

# Check MPIJob status
# kubectl get mpijob
NAME                AGE
mpi-multus-gfx942   53s

# Detailed MPIJob inspection (recommended for debugging)

# kubectl describe mpijob mpi-multus-gfx942

# kubectl logs mpi-multus-gfx942-worker-0 -c mpi-worker
# kubectl logs mpi-multus-gfx942-worker-1 -c mpi-worker

# kubectl exec -it mpi-multus-gfx942-worker-0 -- /bin/bash
# kubectl exec -it mpi-multus-gfx942-worker-1 -- /bin/bash
```

``` bash
# The MPI launcher has finished execution
# kubectl get pods -o wide
NAME                               READY   STATUS      RESTARTS   AGE     IP             NODE                        NOMINATED NODE   READINESS GATES
mpi-multus-gfx942-launcher-fdxv9   0/1     Completed   0          5m37s   10.119.0.14    rs-mn-cpu-pool-3c35j6       <none>           <none>
mpi-multus-gfx942-worker-0         1/1     Running     0          5m37s   10.119.1.56    rs-mn-mi325x8-pool-3cmiju   <none>           <none>
mpi-multus-gfx942-worker-1         1/1     Running     0          5m37s   10.119.0.235   rs-mn-mi325x8-pool-3c35jt   <none>           <none>

# kubectl logs -f mpi-multus-gfx942-launcher-fdxv9
Warning: Permanently added '[mpi-multus-gfx942-worker-1.mpi-multus-gfx942.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-gfx942-worker-0.mpi-multus-gfx942.default.svc]:2222' (ED25519) to the list of known hosts.
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 100 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version HEAD:5272cd1
# Using devices
#  Rank  0 Group  0 Pid    134 on mpi-multus-gfx942-worker-0 device  0 [0000:83:00] 
#  Rank  1 Group  0 Pid    135 on mpi-multus-gfx942-worker-0 device  1 [0000:8b:00] 
#  Rank  2 Group  0 Pid    136 on mpi-multus-gfx942-worker-0 device  2 [0000:93:00] 
#  Rank  3 Group  0 Pid    137 on mpi-multus-gfx942-worker-0 device  3 [0000:9b:00] 
#  Rank  4 Group  0 Pid    138 on mpi-multus-gfx942-worker-0 device  4 [0000:a3:00] 
#  Rank  5 Group  0 Pid    139 on mpi-multus-gfx942-worker-0 device  5 [0000:ab:00] 
#  Rank  6 Group  0 Pid    140 on mpi-multus-gfx942-worker-0 device  6 [0000:b3:00] 
#  Rank  7 Group  0 Pid    141 on mpi-multus-gfx942-worker-0 device  7 [0000:bb:00] 
#  Rank  8 Group  0 Pid    118 on mpi-multus-gfx942-worker-1 device  0 [0000:83:00] 
#  Rank  9 Group  0 Pid    119 on mpi-multus-gfx942-worker-1 device  1 [0000:8b:00] 
#  Rank 10 Group  0 Pid    120 on mpi-multus-gfx942-worker-1 device  2 [0000:93:00] 
#  Rank 11 Group  0 Pid    121 on mpi-multus-gfx942-worker-1 device  3 [0000:9b:00] 
#  Rank 12 Group  0 Pid    122 on mpi-multus-gfx942-worker-1 device  4 [0000:a3:00] 
#  Rank 13 Group  0 Pid    123 on mpi-multus-gfx942-worker-1 device  5 [0000:ab:00] 
#  Rank 14 Group  0 Pid    124 on mpi-multus-gfx942-worker-1 device  6 [0000:b3:00] 
#  Rank 15 Group  0 Pid    125 on mpi-multus-gfx942-worker-1 device  7 [0000:bb:00] 
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong                               
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)                                      
           8             2     float     sum      -1    29.59    0.00    0.00      0    29.82    0.00    0.00      0
          16             4     float     sum      -1    29.56    0.00    0.00      0    29.56    0.00    0.00      0
          32             8     float     sum      -1    28.59    0.00    0.00      0    28.06    0.00    0.00      0
          64            16     float     sum      -1    28.11    0.00    0.00      0    28.87    0.00    0.00      0
         128            32     float     sum      -1    29.58    0.00    0.01      0    28.60    0.00    0.01      0
         256            64     float     sum      -1    28.96    0.01    0.02      0    31.03    0.01    0.02      0
         512           128     float     sum      -1    29.01    0.02    0.03      0    29.08    0.02    0.03      0
        1024           256     float     sum      -1    30.01    0.03    0.06      0    29.86    0.03    0.06      0
        2048           512     float     sum      -1    32.20    0.06    0.12      0    32.33    0.06    0.12      0
        4096          1024     float     sum      -1    33.49    0.12    0.23      0    33.17    0.12    0.23      0
        8192          2048     float     sum      -1    33.45    0.24    0.46      0    33.34    0.25    0.46      0
       16384          4096     float     sum      -1    33.53    0.49    0.92      0    33.51    0.49    0.92      0
       32768          8192     float     sum      -1    33.56    0.98    1.83      0    33.59    0.98    1.83      0
       65536         16384     float     sum      -1    36.13    1.81    3.40      0    35.19    1.86    3.49      0
      131072         32768     float     sum      -1    37.85    3.46    6.49      0    36.88    3.55    6.66      0
      262144         65536     float     sum      -1    44.52    5.89   11.04      0    44.24    5.93   11.11      0
      524288        131072     float     sum      -1    56.71    9.24   17.33      0    57.43    9.13   17.12      0
     1048576        262144     float     sum      -1    53.24   19.70   36.93      0    53.19   19.71   36.96      0
     2097152        524288     float     sum      -1    62.56   33.52   62.86      0    61.59   34.05   63.84      0
     4194304       1048576     float     sum      -1    73.08   57.40  107.62      0    74.08   56.62  106.16      0
     8388608       2097152     float     sum      -1    189.1   44.35   83.16      0    191.0   43.92   82.35      0
    16777216       4194304     float     sum      -1    252.9   66.34  124.38      0    261.9   64.07  120.13      0
    33554432       8388608     float     sum      -1    413.2   81.21  152.27      0    394.8   84.98  159.34      0
    67108864      16777216     float     sum      -1    601.4  111.59  209.23      0    603.4  111.21  208.52      0
   134217728      33554432     float     sum      -1    840.1  159.76  299.55      0    836.1  160.53  301.00      0
   268435456      67108864     float     sum      -1   1469.5  182.67  342.51      0   1463.6  183.40  343.88      0
   536870912     134217728     float     sum      -1   2859.9  187.72  351.98      0   2846.9  188.58  353.59      0
  1073741824     268435456     float     sum      -1   5692.7  188.62  353.66      0   5699.2  188.40  353.25      0
  2147483648     536870912     float     sum      -1    11405  188.29  353.05      0    11437  187.76  352.05      0
  4294967296    1073741824     float     sum      -1    22652  189.60  355.51      0    22762  188.69  353.80      0
  8589934592    2147483648     float     sum      -1    45295  189.64  355.58      0    45706  187.94  352.38      0
 17179869184    4294967296     float     sum      -1    90004  190.88  357.90      0    90118  190.64  357.45      0
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 112.108 
#
# Collective test concluded: all_reduce_perf

# Delete MPIJob (cleanup) - This removes all resources created by the MPI Operator:
# - Launcher pod (MPI master)
# - Worker pods (MPI ranks)
# - Associated MPI runtime resources

# kubectl delete mpijob mpi-multus-gfx942
mpijob.kubeflow.org "mpi-multus-gfx942" deleted from default namespace
```

</details>


