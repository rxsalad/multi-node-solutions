# Multi-Node PyTorch Testing for MI350x8 GPU Droplets

Follow [the guide](mi350x8-mn-environment.MD) to prepare the environment before proceeding.

This guide provides a basic introduction and step-by-step instructions for manually performing PyTorch testing in a multi-node MI350x8 GPU droplet environment.

The equivalent configuration for MI325x8 nodes, along with the key differences, is also provided.

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
| eth1-vpc       | 10.120.0.6/20  | 10.120.0.7/20 | 
| eth2-roce      | 192.168.52.101/24 | 192.168.52.102/24  | 
| eth3-roce      | 192.168.53.101/24 | 192.168.53.102/24  | 
| eth4-roce      | 192.168.54.101/24 | 192.168.54.102/24  | 
| eth5-roce      | 192.168.55.101/24 | 192.168.55.102/24  |
| eth6-roce      | 192.168.56.101/24 | 192.168.56.102/24  | 
| eth7-roce      | 192.168.57.101/24 | 192.168.57.102/24  | 
| eth8-roce      | 192.168.58.101/24 | 192.168.58.102/24  | 
| eth9-roce      | 192.168.59.101/24 | 192.168.59.102/24  | 


Check the GPU driver and ROCm stack on the GPU droplets:

``` shell
root@rs-mi350x8-mn-test1:~# amd-smi version
AMDSMI Tool: 26.0.2+39589fda | AMDSMI Library version: 26.0.2 | ROCm version: 7.0.2 | amdgpu version: 6.14.14 | amd_hsmp version: N/A

root@rs-mi350x8-mn-test1:~# cat /opt/rocm/.info/version
7.0.2

root@rs-mi350x8-mn-test1:~# ls /opt/rocm/lib | grep rccl
librccl.so
librccl.so.1
librccl.so.1.0.70002

root@rs-mi350x8-mn-test1:~# uname -r
6.8.0-106-generic

root@rs-mi350x8-mn-test1:~# dkms status
amdgpu/6.14.14-2226257.24.04, 6.8.0-117-generic, x86_64: installed
ionic/25.08.4.004, 6.8.0-117-generic, x86_64: installed
mlnx-ofed-kernel/24.10.OFED.24.10.3.2.5.1, 6.8.0-117-generic, x86_64: installed
```

## Concepts

ROCm is AMD’s GPU computing software stack. It includes kernel-level GPU drivers, user-space runtime libraries, [the HIP programming model](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html), performance libraries such as rocBLAS and MIOpen for linear algebra and deep learning workloads, and RCCL for distributed communication. Together, these components provide the core software foundation that enables AMD GPUs to efficiently run machine learning and high-performance computing workloads.

When [installing PyTorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html) with a ROCm index on the host, such as pip install torch --index-url https://download.pytorch.org/whl/rocm7.0, you obtain a PyTorch build compiled for AMD GPUs via the HIP compiler. It exposes the standard PyTorch API, but at runtime it depends on the system’s **ROCm user-space libraries**, including components such as the HIP runtime, rocBLAS, MIOpen, and RCCL. The wheel does not ship a full ROCm stack; instead, it is built against a specific ROCm version and **dynamically links** to compatible ROCm libraries already present on the host.

More generally, the PyTorch ROCm wheel is an application-layer build that integrates with the system ROCm installation rather than replacing it. This means compatibility is primarily determined by **version alignment** between the wheel and the host-installed ROCm user-space stack.

[A PyTorch ROCm container](https://hub.docker.com/r/rocm/pytorch) takes this further by bundling most ROCm user-space components (HIP, rocBLAS, MIOpen, RCCL, etc.) together with PyTorch inside the image. However, it still relies on the host for the kernel-level AMD GPU driver and device interfaces (e.g., `/dev/kfd`, `/dev/dri`, `/dev/infiniBand`). This approach reduces dependency drift and version mismatches, while keeping low-level hardware control on the host system.

It is important to distinguish between **host-reported ROCm versions** and **the user-space ROCm stack used inside containers**. Tools such as amd-smi report the ROCm user-space components installed on the host system, which may differ from what a container actually uses at runtime. In ROCm container environments (e.g., PyTorch or vLLM images), the user-space ROCm libraries are typically provided by the container image itself, while the kernel-level driver stack is always supplied by the host.

As a result, the effective ROCm version inside a container can differ from the host’s installed version. For example, a container may include a newer ROCm user-space stack (e.g., `7.2.2`) even if the host’s user-space installation is older (e.g., `7.0.2`), provided the host kernel driver remains compatible.


## Two Main Approaches for Running AI/ML Workloads

### Direct Host Execution

In this approach, applications (e.g., PyTorch or vLLM) run directly on the host operating system using the ROCm stack installed on the system. Both the user-space libraries and runtime components are provided by the host installation.

This model offers tighter integration with the system but increases the risk of version conflicts between applications and the installed ROCm stack, especially when multiple workloads require different ROCm versions.

Install PyTorch on the GPU droplets following the instructions in [the link](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#use-a-wheels-package), and then verify GPU access, ROCm/PyTorch runtime, RCCL backend, and Python version on the host:


``` shell
root@rs-mi350x8-mn-test1:~# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0  --break-system-packages

root@rs-mi350x8-mn-test1:~# python3 -c "import torch; print(torch.cuda.is_available())"
True

root@rs-mi350x8-mn-test1:~# python3 -c "import torch; print(torch.__version__)"
2.10.0+rocm7.0

root@rs-mi350x8-mn-test1:~# python3 -c "import torch; print(torch.distributed.is_nccl_available())"
True

root@rs-mi350x8-mn-test1:~# python3 -c "import torch; print(torch.cuda.nccl.version())"
(2, 26, 6)

root@rs-mi350x8-mn-test1:~# python3 --version
Python 3.12.3
```

### Containers

In contrast, in the container-based approach, applications run inside a container that bundles most ROCm user-space components. This provides a consistent, reproducible software environment and helps reduce dependency and version mismatches across systems.

However, the container still relies on the host system for the kernel-level AMD GPU driver and device interfaces. This separation allows portability of user-space software while keeping low-level GPU control on the host.

Install Docker on the droplets by following the instructions in [the link](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository):

``` shell
root@rs-mi350x8-mn-test1:~# systemctl status docker
● docker.service - Docker Application Container Engine
     Loaded: loaded (/usr/lib/systemd/system/docker.service; enabled; preset: enabled)
     Active: active (running) since Sun 2026-05-24 19:30:39 UTC; 9s ago
TriggeredBy: ● docker.socket
       Docs: https://docs.docker.com
   Main PID: 330172 (dockerd)
      Tasks: 60
     Memory: 47.3M (peak: 49.4M)
        CPU: 240ms
     CGroup: /system.slice/docker.service
             └─330172 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
```

To run ROCm containers, follow [this guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html). Start a container and verify GPU access, the ROCm/PyTorch runtime, the RCCL backend, and the Python version inside the container.


``` shell
# Host-reported ROCm version: 7.0.2
root@rs-mi350x8-mn-test2:~# amd-smi version
AMDSMI Tool: 26.0.2+39589fda | AMDSMI Library version: 26.0.2 | ROCm version: 7.0.2 | amdgpu version: 6.14.14 | amd_hsmp version: N/A

root@rs-mi350x8-mn-test1:~# docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1 \
  /bin/bash

# Container-reported ROCm version: 7.2.2
root@69f1f41f3f76:/# amd-smi version
AMDSMI Tool: 26.2.2+671d39a71e | AMDSMI Library version: 26.2.2 | ROCm version: 7.2.2 | amdgpu version: 6.14.14 | hsmp version: N/A

root@69f1f41f3f76:/# python3 -c "import torch; print(torch.cuda.is_available())"
True

root@69f1f41f3f76:/# python3 -c "import torch; print(torch.__version__)"
2.7.1+rocm7.2.2.git1dab218d

# PyTorch uses the NCCL-compatible API naming for both NVIDIA NCCL and AMD RCCL
root@69f1f41f3f76:/# python3 -c "import torch; print(torch.distributed.is_nccl_available())"
True

root@69f1f41f3f76:/# python3 -c "import torch; print(torch.cuda.nccl.version())"
(2, 27, 7)

root@69f1f41f3f76:/# python --version
Python 3.12.3
```

The above method explicitly passes the required GPU devices (`/dev/kfd`, `/dev/dri`) into the container and manually relaxes permissions and security settings, enabling applications inside the container to access the underlying hardware. It works with standard Docker but is more verbose and requires the user to know and specify all necessary flags. We can also install and use [AMD Container Tookit](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/overview.html), which simplifies GPU access within Docker environments, improves device discovery, and provides better integration with modern container technologies.

## How to Access RDMA/RoCE NICs within Containers

RoCE NICs expose two tightly coupled networking planes on the same physical hardware: **the Linux Ethernet networking stack** and **the RDMA stack**.

- The Ethernet stack is represented by standard netdev interfaces (e.g., `ethX`) and provides conventional IP networking functions such as routing, TCP/UDP communication, MTU configuration and VLAN tagging, etc. **In RoCE deployments, this layer is essential, as RDMA traffic is carried over Ethernet and relies on a consistent L2/L3 network configuration.**

- In parallel, the RDMA stack is exposed through the rdma-core subsystem as devices such as `ionic_X on MI350x8` (or `mlx5_X on MI325x8`). This layer enables low-latency, kernel-bypass data movement using RDMA verbs (queue pairs, memory regions, completion queues), while still relying on the underlying Ethernet network for transport semantics.

Although logically distinct, the two stacks are tightly coupled at both the driver and hardware levels: each RDMA port maps directly to a corresponding ethX interface on the same NIC. This one-to-one binding allows a single adapter to simultaneously support traditional Ethernet networking and high-performance RDMA over the same physical fabric.

``` shell
root@rs-mi350x8-mn-test1:~# ip -br a
lo               UNKNOWN        127.0.0.1/8 ::1/128 
eth0             UP             178.128.2.192/20 10.46.0.8/16 fe80::d83a:4fff:fe5a:31/64 
eth1             UP             10.120.0.6/20 fe80::243c:b0ff:fe8f:7486/64 
eth2             UP             192.168.52.101/24 
eth3             UP             192.168.53.101/24 
eth4             UP             192.168.54.101/24 
eth5             UP             192.168.55.101/24 
eth6             UP             192.168.56.101/24 
eth7             UP             192.168.57.101/24 
eth8             UP             192.168.58.101/24 
eth9             UP             192.168.59.101/24 
docker0          DOWN           172.17.0.1/16 fe80::9cfc:32ff:fe46:f3b1/64 

root@rs-mi350x8-mn-test1:~# rdma link
link ionic_0/1 state ACTIVE physical_state LINK_UP netdev eth2 
link ionic_1/1 state ACTIVE physical_state LINK_UP netdev eth3 
link ionic_2/1 state ACTIVE physical_state LINK_UP netdev eth4 
link ionic_3/1 state ACTIVE physical_state LINK_UP netdev eth5 
link ionic_4/1 state ACTIVE physical_state LINK_UP netdev eth6 
link ionic_5/1 state ACTIVE physical_state LINK_UP netdev eth7 
link ionic_6/1 state ACTIVE physical_state LINK_UP netdev eth8 
link ionic_7/1 state ACTIVE physical_state LINK_UP netdev eth9 

root@rs-mi350x8-mn-test1:~# ibv_devices
    device          	   node GUID
    ------          	----------------
    ionic_0         	049081fffe4be670
    ionic_1         	049081fffe4bca20
    ionic_2         	049081fffe4c1a90
    ionic_3         	049081fffe4bfc78
    ionic_4         	049081fffe4c1790
    ionic_5         	049081fffe4c0f98
    ionic_6         	049081fffe4c1610
    ionic_7         	049081fffe4bf510

root@rs-mi350x8-mn-test1:~# ibv_devinfo -v | grep GID
			GID[  0]:		fe80::490:81ff:fe4b:e670, RoCE v2
			GID[  1]:		::ffff:192.168.52.101, RoCE v2
			GID[  0]:		fe80::490:81ff:fe4b:ca20, RoCE v2
			GID[  1]:		::ffff:192.168.53.101, RoCE v2
			GID[  0]:		fe80::490:81ff:fe4c:1a90, RoCE v2
			GID[  1]:		::ffff:192.168.54.101, RoCE v2
			GID[  0]:		fe80::490:81ff:fe4b:fc78, RoCE v2
			GID[  1]:		::ffff:192.168.55.101, RoCE v2
			GID[  0]:		fe80::490:81ff:fe4c:1790, RoCE v2
			GID[  1]:		::ffff:192.168.56.101, RoCE v2
			GID[  0]:		fe80::490:81ff:fe4c:f98, RoCE v2
			GID[  1]:		::ffff:192.168.57.101, RoCE v2
			GID[  0]:		fe80::490:81ff:fe4c:1610, RoCE v2
			GID[  1]:		::ffff:192.168.58.101, RoCE v2
			GID[  0]:		fe80::490:81ff:fe4b:f510, RoCE v2
			GID[  1]:		::ffff:192.168.59.101, RoCE v2
```

### Accessing RoCE Devices Inside Containers

Running RoCE workloads inside containers requires exposing both the RDMA subsystem and the associated Ethernet interfaces to the container.

First, the container must have access to the RDMA character devices under `/dev/infiniband`, which allow RDMA applications to create and manage verbs objects such as queue pairs, memory regions, and completion queues. This is typically achieved using:

`--device /dev/infiniband`

Second, the container must also have access to the corresponding ethX interfaces that provide the necessary network context, including IP configuration, MTU, VLAN membership, and QoS settings. Without this layer, RDMA communication may fail during address resolution (e.g., GID resolution) or may be unable to reach remote endpoints even if verbs initialization succeeds.

There are several ways to address this depending on the deployment model. **In non-Kubernetes environments**, the simplest approach is to share the host network namespace with the container:

`--network host`

This ensures the container has full visibility of the host’s Ethernet interfaces and associated configuration, making RoCE behavior consistent with the host network stack.

### Example Docker Command

#### MI350x8 Nodes

Here is an example command for `MI350x8` nodes that runs a fully privileged container with direct access to GPUs and RDMA/RoCE devices, optimized for high-performance multi-node applications with minimal isolation overhead:

``` shell
# --device=/dev/kfd: Gives access to AMD ROCm GPU compute driver.
# --device=/dev/dri: Enables GPU device access via the Linux graphics interface.
# --device=/dev/infiniband: Exposes InfiniBand RDMA hardware to the container.

# --network host: Uses the host network stack directly without isolation.
# --group-add video: Grants permission to access GPU device files.

# --cap-add=SYS_PTRACE: Allows process tracing and debugging.
# --security-opt seccomp=unconfined: Removes syscall filtering restrictions.
# --privileged: Gives full host-level access to the container (already covering SYS_PTRACE and seccomp relaxation, can be removed for production).

# --ipc host: Shares the host’s shared memory and IPC namespace (ignoring shm-size).
# --shm-size 128G: Sets large shared memory for inter-process communication.

root@rs-mi350x8-mn-test1:~# docker run --rm -it --hostname test-server1 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --shm-size 128G \
  rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1 \
  bash -c "
    set -ex

    apt update
    apt install -y iproute2 iputils-ping infiniband-diags ibverbs-utils

    echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdainic/pensando/ubuntu/1.117.1-a-63 noble main' > /etc/apt/sources.list.d/ainic.list    
    apt update &&  apt install -y libionic1

    /bin/bash
  "

root@test-server1:/# ip -br a
lo               UNKNOWN        127.0.0.1/8 ::1/128 
eth0             UP             178.128.2.192/20 10.46.0.8/16 fe80::d83a:4fff:fe5a:31/64 
eth1             UP             10.120.0.6/20 fe80::243c:b0ff:fe8f:7486/64 
eth2             UP             192.168.52.101/24 
eth3             UP             192.168.53.101/24 
eth4             UP             192.168.54.101/24 
eth5             UP             192.168.55.101/24 
eth6             UP             192.168.56.101/24 
eth7             UP             192.168.57.101/24 
eth8             UP             192.168.58.101/24 
eth9             UP             192.168.59.101/24 
docker0          DOWN           172.17.0.1/16 fe80::9cfc:32ff:fe46:f3b1/64 

root@test-server1:/# for i in {2..9}; do ping -c 1 192.168.5${i}.102; done

root@test-server1:/# rdma link
link ionic_0/1 state ACTIVE physical_state LINK_UP netdev eth2 
link ionic_1/1 state ACTIVE physical_state LINK_UP netdev eth3 
link ionic_2/1 state ACTIVE physical_state LINK_UP netdev eth4 
link ionic_3/1 state ACTIVE physical_state LINK_UP netdev eth5 
link ionic_4/1 state ACTIVE physical_state LINK_UP netdev eth6 
link ionic_5/1 state ACTIVE physical_state LINK_UP netdev eth7 
link ionic_6/1 state ACTIVE physical_state LINK_UP netdev eth8 
link ionic_7/1 state ACTIVE physical_state LINK_UP netdev eth9 

root@test-server1:/# ls /sys/class/infiniband
ionic_0  ionic_1  ionic_2  ionic_3  ionic_4  ionic_5  ionic_6  ionic_7

root@test-server1:/# ibv_devices
    device          	   node GUID
    ------          	----------------
    ionic_0         	049081fffe4be670
    ionic_1         	049081fffe4bca20
    ionic_2         	049081fffe4c1a90
    ionic_3         	049081fffe4bfc78
    ionic_4         	049081fffe4c1790
    ionic_5         	049081fffe4c0f98
    ionic_6         	049081fffe4c1610
    ionic_7         	049081fffe4bf510
```

The container installs several additional packages at runtime for RoCE networking validation, RDMA diagnostics, connectivity testing, and troubleshooting.

The most important package is `libionic1`, which provides the userspace RDMA provider library required for RoCE/RDMA communication through the [AMD Pensando Pollara NICs](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/network/nic-installation.html#amd-pensando-pollara-400-ai-nic) used on MI350x8 nodes. Without it, userspace RDMA frameworks such as RCCL, NCCL, UCX, and libibverbs may not be able to properly detect or utilize the RDMA devices.

#### MI325x8 Nodes

For MI325x8 nodes, which use [NVIDIA Mellanox ConnectX-7 NICs](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/network/nic-installation.html#nvidia-mellanox-cx-7-400gx1), the required RDMA provider support is already included in the standard RDMA userspace libraries and drivers. To run the same test on `MI325x8` nodes, use the following commands:

``` shell
root@rs-mi325x8-mn-test1:~# docker run --rm -it --hostname test-server1 \
  -v /root/pytorch_example/p2p_bw_test.py:/root/pytorch_example/p2p_bw_test.py \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --shm-size 128G \
  rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1 \
  bash -c "
    set -ex

    apt update
    apt install -y iproute2 iputils-ping infiniband-diags ibverbs-utils

    /bin/bash
  "

root@test-server1:/#  rdma link
link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev eth2 
link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev eth3 
link mlx5_2/1 state ACTIVE physical_state LINK_UP netdev eth4 
link mlx5_3/1 state ACTIVE physical_state LINK_UP netdev eth5 
link mlx5_4/1 state ACTIVE physical_state LINK_UP netdev eth6 
link mlx5_5/1 state ACTIVE physical_state LINK_UP netdev eth7 
link mlx5_6/1 state ACTIVE physical_state LINK_UP netdev eth8 
link mlx5_7/1 state ACTIVE physical_state LINK_UP netdev eth9 

root@test-server1:/#  ls /sys/class/infiniband
mlx5_0  mlx5_1  mlx5_2  mlx5_3  mlx5_4  mlx5_5  mlx5_6  mlx5_7

root@test-server1:/# ibv_devices
    device          	   node GUID
    ------          	----------------
    mlx5_0          	92e317fffe9789c8
    mlx5_1          	92e317fffe979020
    mlx5_2          	92e317fffe978f08
    mlx5_3          	92e317fffe978f28
    mlx5_4          	92e317fffe979098
    mlx5_5          	92e317fffe9789b8
    mlx5_6          	92e317fffe978f50
    mlx5_7          	92e317fffe978ff0
```

## Example: PyTorch P2P Bandwidth Test

Copy the example script from [mi350x8-mn/p2p_bw_test.py](mi350x8-mn/p2p_bw_test.py) to both GPU droplets and place it at the following path:

`/root/pytorch_example/p2p_bw_test.py`

``` python
import os
import torch
import torch.distributed as dist

def main():

    # Initialize distributed process group to set up RCCL communication between all participating processes.
    # init_method="env://":
    #   PyTorch reads all configuration from environment variables:
    #   - MASTER_ADDR  : IP of rank 0 node (coordination server)
    #   - MASTER_PORT  : port used for bootstrap communication
    #   - RANK         : unique ID of this process (0 ... world_size-1)
    #   - WORLD_SIZE   : total number of processes across all nodes
    # backend="nccl":
    #   Uses AMD RCCL or NVIDIA NCCL for GPU-to-GPU communication.
    #   Enables high-performance RDMA, XGM or transfers.
    dist.init_process_group(backend="nccl", init_method="env://")

    # Read runtime configuration
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu_id = int(os.environ["GPU_ID"])

    # Bind process to a specific GPU to ensures all CUDA operations from this process use the correct device
    torch.cuda.set_device(gpu_id)
    # Create a torch.device object for tensor allocations and operations
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Rank {rank} / {world_size}: using GPU {gpu_id}")

    # size_bytes: total data transferred per iteration
    # 16 GiB chosen to stress RDMA / RCCL pipeline sufficiently
    # num_elements: number of float32 elements (4 bytes each)
    size_bytes = 16 * 1024**3
    num_elements = size_bytes // 4
    print(f"Rank {rank} / {world_size}: allocating tensor")

    # Allocate GPU memory
    # tensor: source buffer (rank 0 sends this)
    # recv_tensor: destination buffer (rank 1 receives into)
    tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
    recv_tensor = torch.empty_like(tensor)
    torch.cuda.synchronize() # blocks the CPU until all previously issued GPU operations (compute, memory, NCCL communication) are fully completed

    # Warmup to trigger NCCL/RCCL communicator initialization, and warm up RDMA connections and GPU kernels
    N_WARMUP = 2
    for i in range(N_WARMUP):
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(recv_tensor, src=0)
    torch.cuda.synchronize()

    # Measurement using N_ITERS runs to compute average bandwidth
    N_ITERS = 10
    times = []
    for i in range(N_ITERS):

        torch.cuda.synchronize() # Ensure previous GPU work is finished before timing
        # CUDA events provide GPU-side accurate timing (not CPU wall clock)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # This is a blocking point-to-point operation using NCCL/RDMA
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(recv_tensor, src=0)
        end.record()
        torch.cuda.synchronize() # Ensure GPU operations complete before reading timing

        # Convert CUDA event timing to seconds
        elapsed_ms = start.elapsed_time(end)
        elapsed_sec = elapsed_ms / 1000.0
        times.append(elapsed_sec)

        if rank == 0:
            print(f"iter {i}: {elapsed_sec:.4f} sec")

    # Final statistics - Only rank 0 aggregates and prints results
    if rank == 0:
        avg = sum(times) / len(times)
        bandwidth_gbps = (size_bytes * 8) / avg / 1e9
        bandwidth_giBs = size_bytes / avg / (1024**3)
        print("\n==== RESULT ====")
        print(f"Avg time: {avg:.4f} sec")
        print(f"Bandwidth: {bandwidth_gbps:.2f} Gbps")
        print(f"Bandwidth: {bandwidth_giBs:.2f} GiB/s")

    # Cleanup distributed resources: properly releases NCCL/RCCL communicators and network resources
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### Direct Host Execution

#### MI350x8 Nodes

To run the test on `MI350x8` nodes, start the rank 0 process first on `rs-mi350x8-mn-test1`, then launch the rank 1 process on `rs-mi350x8-mn-test2`.

``` shell
# rs-mi350x8-mn-test1 (10.120.0.6), rs-mi350x8-mn-test2 (10.120.0.7)

root@rs-mi350x8-mn-test1:~/pytorch_example# NCCL_DEBUG=INFO \
NCCL_IB_GID_INDEX=1 \
NCCL_SOCKET_IFNAME=eth1 \
NCCL_IB_QPS_PER_CONNECTION=8 \
NCCL_P2P_CHUNKSIZE=16777216 \
MASTER_ADDR=10.120.0.6 \
MASTER_PORT=12355 \
WORLD_SIZE=2 \
RANK=0 \
NCCL_IB_HCA=ionic_0 \
GPU_ID=0 \
python3 /root/pytorch_example/p2p_bw_test.py

root@rs-mi350x8-mn-test2:~/pytorch_example# NCCL_DEBUG=INFO \
NCCL_IB_GID_INDEX=1 \
NCCL_SOCKET_IFNAME=eth1 \
NCCL_IB_QPS_PER_CONNECTION=8 \
NCCL_P2P_CHUNKSIZE=16777216 \
MASTER_ADDR=10.120.0.6 \
MASTER_PORT=12355 \
WORLD_SIZE=2 \
RANK=1 \
NCCL_IB_HCA=ionic_0 \
GPU_ID=0 \
python3 /root/pytorch_example/p2p_bw_test.py
```

These commands launch a two-node RCCL-based GPU point-to-point benchmark in which rank 0 transfers a large tensor from GPU0 on the first node to GPU0 on the second node over RDMA/RoCE using the ionic_0 NIC. The program measures sustained inter-node GPU-to-GPU bandwidth using the event-based timing across multiple iterations.

At runtime, RCCL first performs a TCP-based bootstrap over the control-plane interface `eth1`. Using the `MASTER_ADDR:MASTER_PORT` rendezvous mechanism, both processes connect, discover each other, and exchange a `ncclUniqueId`, which serves as the initialization token for constructing the distributed communicator. 

After this handshake completes, RCCL transitions from bootstrap to data-plane setup: it selects the RoCE device ionic_0 (as specified by `NCCL_IB_HCA`) and builds RDMA-based GPU communication paths. During this stage, it configures multiple queue pairs and communication channels (e.g., controlled by `NCCL_IB_QPS_PER_CONNECTION`), registers GPU memory for GPUDirect RDMA access, and establishes optimized data paths. Once initialization is finished, eth1 is no longer used for data movement; all tensor traffic flows directly over RDMA through ionic_0, enabling high-bandwidth, low-latency GPU-to-GPU communication across nodes.

It is important to distinguish between the logical RDMA device and the underlying physical network configuration. NCCL communicates with `ionic_0` as the RDMA-capable interface exposed by the RoCE driver, rather than interacting directly with Ethernet interfaces such as `eth2`. However, ionic_0 is physically backed by and configured through the underlying Ethernet interface eth2, meaning that key network properties—such as IP addressing, MTU, VLAN configuration, and link-level tuning—are defined at the eth2 layer. These settings indirectly affect the performance and behavior of RDMA communication, even though NCCL itself operates on the abstracted RDMA device ionic_0.

Check [mi350x8-mn/mi350x8_p2p_bw_test_node1.txt](mi350x8-mn/mi350x8_p2p_bw_test_node1.txt) and [mi350x8-mn/mi350x8_p2p_bw_test_node2.txt](mi350x8-mn/mi350x8_p2p_bw_test_node2.txt) for the corresponding output logs. You can see that the measured bandwidth is still lower than the theoretical 400 Gb/s. This can be further optimized by tuning NCCL/RCCL environment variables such as the number of channels, chunk size, and other communication parameters.

#### MI325x8 Nodes

To run the same test on `MI325x8` nodes, use the following commands:

``` shell
# rs-mi325x8-mn-test1 (10.118.0.10), rs-mi325x8-mn-test2 (10.118.0.13)

root@rs-mi325x8-mn-test1:~/pytorch_example# NCCL_DEBUG=INFO \
NCCL_IB_GID_INDEX=3 \
NCCL_SOCKET_IFNAME=eth1 \
NCCL_IB_QPS_PER_CONNECTION=8 \
NCCL_P2P_CHUNKSIZE=16777216 \
MASTER_ADDR=10.118.0.10 \
MASTER_PORT=12355 \
WORLD_SIZE=2 \
RANK=0 \
NCCL_IB_HCA=mlx5_0 \
GPU_ID=0 \
python3 /root/pytorch_example/p2p_bw_test.py

root@rs-mi325x8-mn-test2:~/pytorch_example# NCCL_DEBUG=INFO \
NCCL_IB_GID_INDEX=3 \
NCCL_SOCKET_IFNAME=eth1 \
NCCL_IB_QPS_PER_CONNECTION=8 \
NCCL_P2P_CHUNKSIZE=16777216 \
MASTER_ADDR=10.118.0.10 \
MASTER_PORT=12355 \
WORLD_SIZE=2 \
RANK=1 \
NCCL_IB_HCA=mlx5_0 \
GPU_ID=0 \
python3 /root/pytorch_example/p2p_bw_test.py
```

The key difference lies in the configuration of `NCCL_IB_GID_INDEX` and `NCCL_IB_HCA`, since MI325x8 nodes use a different NIC type compared to MI350x8 nodes, requiring different RDMA device selection and GID indexing for proper NCCL communication. `MASTER_ADDR` is also different, as it must be set to the IP address of the rank 0 node’s eth1-vpc interface.

### Containers


#### MI350x8 Nodes

To run the test on `MI350x8` nodes, start the rank 0 container on `rs-mi350x8-mn-test1` first, then launch the rank 1 container on `rs-mi350x8-mn-test2`:

``` shell
# rs-mi350x8-mn-test1 (10.120.0.6), rs-mi350x8-mn-test2 (10.120.0.7)

root@rs-mi350x8-mn-test1:~/pytorch_example# docker run --rm -it --hostname test-server1 \
  -v /root/pytorch_example/p2p_bw_test.py:/root/pytorch_example/p2p_bw_test.py \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --shm-size 128G \
  rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1 \
  bash -c "
    set -ex

    echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdainic/pensando/ubuntu/1.117.1-a-63 noble main' > /etc/apt/sources.list.d/ainic.list    
    apt update &&  apt install -y libionic1

    NCCL_DEBUG=INFO \
    NCCL_IB_GID_INDEX=1 \
    NCCL_SOCKET_IFNAME=eth1 \
    NCCL_IB_QPS_PER_CONNECTION=8 \
    NCCL_P2P_CHUNKSIZE=16777216 \
    MASTER_ADDR=10.120.0.6 \
    MASTER_PORT=12355 \
    WORLD_SIZE=2 \
    RANK=0 \
    NCCL_IB_HCA=ionic_0 \
    GPU_ID=0 \
    python3 /root/pytorch_example/p2p_bw_test.py
  "

root@rs-mi350x8-mn-test2:~/pytorch_example# docker run --rm -it --hostname test-server2 \
  -v /root/pytorch_example/p2p_bw_test.py:/root/pytorch_example/p2p_bw_test.py \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --shm-size 128G \
  rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1 \
  bash -c "
    set -ex

    echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdainic/pensando/ubuntu/1.117.1-a-63 noble main' > /etc/apt/sources.list.d/ainic.list    
    apt update &&  apt install -y libionic1

    NCCL_DEBUG=INFO \
    NCCL_IB_GID_INDEX=1 \
    NCCL_SOCKET_IFNAME=eth1 \
    NCCL_IB_QPS_PER_CONNECTION=8 \
    NCCL_P2P_CHUNKSIZE=16777216 \
    MASTER_ADDR=10.120.0.6 \
    MASTER_PORT=12355 \
    WORLD_SIZE=2 \
    RANK=1 \
    NCCL_IB_HCA=ionic_0 \
    GPU_ID=0 \
    python3 /root/pytorch_example/p2p_bw_test.py
  "
```

#### MI325x8 Nodes

To run the same test on `MI325x8` nodes, use the following commands:

``` shell
# rs-mi325x8-mn-test1 (10.118.0.10), rs-mi325x8-mn-test2 (10.118.0.13)

root@rs-mi325x8-mn-test1:~/pytorch_example# docker run --rm -it --hostname test-server1 \
  -v /root/pytorch_example/p2p_bw_test.py:/root/pytorch_example/p2p_bw_test.py \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --shm-size 128G \
  rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1 \
  bash -c "
    set -ex
    NCCL_DEBUG=INFO \
    NCCL_IB_GID_INDEX=3 \
    NCCL_SOCKET_IFNAME=eth1 \
    NCCL_IB_QPS_PER_CONNECTION=8 \
    NCCL_P2P_CHUNKSIZE=16777216 \
    MASTER_ADDR=10.118.0.10 \
    MASTER_PORT=12355 \
    WORLD_SIZE=2 \
    RANK=0 \
    NCCL_IB_HCA=mlx5_0 \
    GPU_ID=0 \
    python3 /root/pytorch_example/p2p_bw_test.py
  "

  root@rs-mi325x8-mn-test2:~/pytorch_example# docker run --rm -it --hostname test-server2 \
  -v /root/pytorch_example/p2p_bw_test.py:/root/pytorch_example/p2p_bw_test.py \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --shm-size 128G \
  rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1 \
    bash -c "
    set -ex
    NCCL_DEBUG=INFO \
    NCCL_IB_GID_INDEX=3 \
    NCCL_SOCKET_IFNAME=eth1 \
    NCCL_IB_QPS_PER_CONNECTION=8 \
    NCCL_P2P_CHUNKSIZE=16777216 \
    MASTER_ADDR=10.118.0.10 \
    MASTER_PORT=12355 \
    WORLD_SIZE=2 \
    RANK=1 \
    NCCL_IB_HCA=mlx5_0 \
    GPU_ID=0 \
    python3 /root/pytorch_example/p2p_bw_test.py
  "
```

## Documentation and Resources

- [AMD ROCm Documentation](https://rocm.docs.amd.com/en/latest/)

- [Running ROCm Docker Containers](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)

- [AMD Container Toolkit](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/quick-start-guide.html)

- [PyTorch on ROCm Installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)

- [ROCm PyTorch Images](https://hub.docker.com/r/rocm/pytorch)

- [RCCL environment variables](https://rocm.docs.amd.com/projects/rccl/en/develop/api-reference/env-variables.html#rccl-environment-variables)

- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

- [Configure RCCL](https://docs.digitalocean.com/products/droplets/how-to/gpu/configure-multi-node/#configure-nccl-or-rccl)






