# Multi-Node Environment Setup and Testing for DOKS with NVIDIA GPUs

This document captures the step-by-step process for setting up a multi-node DOKS environment with NVIDIA B300x8 GPU nodes, configuring RDMA sharing using the Multus CNI and ipvlan CNI, and running the validation tests, including RCCL and PyTorch P2P data transfer over RDMA/RoCE.

For background information on the underlying concepts and theory, please refer to [amd-mn-k8s-environment](amd-mn-k8s-environment.MD), [amd-mn-k8s-rdma-sharing.MD](amd-mn-k8s-rdma-sharing.MD) and [amd-mn-k8s-rccl-testing.MD](amd-mn-k8s-rccl-testing.MD).

The key difference is that AMD MI325x8 or MI350x8 clusters provide 8 rails in the GPU fabric, whereas NVIDIA B300x8 clusters provide 16 rails. As a result, the RDMA/RoCE configuration differs between the two platforms.

## Environment Setup 

Ensure that your team account has at least two B300x8 nodes available within the same GPU Fabric. Create a DOKS cluster with one CPU worker and three B300 x8 GPU workers using doctl as follows:

``` bash
doctl vpcs list
doctl compute size list | grep cpu 
doctl compute size list | grep gpu 
doctl kubernetes cluster list

# Use a fabric-enabled size; otherwise, RDMA/RoCE will not be enabled.
doctl kubernetes cluster create rs-nvidia-rdma-sharing \
  --region mkc1 \
  --version 1.36.0-do.1 \
  --node-pool "name=rs-mn-cpu-pool;size=s-4vcpu-16gb-320gb-intel;count=1" \
  --node-pool "name=rs-mn-b300x8-pool;size=gpu-b300x8-2304gb-lc-fabric-contracted;count=3" \
  --tag "rs-test"

doctl kubernetes cluster get rs-nvidia-rdma-sharing -o json
doctl kubernetes cluster delete rs-nvidia-rdma-sharing
``` 

The 4 workers have been created and are now managed by DOKS.

``` bash
# doctl compute droplet list

584880375    rs-mn-cpu-pool-37ylkn                     165.245.199.211    10.132.0.6                                      16384      4        320     mkc1      Debian do-kube-1.36.0-do.2               7ab3c1c0-364c-43cc-a708-e757dec1376f    active    k8s,k8s:9df54065-3aa8-4067-bbdd-b6814352af5d,k8s:worker    private_networking                                  
584880376    rs-mn-b300x8-pool-37ylk3                  165.245.199.217    10.132.0.7                                      3670016    224      2046    mkc1      Debian do-kube-1.36.0-do.2-gpu-nvidia    7ab3c1c0-364c-43cc-a708-e757dec1376f    active    k8s,k8s:9df54065-3aa8-4067-bbdd-b6814352af5d,k8s:worker    droplet_agent,private_networking                    
584880377    rs-mn-b300x8-pool-37ylk8                  165.245.199.203    10.132.0.5                                      3670016    224      2046    mkc1      Debian do-kube-1.36.0-do.2-gpu-nvidia    7ab3c1c0-364c-43cc-a708-e757dec1376f    active    k8s,k8s:9df54065-3aa8-4067-bbdd-b6814352af5d,k8s:worker    droplet_agent,private_networking                    
584880378    rs-mn-b300x8-pool-37ylku                  165.245.199.226    10.132.0.8                                      3670016    224      2046    mkc1      Debian do-kube-1.36.0-do.2-gpu-nvidia    7ab3c1c0-364c-43cc-a708-e757dec1376f    active    k8s,k8s:9df54065-3aa8-4067-bbdd-b6814352af5d,k8s:worker    droplet_agent,private_networking  
```

## DOKS Worker Check (Host OS)

<details>
<summary>B300x8</summary>

``` bash
# uname -r
6.12.94+deb13-amd64

# cat /etc/os-release
PRETTY_NAME="Debian GNU/Linux 13 (trixie)"
NAME="Debian GNU/Linux"
VERSION_ID="13"
VERSION="13 (trixie)"
VERSION_CODENAME=trixie
DEBIAN_VERSION_FULL=13.5
ID=debian
HOME_URL="https://www.debian.org/"
SUPPORT_URL="https://www.debian.org/support"
BUG_REPORT_URL="https://bugs.debian.org/"

# dkms status
mlnx-ofed-kernel/26.01.OFED.26.01.1.0.0.1, 6.12.94+deb13-amd64, x86_64: installed (Original modules exist)
nvidia/590.48.01, 6.12.94+deb13-amd64, x86_64: installed

# nvidia-smi 
Wed Jul 15 21:57:39 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA B300 SXM6 PC            On  |   00000000:85:00.0 Off |                    0 |
| N/A   29C    P0            234W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA B300 SXM6 PC            On  |   00000000:8D:00.0 Off |                    0 |
| N/A   29C    P0            238W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA B300 SXM6 PC            On  |   00000000:95:00.0 Off |                    0 |
| N/A   29C    P0            238W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA B300 SXM6 PC            On  |   00000000:9D:00.0 Off |                    0 |
| N/A   28C    P0            236W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA B300 SXM6 PC            On  |   00000000:A5:00.0 Off |                    0 |
| N/A   28C    P0            234W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA B300 SXM6 PC            On  |   00000000:AD:00.0 Off |                    0 |
| N/A   27C    P0            183W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA B300 SXM6 PC            On  |   00000000:B5:00.0 Off |                    0 |
| N/A   27C    P0            182W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA B300 SXM6 PC            On  |   00000000:BD:00.0 Off |                    0 |
| N/A   27C    P0            182W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

# nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    NIC2    NIC3    NIC4    NIC5    NIC6NIC7     NIC8    NIC9    NIC10   NIC11   NIC12   NIC13   NIC14   NIC15   CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18    PXB     PXB     NODE    NODE    NODE    NODE    NODENODE     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-111   0               N/A
GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18    NODE    NODE    PXB     PXB     NODE    NODE    NODENODE     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-111   0               N/A
GPU2    NV18    NV18     X      NV18    NV18    NV18    NV18    NV18    NODE    NODE    NODE    NODE    PXB     PXB     NODENODE     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-111   0               N/A
GPU3    NV18    NV18    NV18     X      NV18    NV18    NV18    NV18    NODE    NODE    NODE    NODE    NODE    NODE    PXB PXB      SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-111   0               N/A
GPU4    NV18    NV18    NV18    NV18     X      NV18    NV18    NV18    SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      PXB     PXB     NODE    NODE    NODE    NODE    NODE    NODE    112-223 1               N/A
GPU5    NV18    NV18    NV18    NV18    NV18     X      NV18    NV18    SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      NODE    NODE    PXB     PXB     NODE    NODE    NODE    NODE    112-223 1               N/A
GPU6    NV18    NV18    NV18    NV18    NV18    NV18     X      NV18    SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      NODE    NODE    NODE    NODE    PXB     PXB     NODE    NODE    112-223 1               N/A
GPU7    NV18    NV18    NV18    NV18    NV18    NV18    NV18     X      SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      NODE    NODE    NODE    NODE    NODE    NODE    PXB     PXB     112-223 1               N/A
NIC0    PXB     NODE    NODE    NODE    SYS     SYS     SYS     SYS      X      PIX     NODE    NODE    NODE    NODE    NODENODE     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC1    PXB     NODE    NODE    NODE    SYS     SYS     SYS     SYS     PIX      X      NODE    NODE    NODE    NODE    NODENODE     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC2    NODE    PXB     NODE    NODE    SYS     SYS     SYS     SYS     NODE    NODE     X      PIX     NODE    NODE    NODENODE     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC3    NODE    PXB     NODE    NODE    SYS     SYS     SYS     SYS     NODE    NODE    PIX      X      NODE    NODE    NODENODE     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC4    NODE    NODE    PXB     NODE    SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE     X      PIX     NODENODE     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC5    NODE    NODE    PXB     NODE    SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    PIX      X      NODENODE     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC6    NODE    NODE    NODE    PXB     SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    NODE    NODE     X  PIX      SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC7    NODE    NODE    NODE    PXB     SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    NODE    NODE    PIX  X       SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC8    SYS     SYS     SYS     SYS     PXB     NODE    NODE    NODE    SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS       X      PIX     NODE    NODE    NODE    NODE    NODE    NODE
NIC9    SYS     SYS     SYS     SYS     PXB     NODE    NODE    NODE    SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      PIX      X      NODE    NODE    NODE    NODE    NODE    NODE
NIC10   SYS     SYS     SYS     SYS     NODE    PXB     NODE    NODE    SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      NODE    NODE     X      PIX     NODE    NODE    NODE    NODE
NIC11   SYS     SYS     SYS     SYS     NODE    PXB     NODE    NODE    SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      NODE    NODE    PIX      X      NODE    NODE    NODE    NODE
NIC12   SYS     SYS     SYS     SYS     NODE    NODE    PXB     NODE    SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      NODE    NODE    NODE    NODE     X      PIX     NODE    NODE
NIC13   SYS     SYS     SYS     SYS     NODE    NODE    PXB     NODE    SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      NODE    NODE    NODE    NODE    PIX      X      NODE    NODE
NIC14   SYS     SYS     SYS     SYS     NODE    NODE    NODE    PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      NODE    NODE    NODE    NODE    NODE    NODE     X      PIX
NIC15   SYS     SYS     SYS     SYS     NODE    NODE    NODE    PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS SYS      NODE    NODE    NODE    NODE    NODE    NODE    PIX      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7
  NIC8: mlx5_8
  NIC9: mlx5_9
  NIC10: mlx5_10
  NIC11: mlx5_11
  NIC12: mlx5_12
  NIC13: mlx5_13
  NIC14: mlx5_14
  NIC15: mlx5_15

# ls /sys/class/infiniband/
mlx5_0  mlx5_10  mlx5_12  mlx5_14  mlx5_16  mlx5_18  mlx5_2  mlx5_4  mlx5_6  mlx5_8
mlx5_1  mlx5_11  mlx5_13  mlx5_15  mlx5_17  mlx5_19  mlx5_3  mlx5_5  mlx5_7  mlx5_9

# ibv_devices
    device                 node GUID
    ------              ----------------
    mlx5_0              ae3ae2fffebc9132
    mlx5_1              ae3ae2fffebc9133
    mlx5_2              ae3ae2fffebc9152
    mlx5_3              ae3ae2fffebc9153
    mlx5_4              ae3ae2fffebc9162
    mlx5_5              ae3ae2fffebc9163
    mlx5_6              ae3ae2fffebc9142
    mlx5_7              ae3ae2fffebc9143
    mlx5_8              ae3ae2fffebc91a2
    mlx5_9              ae3ae2fffebc91a3
    mlx5_10             ae3ae2fffebc9182
    mlx5_11             ae3ae2fffebc9183
    mlx5_12             ae3ae2fffebc9172
    mlx5_13             ae3ae2fffebc9173
    mlx5_14             ae3ae2fffebc9192
    mlx5_15             ae3ae2fffebc9193
    mlx5_16             ac3ae20300bc912e
    mlx5_17             ac3ae20300bc912f
    mlx5_18             ac3ae20300bc9130
    mlx5_19             ac3ae20300bc9131

# ip -br a | grep fabric
fabric0          UP             fe80::ac3a:e2ff:febc:9132/64 
fabric1          UP             fe80::ac3a:e2ff:febc:9133/64 
fabric2          UP             fe80::ac3a:e2ff:febc:9152/64 
fabric3          UP             fe80::ac3a:e2ff:febc:9153/64 
fabric4          UP             fe80::ac3a:e2ff:febc:9162/64 
fabric5          UP             fe80::ac3a:e2ff:febc:9163/64 
fabric6          UP             fe80::ac3a:e2ff:febc:9142/64 
fabric7          UP             fe80::ac3a:e2ff:febc:9143/64 
fabric8          UP             fe80::ac3a:e2ff:febc:91a2/64 
fabric9          UP             fe80::ac3a:e2ff:febc:91a3/64 
fabric10         UP             fe80::ac3a:e2ff:febc:9182/64 
fabric11         UP             fe80::ac3a:e2ff:febc:9183/64 
fabric12         UP             fe80::ac3a:e2ff:febc:9172/64 
fabric13         UP             fe80::ac3a:e2ff:febc:9173/64 
fabric14         UP             fe80::ac3a:e2ff:febc:9192/64 
fabric15         UP             fe80::ac3a:e2ff:febc:9193/64 

# ibv_devinfo -v | grep GID
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9132, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9132, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9133, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9133, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9152, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9152, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9153, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9153, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9162, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9162, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9163, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9163, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9142, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9142, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9143, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9143, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:91a2, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:91a2, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:91a3, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:91a3, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9182, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9182, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9183, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9183, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9172, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9172, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9173, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9173, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9192, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9192, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9193, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9193, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e203:00bc:912e
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e203:00bc:912f
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e203:00bc:9130
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e203:00bc:9131

# rdma link
link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev fabric0 
link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev fabric1 
link mlx5_2/1 state ACTIVE physical_state LINK_UP netdev fabric2 
link mlx5_3/1 state ACTIVE physical_state LINK_UP netdev fabric3 
link mlx5_4/1 state ACTIVE physical_state LINK_UP netdev fabric4 
link mlx5_5/1 state ACTIVE physical_state LINK_UP netdev fabric5 
link mlx5_6/1 state ACTIVE physical_state LINK_UP netdev fabric6 
link mlx5_7/1 state ACTIVE physical_state LINK_UP netdev fabric7 
link mlx5_8/1 state ACTIVE physical_state LINK_UP netdev fabric8 
link mlx5_9/1 state ACTIVE physical_state LINK_UP netdev fabric9 
link mlx5_10/1 state ACTIVE physical_state LINK_UP netdev fabric10 
link mlx5_11/1 state ACTIVE physical_state LINK_UP netdev fabric11 
link mlx5_12/1 state ACTIVE physical_state LINK_UP netdev fabric12 
link mlx5_13/1 state ACTIVE physical_state LINK_UP netdev fabric13 
link mlx5_14/1 state ACTIVE physical_state LINK_UP netdev fabric14 
link mlx5_15/1 state ACTIVE physical_state LINK_UP netdev fabric15 
link mlx5_16/1 subnet_prefix fe80:0000:0000:0000 lid 1 sm_lid 1 lmc 0 state ACTIVE physical_state LINK_UP 
link mlx5_17/1 subnet_prefix fe80:0000:0000:0000 lid 3 sm_lid 1 lmc 0 state ACTIVE physical_state LINK_UP 
link mlx5_18/1 subnet_prefix fe80:0000:0000:0000 lid 6 sm_lid 1 lmc 0 state ACTIVE physical_state LINK_UP 
link mlx5_19/1 subnet_prefix fe80:0000:0000:0000 lid 5 sm_lid 1 lmc 0 state ACTIVE physical_state LINK_UP                        
```

</details>

## DOKS Cluster Check

### Basic Cluster Check

Perform the following checks to verify the health and readiness of the DOKS cluster:

<details>
<summary>DOKS Cluster</summary>

``` bash
# kubectl get nodes -o wide
NAME                       STATUS   ROLES    AGE     VERSION   INTERNAL-IP   EXTERNAL-IP       OS-IMAGE                       KERNEL-VERSION                CONTAINER-RUNTIME
rs-mn-b300x8-pool-37ylk3   Ready    <none>   3h52m   v1.36.0   10.132.0.7    165.245.199.217   Debian GNU/Linux 13 (trixie)   6.12.94+deb13-amd64 (amd64)   containerd://2.2.3
rs-mn-b300x8-pool-37ylk8   Ready    <none>   3h52m   v1.36.0   10.132.0.5    165.245.199.203   Debian GNU/Linux 13 (trixie)   6.12.94+deb13-amd64 (amd64)   containerd://2.2.3
rs-mn-b300x8-pool-37ylku   Ready    <none>   3h52m   v1.36.0   10.132.0.8    165.245.199.226   Debian GNU/Linux 13 (trixie)   6.12.94+deb13-amd64 (amd64)   containerd://2.2.3
rs-mn-cpu-pool-37ylkn      Ready    <none>   4h7m    v1.36.0   10.132.0.6    165.245.199.211   Debian GNU/Linux 13 (trixie)   6.12.94+deb13-amd64 (amd64)   containerd://2.2.3

# kubectl get daemonset -ALL
NAMESPACE     NAME                                        DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                                                                                                  AGE    L
kube-system   cilium                                      4         4         4       4            4           kubernetes.io/os=linux                                                                                         4h8m   
kube-system   csi-do-node                                 4         4         4       4            4           kubernetes.io/os=linux                                                                                         4h4m   
kube-system   do-node-agent                               1         1         1       1            1           kubernetes.io/os=linux                                                                                         4h3m   
kube-system   do-node-agent-amd-device-metrics-exporter   0         0         0       0            0           doks.digitalocean.com/gpu-brand=amd,kubernetes.io/os=linux                                                     4h3m   
kube-system   do-node-agent-nvidia-dcgm-exporter          3         3         3       3            3           doks.digitalocean.com/gpu-brand=nvidia,doks.digitalocean.com/nvidia-dcgm-enabled=true,kubernetes.io/os=linux   4h3m   
kube-system   doks-telemetry-config-reloader              4         4         4       4            4           kubernetes.io/os=linux                                                                                         4h9m   
kube-system   k8s-nvidia-device-plugin                    3         3         3       3            3           doks.digitalocean.com/gpu-brand=nvidia,kubernetes.io/os=linux                                                  4h3m   
kube-system   rdma-shared-dp-ds                           3         3         3       3            3           doks.digitalocean.com/is-gpu-fabric-connected=true,kubernetes.io/os=linux                                      4h3m   

# kubectl get pods -ALL
NAMESPACE     NAME                                       READY   STATUS    RESTARTS       AGE     L
kube-system   cilium-2lrcg                               2/2     Running   0              3h53m   
kube-system   cilium-gxdgd                               2/2     Running   0              4h8m    
kube-system   cilium-v8f5b                               2/2     Running   0              3h53m   
kube-system   cilium-z5gbc                               2/2     Running   0              3h53m   
kube-system   coredns-869c9f5ffb-9b6lp                   1/1     Running   0              3h53m   
kube-system   coredns-869c9f5ffb-bmxx9                   1/1     Running   0              4h7m    
kube-system   coredns-869c9f5ffb-qlnkn                   1/1     Running   0              3h53m   
kube-system   csi-do-node-75rgk                          2/2     Running   0              3h53m   
kube-system   csi-do-node-gvh7q                          2/2     Running   0              3h53m   
kube-system   csi-do-node-hr7vp                          2/2     Running   0              3h53m   
kube-system   csi-do-node-zhk9l                          2/2     Running   0              4h4m    
kube-system   do-node-agent-59v6j                        1/1     Running   0              4h4m    
kube-system   do-node-agent-nvidia-dcgm-exporter-4xjdt   3/3     Running   0              3h53m   
kube-system   do-node-agent-nvidia-dcgm-exporter-hw4tq   3/3     Running   0              3h53m   
kube-system   do-node-agent-nvidia-dcgm-exporter-jh22c   3/3     Running   0              3h53m   
kube-system   doks-telemetry-config-reloader-4dksr       1/1     Running   0              3h53m   
kube-system   doks-telemetry-config-reloader-fb648       1/1     Running   0              4h8m    
kube-system   doks-telemetry-config-reloader-knk82       1/1     Running   0              3h53m   
kube-system   doks-telemetry-config-reloader-kx2jf       1/1     Running   0              3h53m   
kube-system   hubble-relay-64b8659f77-f5vhp              1/1     Running   1 (4h5m ago)   4h9m    
kube-system   hubble-ui-5bb8b9bd74-8cwwt                 2/2     Running   0              4h9m    
kube-system   k8s-nvidia-device-plugin-8kslh             1/1     Running   0              3h53m   
kube-system   k8s-nvidia-device-plugin-d9d6t             1/1     Running   0              3h53m   
kube-system   k8s-nvidia-device-plugin-gkrl5             1/1     Running   0              3h53m   
kube-system   konnectivity-agent-78597f974f-q2mkd        1/1     Running   0              4h9m    
kube-system   konnectivity-agent-78597f974f-xdtdt        1/1     Running   0              4h9m    
kube-system   rdma-shared-dp-ds-24pf8                    1/1     Running   0              3h53m   
kube-system   rdma-shared-dp-ds-fvqsw                    1/1     Running   0              3h53m   
kube-system   rdma-shared-dp-ds-h5zgb                    1/1     Running   0              3h53m  
```

</details>

### Managed GPU Droplets

Perform the following checks on DOKS workers:

<details>
<summary>B300x8</summary>

``` bash
# kubectl describe node rs-mn-b300x8-pool-37ylku
Name:               rs-mn-b300x8-pool-37ylku
Roles:              <none>
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/instance-type=gpu-b300x8-2304gb-lc-fabric-contracted
                    beta.kubernetes.io/os=linux
                    doks.digitalocean.com/gpu-brand=nvidia
                    doks.digitalocean.com/gpu-model=b300
                    doks.digitalocean.com/is-gpu-fabric-connected=true
                    doks.digitalocean.com/managed=true
                    doks.digitalocean.com/node-id=1fe11d5e-dd76-4c03-9e98-00711f742baa
                    doks.digitalocean.com/node-pool=rs-mn-b300x8-pool
                    doks.digitalocean.com/node-pool-id=512834ce-cb3d-4922-9fee-91fdd4ec92cc
                    doks.digitalocean.com/nvidia-dcgm-enabled=true
                    doks.digitalocean.com/version=1.36.0-do.2
                    failure-domain.beta.kubernetes.io/region=mkc1
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=rs-mn-b300x8-pool-37ylku
                    kubernetes.io/os=linux
                    node.kubernetes.io/instance-type=gpu-b300x8-2304gb-lc-fabric-contracted
                    nvidia.com/gpu=8
                    region=mkc1
                    topology.kubernetes.io/region=mkc1
Annotations:        alpha.kubernetes.io/provided-node-ip: 10.132.0.8
                    csi.volume.kubernetes.io/nodeid: {"dobs.csi.digitalocean.com":"584880378"}
                    network.cilium.io/ipv4-Ingress-ip: 10.151.1.40
                    network.cilium.io/ipv4-cilium-host: 10.151.1.62
                    network.cilium.io/ipv4-health-ip: 10.151.1.11
                    network.cilium.io/ipv4-pod-cidr: 10.151.1.0/25
                    node.alpha.kubernetes.io/ttl: 0
                    volumes.kubernetes.io/controller-managed-attach-detach: true
CreationTimestamp:  Wed, 15 Jul 2026 18:12:03 +0000
Taints:             nvidia.com/gpu:NoSchedule
Unschedulable:      false
Lease:
  HolderIdentity:  rs-mn-b300x8-pool-37ylku
  AcquireTime:     <unset>
  RenewTime:       Wed, 15 Jul 2026 22:08:51 +0000
Conditions:
  Type                                  Status  LastHeartbeatTime                 LastTransitionTime                Reason                       Message
  ----                                  ------  -----------------                 ------------------                ------                       -------
  NetworkUnavailable                    False   Wed, 15 Jul 2026 18:12:29 +0000   Wed, 15 Jul 2026 18:12:29 +0000   CiliumIsUp                   Cilium is running on this node
  MemoryPressure                        False   Wed, 15 Jul 2026 22:08:51 +0000   Wed, 15 Jul 2026 18:12:03 +0000   KubeletHasSufficientMemory   kubelet has sufficient memory available
  DiskPressure                          False   Wed, 15 Jul 2026 22:08:51 +0000   Wed, 15 Jul 2026 18:12:03 +0000   KubeletHasNoDiskPressure     kubelet has no disk pressure
  PIDPressure                           False   Wed, 15 Jul 2026 22:08:51 +0000   Wed, 15 Jul 2026 18:12:03 +0000   KubeletHasSufficientPID      kubelet has sufficient PID available
  Ready                                 True    Wed, 15 Jul 2026 22:08:51 +0000   Wed, 15 Jul 2026 18:12:22 +0000   KubeletReady                 kubelet is posting ready status
  doks.kubernetes.io/NvidiaGpuReady     True    Wed, 15 Jul 2026 22:08:42 +0000   Wed, 15 Jul 2026 18:13:12 +0000   EndpointOK                   Endpoint reports ready at http://127.0.0.1:5000/metrics
  doks.kubernetes.io/CiliumAgentReady   True    Wed, 15 Jul 2026 22:08:51 +0000   Wed, 15 Jul 2026 18:12:51 +0000   EndpointOK                   Endpoint reports ready at http://127.0.0.1:9879/healthz
Addresses:
  InternalIP:  10.132.0.8
  Hostname:    rs-mn-b300x8-pool-37ylku
  ExternalIP:  165.245.199.226
Capacity:
  cpu:                224
  ephemeral-storage:  2111508792Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             3699199600Ki
  nvidia.com/gpu:     8
  pods:               110
  rdma/fabric0:       100
  rdma/fabric1:       100
  rdma/fabric10:      100
  rdma/fabric11:      100
  rdma/fabric12:      100
  rdma/fabric13:      100
  rdma/fabric14:      100
  rdma/fabric15:      100
  rdma/fabric2:       100
  rdma/fabric3:       100
  rdma/fabric4:       100
  rdma/fabric5:       100
  rdma/fabric6:       100
  rdma/fabric7:       100
  rdma/fabric8:       100
  rdma/fabric9:       100
Allocatable:
  cpu:                223340m
  ephemeral-storage:  1945966499486
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             3617268336Ki
  nvidia.com/gpu:     8
  pods:               110
  rdma/fabric0:       100
  rdma/fabric1:       100
  rdma/fabric10:      100
  rdma/fabric11:      100
  rdma/fabric12:      100
  rdma/fabric13:      100
  rdma/fabric14:      100
  rdma/fabric15:      100
  rdma/fabric2:       100
  rdma/fabric3:       100
  rdma/fabric4:       100
  rdma/fabric5:       100
  rdma/fabric6:       100
  rdma/fabric7:       100
  rdma/fabric8:       100
  rdma/fabric9:       100
System Info:
  Machine ID:                 049042706d054e82bfbcb09514d81534
  System UUID:                04904270-6d05-4e82-bfbc-b09514d81534
  Boot ID:                    3c52ddd4-357e-445a-b197-aa534f323d48
  Kernel Version:             6.12.94+deb13-amd64
  OS Image:                   Debian GNU/Linux 13 (trixie)
  Operating System:           linux
  Architecture:               amd64
  Container Runtime Version:  containerd://2.2.3
  Kubelet Version:            v1.36.0
  Kube-Proxy Version:         
ProviderID:                   digitalocean://584880378
Non-terminated Pods:          (6 in total)
  Namespace                   Name                                        CPU Requests  CPU Limits  Memory Requests  Memory Limits  Age
  ---------                   ----                                        ------------  ----------  ---------------  -------------  ---
  kube-system                 cilium-2lrcg                                310m (0%)     1 (0%)      332Mi (0%)       1Gi (0%)       3h56m
  kube-system                 csi-do-node-hr7vp                           0 (0%)        0 (0%)      0 (0%)           0 (0%)         3h56m
  kube-system                 do-node-agent-nvidia-dcgm-exporter-jh22c    312m (0%)     1050m (0%)  312Mi (0%)       1356Mi (0%)    3h56m
  kube-system                 doks-telemetry-config-reloader-kx2jf        0 (0%)        0 (0%)      0 (0%)           0 (0%)         3h56m
  kube-system                 k8s-nvidia-device-plugin-d9d6t              0 (0%)        0 (0%)      0 (0%)           0 (0%)         3h56m
  kube-system                 rdma-shared-dp-ds-24pf8                     0 (0%)        0 (0%)      0 (0%)           0 (0%)         3h56m
Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests    Limits
  --------           --------    ------
  cpu                622m (0%)   2050m (0%)
  memory             644Mi (0%)  2380Mi (0%)
  ephemeral-storage  5Gi (0%)    5Gi (0%)
  hugepages-1Gi      0 (0%)      0 (0%)
  hugepages-2Mi      0 (0%)      0 (0%)
  nvidia.com/gpu     0           0
  rdma/fabric0       0           0
  rdma/fabric1       0           0
  rdma/fabric10      0           0
  rdma/fabric11      0           0
  rdma/fabric12      0           0
  rdma/fabric13      0           0
  rdma/fabric14      0           0
  rdma/fabric15      0           0
  rdma/fabric2       0           0
  rdma/fabric3       0           0
  rdma/fabric4       0           0
  rdma/fabric5       0           0
  rdma/fabric6       0           0
  rdma/fabric7       0           0
  rdma/fabric8       0           0
  rdma/fabric9       0           0
Events:              <none>
```

</details>

### Key Plugins 


#### do-node-agent-nvidia-dcgm-exporter

The do-node-agent-nvidia-dcgm-exporter DaemonSet provides NVIDIA GPU observability and node health reporting for DOKS clusters. It runs on every NVIDIA GPU node and collects detailed GPU telemetry—including utilization, memory, temperature, power, and other hardware metrics—using the nvidia-dcgm-container (DCGM Exporter). Metrics are exposed on a local endpoint (127.0.0.1:5000/metrics). A readiness component (nrr-status-patcher) continuously monitors this endpoint and updates the Kubernetes node condition doks.kubernetes.io/NvidiaGpuReady based on GPU health. The do-node-agent consumes both GPU and Kubernetes metrics and forwards them to DO's monitoring backend, enriching the data with cluster and node pool metadata for centralized monitoring and observability.

``` bash
kubectl -n kube-system describe daemonset do-node-agent-nvidia-dcgm-exporter

kubectl -n kube-system logs -f daemonset/do-node-agent-nvidia-dcgm-exporter 
kubectl -n kube-system get pods -o wide | grep do-node-agent-nvidia-dcgm-exporter
kubectl -n kube-system logs -f do-node-agent-nvidia-dcgm-exporter-4xjdt -c nrr-status-patcher
kubectl -n kube-system logs -f do-node-agent-nvidia-dcgm-exporter-4xjdt -c do-node-agent
kubectl -n kube-system logs -f do-node-agent-nvidia-dcgm-exporter-4xjdt -c nvidia-dcgm-container
```

#### k8s-nvidia-device-plugin

The k8s-nvidia-device-plugin DaemonSet exposes NVIDIA GPUs as schedulable Kubernetes resources by implementing the Kubernetes Device Plugin API. It runs on every NVIDIA GPU node and discovers GPU devices using the NVIDIA Container Toolkit and host driver interfaces. The plugin registers with kubelet through a gRPC socket under /var/lib/kubelet/device-plugins, allowing kubelet to advertise GPU resources (for example, nvidia.com/gpu) to the Kubernetes scheduler. As a result, GPU workloads are scheduled only on nodes with available NVIDIA GPU resources.

GPU devices are not manually mounted into pods. Instead, they are allocated through the Kubernetes Device Plugin framework. When a pod requests resources such as nvidia.com/gpu, kubelet invokes the NVIDIA Device Plugin's Allocate() gRPC call. The plugin selects the requested GPU devices and returns the container runtime configuration required to access them. Kubelet incorporates this information into the OCI container specification, and the container runtime (such as containerd or runc) injects the required NVIDIA device nodes, driver libraries, and environment variables into the container, making the allocated GPUs available to the workload.

The NVIDIA Device Plugin also supports the Container Device Interface (CDI). In DOKS, the DaemonSet mounts /var/run/cdi, enabling the plugin to generate CDI device specifications that describe the allocated GPUs and their required container configuration. CDI provides a standardized mechanism for runtimes to inject GPU devices into containers and is used by Kubernetes Dynamic Resource Allocation (DRA). With DRA, GPU resources are requested through ResourceClaims and managed by the NVIDIA DRA Resource Driver rather than the Device Plugin framework. Device Plugin and DRA are mutually exclusive allocation mechanisms for a given resource. While the Device Plugin can inject GPU devices directly, DRA relies on CDI specifications, which are consumed by the container runtime at container startup to configure GPU access.

``` bash
kubectl -n kube-system describe daemonset k8s-nvidia-device-plugin

kubectl -n kube-system logs -f daemonset/k8s-nvidia-device-plugin
kubectl -n kube-system get pods -o wide | grep k8s-nvidia-device-plugin
kubectl -n kube-system logs -f k8s-nvidia-device-plugin-8kslh 
```

#### rdma-shared-dp-ds 

The rdma-shared-dp-ds DaemonSet detects RDMA-capable hardware on the host, and registers RDMA resources with kubelet using the Kubernetes Device Plugin API over a gRPC socket under /var/lib/kubelet/device-plugins. Once registered, kubelet advertises RDMA capabilities as schedulable resources, allowing workloads to request and be allocated high-bandwidth, low-latency network access.

As with GPU devices, RDMA devices are not manually mounted into pods. Instead, they are exposed through the Kubernetes Device Plugin framework. When a pod requests RDMA resources, kubelet invokes the RDMA device plugin’s Allocate() gRPC call. The plugin selects the appropriate host devices and returns a container runtime specification describing the required device mappings, which kubelet translates into an OCI-compatible container specification. The container runtime (e.g., containerd/runc) performs the necessary bind mounts from the host into the container’s /dev namespace, making devices such as /dev/infiniband available inside the pod.

``` bash
kubectl -n kube-system describe daemonset rdma-shared-dp-ds

kubectl -n kube-system describe configmap rdma-devices 
kubectl -n kube-system logs -f daemonset/rdma-shared-dp-ds
kubectl -n kube-system get pods -o wide | grep rdma-shared-dp-ds
kubectl -n kube-system logs -f rdma-shared-dp-ds-24pf8
```

## Enabling RDMA Sharing 

### Step 1 - Multus CNI

[Install the Multus CNI plugin](https://docs.digitalocean.com/products/kubernetes/how-to/configure-multinode-gpus/#required-plugins):

``` bash
# kubectl apply -f https://raw.githubusercontent.com/k8snetworkplumbingwg/multus-cni/master/deployments/multus-daemonset-thick.yml

customresourcedefinition.apiextensions.k8s.io/network-attachment-definitions.k8s.cni.cncf.io created
clusterrole.rbac.authorization.k8s.io/multus created
clusterrolebinding.rbac.authorization.k8s.io/multus created
serviceaccount/multus created
configmap/multus-daemon-config created
daemonset.apps/kube-multus-ds created

# kubectl get pods -A | grep multus
kube-system   kube-multus-ds-6rmbq                       1/1     Running   0               20s
kube-system   kube-multus-ds-9cljn                       1/1     Running   0               20s
kube-system   kube-multus-ds-h8pl2                       1/1     Running   0               20s
kube-system   kube-multus-ds-rcnwt                       1/1     Running   0               20s
```

kube-multus-ds installs and configures Multus on every node, making it the primary CNI so that kubelet’s standard CNI request is routed through Multus.

When kubelet creates a Pod, it calls the container runtime, which then invokes the CNI plugin to set up networking. Because Multus is configured as the primary CNI in /etc/cni/net.d/, this request is first handled by Multus instead of a single CNI plugin. Multus reads the Pod’s network annotations and delegates interface setup to the appropriate CNI plugins (e.g., Cilium for eth0 and host-device for RDMA interfaces). These plugins attach their respective interfaces into the Pod’s network namespace, after which Multus returns the combined network configuration back through the CNI runtime to kubelet.

```
# DOKS Worker
# ls /etc/cni/net.d/
00-multus.conf	
05-cilium.conflist

# cat /etc/cni/net.d/00-multus.conf
{
  "cniVersion":"0.3.1",
  "logLevel":"verbose",
  "logToStderr":true,
  "name":"multus-cni-network",
  "clusterNetwork":"/host/etc/cni/net.d/05-cilium.conflist",
  "type":"multus-shim"
}
```

### Step 2 - Whereabouts IPAM CNI 

Install the Whereabouts IPAM CNI by following the instructions in [this guide](https://github.com/k8snetworkplumbingwg/whereabouts).

``` bash
# git clone https://github.com/k8snetworkplumbingwg/whereabouts && cd whereabouts
# kubectl apply \
    -f doc/crds/daemonset-install.yaml \
    -f doc/crds/whereabouts.cni.cncf.io_ippools.yaml \
    -f doc/crds/whereabouts.cni.cncf.io_overlappingrangeipreservations.yaml \
    -f doc/crds/reconciler-deployment.yaml

serviceaccount/whereabouts created
clusterrolebinding.rbac.authorization.k8s.io/whereabouts created
clusterrole.rbac.authorization.k8s.io/whereabouts-cni created
configmap/whereabouts-config created
daemonset.apps/whereabouts created
customresourcedefinition.apiextensions.k8s.io/ippools.whereabouts.cni.cncf.io created
customresourcedefinition.apiextensions.k8s.io/overlappingrangeipreservations.whereabouts.cni.cncf.io created
deployment.apps/whereabouts-reconciler created

# kubectl -n kube-system describe configmap whereabouts-config 
# kubectl -n kube-system describe daemonset whereabouts
```

### Step 3 - ipvlan CNI

The ipvlan CNI plugin is preinstalled in DOKS workers:

``` bash
ls -l /opt/cni/bin/

total 266488
-rw-r--r-- 1 root root    11357 Apr 24 10:38 LICENSE
-rw-r--r-- 1 root root     2343 Apr 24 10:38 README.md
-rwxr-xr-x 1 root root  5042926 Apr 24 10:38 bandwidth
-rwxr-xr-x 1 root root  5698763 Apr 24 10:38 bridge
-rwxr-xr-x 1 root root 81425208 Jul 15 18:12 cilium-cni
-rwxr-xr-x 1 root root 13725422 Apr 24 10:38 dhcp
-rwxr-xr-x 1 root root  5251069 Apr 24 10:38 dummy
-rwxr-xr-x 1 root root  5702145 Apr 24 10:38 firewall
-rwxr-xr-x 1 root root  5159967 Apr 24 10:38 host-device
-rwxr-xr-x 1 root root  4350778 Apr 24 10:38 host-local
-rwxr-xr-x 1 root root  5274322 Apr 24 10:38 ipvlan        # Preinstalled
-rwxr-xr-x 1 root root  4302030 Apr 24 10:38 loopback
-rwxr-xr-x 1 root root  5307111 Apr 24 10:38 macvlan
-rwxr-xr-x 1 root root 49476536 Jul 15 22:27 multus-shim
-rwxr-xr-x 1 root root  3719530 Jul 15 22:27 passthru
-rwxr-xr-x 1 root root  5108385 Apr 24 10:38 portmap
-rwxr-xr-x 1 root root  5475470 Apr 24 10:38 ptp
-rwxr-xr-x 1 root root  4525826 Apr 24 10:38 sbr
-rwxr-xr-x 1 root root  3776708 Apr 24 10:38 static
-rwxr-xr-x 1 root root  5453308 Apr 24 10:38 tap
-rwxr-xr-x 1 root root  4389084 Apr 24 10:38 tuning
-rwxr-xr-x 1 root root  5267679 Apr 24 10:38 vlan
-rwxr-xr-x 1 root root  4685012 Apr 24 10:38 vrf
-rwxr-xr-x 1 root root 39700247 Jul 15 22:31 whereabouts   # Installed by the Whereabouts IPAM CNI
```

### Step 4 - NAD


[NetworkAttachmentDefinitions](https://github.com/k8snetworkplumbingwg/multus-cni/blob/master/docs/how-to-use.md#create-network-attachment-definition) (NADs) define the configuration for secondary networks but do not create or attach network interfaces themselves. Instead, pods reference NADs through annotations, and Multus CNI uses the referenced definitions to create and attach the corresponding interfaces during pod creation. A NAD only needs to be defined once and can reside in any namespace, provided it is accessible to Multus when the pod is created.

Prepare and deploy [nvidia-mn-doks/ipvlan-network-attachments.yaml](rdma-sharing/ipvlan-network-attachments.yaml) configuration file to create virtual fabricX interfaces for pods while using Multus CNI with the ipvlan CNI plugin.

<details>
<summary>ipvlan-network-attachments.yaml</summary>

``` yaml
apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric0
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric0",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.100.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric1
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric1",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.101.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric2
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric2",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.102.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric3
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric3",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.103.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric4
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric4",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.104.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric5
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric5",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.105.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric6
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric6",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.106.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric7
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric7",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.107.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric8
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric8",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.108.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric9
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric9",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.109.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric10
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric10",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.110.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric11
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric11",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.111.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric12
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric12",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.112.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric13
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric13",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.113.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric14
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric14",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.114.0/24"
      }
    }'
---

apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: ipvlan-fabric15
spec:
  config: '{
      "cniVersion": "0.3.1",
      "type": "ipvlan",
      "master": "fabric15",
      "mode": "l2",
      "ipam": {  
        "type": "whereabouts",
        "range": "192.168.115.0/24"
      }
    }'
---
```
</details>
<br>


``` bash
# kubectl apply -f nvidia-mn-doks/ipvlan-network-attachments.yaml  

networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric0 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric1 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric2 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric3 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric4 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric5 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric6 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric7 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric8 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric9 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric10 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric11 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric12 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric13 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric14 created
networkattachmentdefinition.k8s.cni.cncf.io/ipvlan-fabric15 created

kubectl get network-attachment-definitions
kubectl describe network-attachment-definitions
```

## Summary – Pod GPU, RDMA and Network Setup Flow


| Component                                       | Action                                                          | Output                                                |
| ----------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------- |
| **kubelet**                                     | Creates the pod and invokes the CNI plugins                     | Starts the pod networking workflow                    |
| **Multus CNI**                                  | Orchestrates multiple CNI plugins                               | Attaches the primary and secondary network interfaces |
| **↳ Cilium (primary CNI)**                      | Configures the default Kubernetes pod network                   | `eth0` inside the pod                                 |
| **↳ ipvlan + Whereabouts IPAM (secondary CNI)** | Creates additional `ipvlan` interfaces and assigns IP addresses | `fabric0`–`fabric15` inside the pod                    |
| **RDMA Device Plugin**                          | Advertises RDMA resources and mounts RDMA device files          | `/dev/infiniband` inside the pod                      |
| **GPU Device Plugin**                           | Advertises GPU resources and mounts GPU device files            | `/dev/nvidia0`, `/dev/nvidiactl` and `/dev/nvidia-uvm` inside the pod              |

## Multi-Node Depolyment Example 

### Application Deployment

Create a Kubernetes ConfigMap that stores [nvidia-mn-doks/p2p_bw_test.py](nvidia-mn-doks/p2p_bw_test.py). The ConfigMap will be mounted into two Pods, allowing both Pods to access the same script for RDMA/RoCE testing.

``` bash
# Mount the test script into the Pods.
# kubectl create configmap mn-python-script --from-file=nvidia-mn-doks/p2p_bw_test.py

# kubectl describe configmap mn-python-script
# kubectl delete configmap mn-python-script
```

Build and deploy the Kubernetes manifest [nvidia-mn-doks/mn-ipvlan-6pods-3nodes.yaml](nvidia-mn-doks/mn-ipvlan-6pods-3nodes.yaml) to create four 6-GPU Pods across three DOKS workers with RDMA sharing enabled on all RDMA devices. 

<details>
<summary>mn-ipvlan-6pods-3nodes.yaml</summary>

``` yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mn-ipvlan
spec:
  replicas: 6
  selector:
    matchLabels:
      app: mn-ipvlan
  template:
    metadata:
      labels:
        app: mn-ipvlan
      annotations:
        k8s.v1.cni.cncf.io/networks: >-
          ipvlan-fabric0@net0,
          ipvlan-fabric1@net1,
          ipvlan-fabric2@net2,
          ipvlan-fabric3@net3,
          ipvlan-fabric4@net4,
          ipvlan-fabric5@net5,
          ipvlan-fabric6@net6,
          ipvlan-fabric7@net7,
          ipvlan-fabric8@net8,
          ipvlan-fabric9@net9,
          ipvlan-fabric10@net10,
          ipvlan-fabric11@net11,
          ipvlan-fabric12@net12,
          ipvlan-fabric13@net13,
          ipvlan-fabric14@net14,
          ipvlan-fabric15@net15
    spec:
      restartPolicy: Always

      nodeSelector:
        doks.digitalocean.com/gpu-brand: nvidia

      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

      volumes:
        - name: temp-hf-cache
          hostPath:
            path: /root/.cache/huggingface
            type: DirectoryOrCreate

        - name: dshm 
          emptyDir:
            medium: Memory
            sizeLimit: 128Gi

        - name: temp-python-script       
          configMap:
            name: mn-python-script

      containers:

        - name: server
          image: pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
              name: http-example

          command: ["/bin/bash", "-lc"]
          args:
            - |
             set -ex
          
             apt update
             apt install -y iproute2 iputils-ping infiniband-diags ibverbs-utils net-tools
             

             exec sleep infinity 
 
          securityContext:

            capabilities:
              add:
              - IPC_LOCK

          volumeMounts:
            - name: temp-hf-cache
              mountPath: /root/.cache/huggingface

            - name: dshm
              mountPath: /dev/shm

            - name: temp-python-script
              mountPath: /workspace/p2p_bw_test.py
              subPath: p2p_bw_test.py

          resources:
            limits:
              nvidia.com/gpu: "4"
              rdma/fabric0: 10
              rdma/fabric1: 10
              rdma/fabric2: 10
              rdma/fabric3: 10
              rdma/fabric4: 10
              rdma/fabric5: 10
              rdma/fabric6: 10
              rdma/fabric7: 10
              rdma/fabric8: 10
              rdma/fabric9: 10
              rdma/fabric10: 10
              rdma/fabric11: 10
              rdma/fabric12: 10
              rdma/fabric13: 10
              rdma/fabric13: 10
              rdma/fabric14: 10
              rdma/fabric15: 10
            requests:
              nvidia.com/gpu: "4"
              rdma/fabric0: 10
              rdma/fabric1: 10
              rdma/fabric2: 10
              rdma/fabric3: 10
              rdma/fabric4: 10
              rdma/fabric5: 10
              rdma/fabric6: 10
              rdma/fabric7: 10
              rdma/fabric8: 10
              rdma/fabric9: 10
              rdma/fabric10: 10
              rdma/fabric11: 10
              rdma/fabric12: 10
              rdma/fabric13: 10
              rdma/fabric13: 10
              rdma/fabric14: 10
              rdma/fabric15: 10
```

</details>
</br>



<details>
<summary>B300x8 DOKS</summary>

``` bash
# kubectl get nodes -o wide
NAME                       STATUS   ROLES    AGE     VERSION   INTERNAL-IP   EXTERNAL-IP       OS-IMAGE                       KERNEL-VERSION                CONTAINER-RUNTIME
rs-mn-b300x8-pool-37ylk3   Ready    <none>   3h52m   v1.36.0   10.132.0.7    165.245.199.217   Debian GNU/Linux 13 (trixie)   6.12.94+deb13-amd64 (amd64)   containerd://2.2.3
rs-mn-b300x8-pool-37ylk8   Ready    <none>   3h52m   v1.36.0   10.132.0.5    165.245.199.203   Debian GNU/Linux 13 (trixie)   6.12.94+deb13-amd64 (amd64)   containerd://2.2.3
rs-mn-b300x8-pool-37ylku   Ready    <none>   3h52m   v1.36.0   10.132.0.8    165.245.199.226   Debian GNU/Linux 13 (trixie)   6.12.94+deb13-amd64 (amd64)   containerd://2.2.3
rs-mn-cpu-pool-37ylkn      Ready    <none>   4h7m    v1.36.0   10.132.0.6    165.245.199.211   Debian GNU/Linux 13 (trixie)   6.12.94+deb13-amd64 (amd64)   containerd://2.2.3

# kubectl apply -f nvidia-mn-doks/mn-ipvlan-6pods-3nodes.yaml

# kubectl get pods -o wide
NAME          READY   STATUS    RESTARTS   AGE   IP             NODE                       NOMINATED NODE   READINESS GATES
mn-ipvlan-0   1/1     Running   0          66s   10.151.0.245   rs-mn-b300x8-pool-37ylk3   <none>           <none>
mn-ipvlan-1   1/1     Running   0          58s   10.151.1.36    rs-mn-b300x8-pool-37ylku   <none>           <none>
mn-ipvlan-2   1/1     Running   0          49s   10.151.1.170   rs-mn-b300x8-pool-37ylk8   <none>           <none>
mn-ipvlan-3   1/1     Running   0          41s   10.151.0.229   rs-mn-b300x8-pool-37ylk3   <none>           <none>
mn-ipvlan-4   1/1     Running   0          33s   10.151.1.160   rs-mn-b300x8-pool-37ylk8   <none>           <none>
mn-ipvlan-5   1/1     Running   0          24s   10.151.1.119   rs-mn-b300x8-pool-37ylku   <none>           <none>

# kubectl delete -f nvidia-mn-doks/mn-ipvlan-6pods-3nodes.yaml

# 20 shares are used for each RDMA device on the DOKS worker
# kubectl describe node rs-mn-b300x8-pool-37ylku
Name:               rs-mn-b300x8-pool-37ylku
Roles:              <none>
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/instance-type=gpu-b300x8-2304gb-lc-fabric-contracted
                    beta.kubernetes.io/os=linux
                    doks.digitalocean.com/gpu-brand=nvidia
                    doks.digitalocean.com/gpu-model=b300
                    doks.digitalocean.com/is-gpu-fabric-connected=true
                    doks.digitalocean.com/managed=true
                    doks.digitalocean.com/node-id=1fe11d5e-dd76-4c03-9e98-00711f742baa
                    doks.digitalocean.com/node-pool=rs-mn-b300x8-pool
                    doks.digitalocean.com/node-pool-id=512834ce-cb3d-4922-9fee-91fdd4ec92cc
                    doks.digitalocean.com/nvidia-dcgm-enabled=true
                    doks.digitalocean.com/version=1.36.0-do.2
                    failure-domain.beta.kubernetes.io/region=mkc1
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=rs-mn-b300x8-pool-37ylku
                    kubernetes.io/os=linux
                    node.kubernetes.io/instance-type=gpu-b300x8-2304gb-lc-fabric-contracted
                    nvidia.com/gpu=8
                    region=mkc1
                    topology.kubernetes.io/region=mkc1
Annotations:        alpha.kubernetes.io/provided-node-ip: 10.132.0.8
                    csi.volume.kubernetes.io/nodeid: {"dobs.csi.digitalocean.com":"584880378"}
                    network.cilium.io/ipv4-Ingress-ip: 10.151.1.40
                    network.cilium.io/ipv4-cilium-host: 10.151.1.62
                    network.cilium.io/ipv4-health-ip: 10.151.1.11
                    network.cilium.io/ipv4-pod-cidr: 10.151.1.0/25
                    node.alpha.kubernetes.io/ttl: 0
                    volumes.kubernetes.io/controller-managed-attach-detach: true
CreationTimestamp:  Wed, 15 Jul 2026 18:12:03 +0000
Taints:             nvidia.com/gpu:NoSchedule
Unschedulable:      false
Lease:
  HolderIdentity:  rs-mn-b300x8-pool-37ylku
  AcquireTime:     <unset>
  RenewTime:       Wed, 15 Jul 2026 23:49:25 +0000
Conditions:
  Type                                  Status  LastHeartbeatTime                 LastTransitionTime                Reason                       Message
  ----                                  ------  -----------------                 ------------------                ------                       -------
  NetworkUnavailable                    False   Wed, 15 Jul 2026 18:12:29 +0000   Wed, 15 Jul 2026 18:12:29 +0000   CiliumIsUp                   Cilium is running on this node
  MemoryPressure                        False   Wed, 15 Jul 2026 23:49:27 +0000   Wed, 15 Jul 2026 18:12:03 +0000   KubeletHasSufficientMemory   kubelet has sufficient memory available
  DiskPressure                          False   Wed, 15 Jul 2026 23:49:27 +0000   Wed, 15 Jul 2026 18:12:03 +0000   KubeletHasNoDiskPressure     kubelet has no disk pressure
  PIDPressure                           False   Wed, 15 Jul 2026 23:49:27 +0000   Wed, 15 Jul 2026 18:12:03 +0000   KubeletHasSufficientPID      kubelet has sufficient PID available
  Ready                                 True    Wed, 15 Jul 2026 23:49:27 +0000   Wed, 15 Jul 2026 18:12:22 +0000   KubeletReady                 kubelet is posting ready status
  doks.kubernetes.io/NvidiaGpuReady     True    Wed, 15 Jul 2026 23:49:12 +0000   Wed, 15 Jul 2026 18:13:12 +0000   EndpointOK                   Endpoint reports ready at http://127.0.0.1:5000/metrics
  doks.kubernetes.io/CiliumAgentReady   True    Wed, 15 Jul 2026 23:49:26 +0000   Wed, 15 Jul 2026 18:12:51 +0000   EndpointOK                   Endpoint reports ready at http://127.0.0.1:9879/healthz
Addresses:
  InternalIP:  10.132.0.8
  Hostname:    rs-mn-b300x8-pool-37ylku
  ExternalIP:  165.245.199.226
Capacity:
  cpu:                224
  ephemeral-storage:  2111508792Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             3699199600Ki
  nvidia.com/gpu:     8
  pods:               110
  rdma/fabric0:       100
  rdma/fabric1:       100
  rdma/fabric10:      100
  rdma/fabric11:      100
  rdma/fabric12:      100
  rdma/fabric13:      100
  rdma/fabric14:      100
  rdma/fabric15:      100
  rdma/fabric2:       100
  rdma/fabric3:       100
  rdma/fabric4:       100
  rdma/fabric5:       100
  rdma/fabric6:       100
  rdma/fabric7:       100
  rdma/fabric8:       100
  rdma/fabric9:       100
Allocatable:
  cpu:                223340m
  ephemeral-storage:  1945966499486
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             3617268336Ki
  nvidia.com/gpu:     8
  pods:               110
  rdma/fabric0:       100
  rdma/fabric1:       100
  rdma/fabric10:      100
  rdma/fabric11:      100
  rdma/fabric12:      100
  rdma/fabric13:      100
  rdma/fabric14:      100
  rdma/fabric15:      100
  rdma/fabric2:       100
  rdma/fabric3:       100
  rdma/fabric4:       100
  rdma/fabric5:       100
  rdma/fabric6:       100
  rdma/fabric7:       100
  rdma/fabric8:       100
  rdma/fabric9:       100
System Info:
  Machine ID:                 049042706d054e82bfbcb09514d81534
  System UUID:                04904270-6d05-4e82-bfbc-b09514d81534
  Boot ID:                    3c52ddd4-357e-445a-b197-aa534f323d48
  Kernel Version:             6.12.94+deb13-amd64
  OS Image:                   Debian GNU/Linux 13 (trixie)
  Operating System:           linux
  Architecture:               amd64
  Container Runtime Version:  containerd://2.2.3
  Kubelet Version:            v1.36.0
  Kube-Proxy Version:         
ProviderID:                   digitalocean://584880378
Non-terminated Pods:          (10 in total)
  Namespace                   Name                                        CPU Requests  CPU Limits  Memory Requests  Memory Limits  Age
  ---------                   ----                                        ------------  ----------  ---------------  -------------  ---
  default                     mn-ipvlan-1                                 0 (0%)        0 (0%)      0 (0%)           0 (0%)         8m14s
  default                     mn-ipvlan-5                                 0 (0%)        0 (0%)      0 (0%)           0 (0%)         7m40s
  kube-system                 cilium-2lrcg                                310m (0%)     1 (0%)      332Mi (0%)       1Gi (0%)       5h37m
  kube-system                 csi-do-node-hr7vp                           0 (0%)        0 (0%)      0 (0%)           0 (0%)         5h37m
  kube-system                 do-node-agent-nvidia-dcgm-exporter-jh22c    312m (0%)     1050m (0%)  312Mi (0%)       1356Mi (0%)    5h37m
  kube-system                 doks-telemetry-config-reloader-kx2jf        0 (0%)        0 (0%)      0 (0%)           0 (0%)         5h37m
  kube-system                 k8s-nvidia-device-plugin-d9d6t              0 (0%)        0 (0%)      0 (0%)           0 (0%)         5h37m
  kube-system                 kube-multus-ds-rcnwt                        100m (0%)     100m (0%)   50Mi (0%)        50Mi (0%)      81m
  kube-system                 rdma-shared-dp-ds-24pf8                     0 (0%)        0 (0%)      0 (0%)           0 (0%)         5h37m
  kube-system                 whereabouts-v5qff                           100m (0%)     100m (0%)   100Mi (0%)       200Mi (0%)     77m
Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests    Limits
  --------           --------    ------
  cpu                822m (0%)   2250m (1%)
  memory             794Mi (0%)  2630Mi (0%)
  ephemeral-storage  5Gi (0%)    5Gi (0%)
  hugepages-1Gi      0 (0%)      0 (0%)
  hugepages-2Mi      0 (0%)      0 (0%)
  nvidia.com/gpu     8           8
  rdma/fabric0       20          20
  rdma/fabric1       20          20
  rdma/fabric10      20          20
  rdma/fabric11      20          20
  rdma/fabric12      20          20
  rdma/fabric13      20          20
  rdma/fabric14      20          20
  rdma/fabric15      20          20
  rdma/fabric2       20          20
  rdma/fabric3       20          20
  rdma/fabric4       20          20
  rdma/fabric5       20          20
  rdma/fabric6       20          20
  rdma/fabric7       20          20
  rdma/fabric8       20          20
  rdma/fabric9       20          20
Events:              <none>
```

</details>
</br>


### GID on the Host

<details>
<summary>B300x8: Host OS</summary>

``` bash
root@rs-mn-b300x8-pool-37ylku:~# ip -br a | grep fabric
fabric0          UP             fe80::ac3a:e2ff:febc:9132/64 
fabric1          UP             fe80::ac3a:e2ff:febc:9133/64 
fabric2          UP             fe80::ac3a:e2ff:febc:9152/64 
fabric3          UP             fe80::ac3a:e2ff:febc:9153/64 
fabric4          UP             fe80::ac3a:e2ff:febc:9162/64 
fabric5          UP             fe80::ac3a:e2ff:febc:9163/64 
fabric6          UP             fe80::ac3a:e2ff:febc:9142/64 
fabric7          UP             fe80::ac3a:e2ff:febc:9143/64 
fabric8          UP             fe80::ac3a:e2ff:febc:91a2/64 
fabric9          UP             fe80::ac3a:e2ff:febc:91a3/64 
fabric10         UP             fe80::ac3a:e2ff:febc:9182/64 
fabric11         UP             fe80::ac3a:e2ff:febc:9183/64 
fabric12         UP             fe80::ac3a:e2ff:febc:9172/64 
fabric13         UP             fe80::ac3a:e2ff:febc:9173/64 
fabric14         UP             fe80::ac3a:e2ff:febc:9192/64 
fabric15         UP             fe80::ac3a:e2ff:febc:9193/64 

root@rs-mn-b300x8-pool-37ylku:~# ibv_devinfo -v | grep GID
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9132, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9132, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6402, RoCE v1
                        GID[  3]:               ::ffff:192.168.100.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9132, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9132, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6406, RoCE v1
                        GID[  7]:               ::ffff:192.168.100.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9132, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9132, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9133, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9133, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6502, RoCE v1
                        GID[  3]:               ::ffff:192.168.101.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9133, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9133, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6506, RoCE v1
                        GID[  7]:               ::ffff:192.168.101.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9133, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9133, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9152, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9152, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6602, RoCE v1
                        GID[  3]:               ::ffff:192.168.102.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9152, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9152, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6606, RoCE v1
                        GID[  7]:               ::ffff:192.168.102.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9152, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9152, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9153, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9153, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6702, RoCE v1
                        GID[  3]:               ::ffff:192.168.103.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9153, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9153, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6706, RoCE v1
                        GID[  7]:               ::ffff:192.168.103.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9153, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9153, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9162, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9162, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6802, RoCE v1
                        GID[  3]:               ::ffff:192.168.104.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9162, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9162, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6806, RoCE v1
                        GID[  7]:               ::ffff:192.168.104.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9162, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9162, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9163, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9163, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6902, RoCE v1
                        GID[  3]:               ::ffff:192.168.105.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9163, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9163, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6906, RoCE v1
                        GID[  7]:               ::ffff:192.168.105.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9163, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9163, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9142, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9142, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6a02, RoCE v1
                        GID[  3]:               ::ffff:192.168.106.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9142, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9142, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6a06, RoCE v1
                        GID[  7]:               ::ffff:192.168.106.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9142, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9142, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9143, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9143, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6b02, RoCE v1
                        GID[  3]:               ::ffff:192.168.107.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9143, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9143, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6b06, RoCE v1
                        GID[  7]:               ::ffff:192.168.107.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9143, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9143, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:91a2, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:91a2, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6c02, RoCE v1
                        GID[  3]:               ::ffff:192.168.108.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:91a2, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:91a2, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6c06, RoCE v1
                        GID[  7]:               ::ffff:192.168.108.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:91a2, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:91a2, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:91a3, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:91a3, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6d02, RoCE v1
                        GID[  3]:               ::ffff:192.168.109.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:91a3, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:91a3, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6d06, RoCE v1
                        GID[  7]:               ::ffff:192.168.109.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:91a3, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:91a3, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9182, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9182, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6e02, RoCE v1
                        GID[  3]:               ::ffff:192.168.110.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9182, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9182, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6e06, RoCE v1
                        GID[  7]:               ::ffff:192.168.110.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9182, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9182, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9183, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9183, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:6f02, RoCE v1
                        GID[  3]:               ::ffff:192.168.111.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9183, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9183, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:6f06, RoCE v1
                        GID[  7]:               ::ffff:192.168.111.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9183, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9183, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9172, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9172, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:7002, RoCE v1
                        GID[  3]:               ::ffff:192.168.112.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9172, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9172, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:7006, RoCE v1
                        GID[  7]:               ::ffff:192.168.112.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9172, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9172, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9173, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9173, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:7102, RoCE v1
                        GID[  3]:               ::ffff:192.168.113.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9173, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9173, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:7106, RoCE v1
                        GID[  7]:               ::ffff:192.168.113.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9173, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9173, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9192, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9192, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:7202, RoCE v1
                        GID[  3]:               ::ffff:192.168.114.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9192, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9192, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:7206, RoCE v1
                        GID[  7]:               ::ffff:192.168.114.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9192, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9192, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e2ff:febc:9193, RoCE v1
                        GID[  1]:               fe80::ac3a:e2ff:febc:9193, RoCE v2
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:7302, RoCE v1
                        GID[  3]:               ::ffff:192.168.115.2, RoCE v2
                        GID[  4]:               fe80:0000:0000:0000:ae3a:e200:01bc:9193, RoCE v1
                        GID[  5]:               fe80::ae3a:e200:1bc:9193, RoCE v2
                        GID[  6]:               0000:0000:0000:0000:0000:ffff:c0a8:7306, RoCE v1
                        GID[  7]:               ::ffff:192.168.115.6, RoCE v2
                        GID[  8]:               fe80:0000:0000:0000:ae3a:e200:02bc:9193, RoCE v1
                        GID[  9]:               fe80::ae3a:e200:2bc:9193, RoCE v2
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e203:00bc:912e
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e203:00bc:912f
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e203:00bc:9130
                        GID[  0]:               fe80:0000:0000:0000:ac3a:e203:00bc:9131
```
</details>
</br>

### Pod IP Assignment, Routing and GID

<details>
<summary>B300x8: Within the pods</summary>

``` bash
root@mn-ipvlan-1:/workspace# nvidia-smi
Thu Jul 16 00:01:46 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA B300 SXM6 PC            On  |   00000000:85:00.0 Off |                    0 |
| N/A   29C    P0            234W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA B300 SXM6 PC            On  |   00000000:8D:00.0 Off |                    0 |
| N/A   29C    P0            238W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA B300 SXM6 PC            On  |   00000000:95:00.0 Off |                    0 |
| N/A   29C    P0            238W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA B300 SXM6 PC            On  |   00000000:9D:00.0 Off |                    0 |
| N/A   29C    P0            236W / 1100W |       0MiB / 275040MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

root@mn-ipvlan-1:/workspace# ls -l /dev/nvidia*
crw-rw-rw- 1 root root 195, 254 Jul 15 23:41 /dev/nvidia-modeset
crw-rw-rw- 1 root root 235,   0 Jul 15 23:41 /dev/nvidia-uvm
crw-rw-rw- 1 root root 235,   1 Jul 15 23:41 /dev/nvidia-uvm-tools
crw-rw-rw- 1 root root 195,   0 Jul 15 23:41 /dev/nvidia0
crw-rw-rw- 1 root root 195,   1 Jul 15 23:41 /dev/nvidia1
crw-rw-rw- 1 root root 195,   2 Jul 15 23:41 /dev/nvidia2
crw-rw-rw- 1 root root 195,   3 Jul 15 23:41 /dev/nvidia3
crw-rw-rw- 1 root root 195, 255 Jul 15 23:41 /dev/nvidiactl

/dev/nvidia-caps:
total 0
cr-------- 1 root root 241, 1 Jul 16 00:01 nvidia-cap1
cr--r--r-- 1 root root 241, 2 Jul 16 00:01 nvidia-cap2

root@mn-ipvlan-1:/workspace# rdma link
link mlx5_0/1 state ACTIVE physical_state LINK_UP 
link mlx5_1/1 state ACTIVE physical_state LINK_UP 
link mlx5_2/1 state ACTIVE physical_state LINK_UP 
link mlx5_3/1 state ACTIVE physical_state LINK_UP 
link mlx5_4/1 state ACTIVE physical_state LINK_UP 
link mlx5_5/1 state ACTIVE physical_state LINK_UP 
link mlx5_6/1 state ACTIVE physical_state LINK_UP 
link mlx5_7/1 state ACTIVE physical_state LINK_UP 
link mlx5_8/1 state ACTIVE physical_state LINK_UP 
link mlx5_9/1 state ACTIVE physical_state LINK_UP 
link mlx5_10/1 state ACTIVE physical_state LINK_UP 
link mlx5_11/1 state ACTIVE physical_state LINK_UP 
link mlx5_12/1 state ACTIVE physical_state LINK_UP 
link mlx5_13/1 state ACTIVE physical_state LINK_UP 
link mlx5_14/1 state ACTIVE physical_state LINK_UP 
link mlx5_15/1 state ACTIVE physical_state LINK_UP 
link mlx5_16/1 subnet_prefix fe80:0000:0000:0000 lid 1 sm_lid 1 lmc 0 state ACTIVE physical_state LINK_UP 
link mlx5_17/1 subnet_prefix fe80:0000:0000:0000 lid 3 sm_lid 1 lmc 0 state ACTIVE physical_state LINK_UP 
link mlx5_18/1 subnet_prefix fe80:0000:0000:0000 lid 6 sm_lid 1 lmc 0 state ACTIVE physical_state LINK_UP 
link mlx5_19/1 subnet_prefix fe80:0000:0000:0000 lid 5 sm_lid 1 lmc 0 state ACTIVE physical_state LINK_UP 

root@mn-ipvlan-1:/workspace# ibv_devices  
    device          	   node GUID
    ------          	----------------
    mlx5_0          	ae3ae2fffebc9132
    mlx5_1          	ae3ae2fffebc9133
    mlx5_2          	ae3ae2fffebc9152
    mlx5_3          	ae3ae2fffebc9153
    mlx5_4          	ae3ae2fffebc9162
    mlx5_5          	ae3ae2fffebc9163
    mlx5_6          	ae3ae2fffebc9142
    mlx5_7          	ae3ae2fffebc9143
    mlx5_8          	ae3ae2fffebc91a2
    mlx5_9          	ae3ae2fffebc91a3
    mlx5_10         	ae3ae2fffebc9182
    mlx5_11         	ae3ae2fffebc9183
    mlx5_12         	ae3ae2fffebc9172
    mlx5_13         	ae3ae2fffebc9173
    mlx5_14         	ae3ae2fffebc9192
    mlx5_15         	ae3ae2fffebc9193

root@mn-ipvlan-1:/workspace#  ls /dev/infiniband/ -ls
total 0
0 crw-rw-rw- 1 root root  10, 262 Jul 15 23:41 rdma_cm
0 crw------- 1 root root 231,   0 Jul 15 23:41 umad0
0 crw------- 1 root root 231,   1 Jul 15 23:41 umad1
0 crw------- 1 root root 231,  10 Jul 15 23:41 umad10
0 crw------- 1 root root 231,  11 Jul 15 23:41 umad11
0 crw------- 1 root root 231,  12 Jul 15 23:41 umad12
0 crw------- 1 root root 231,  13 Jul 15 23:41 umad13
0 crw------- 1 root root 231,  14 Jul 15 23:41 umad14
0 crw------- 1 root root 231,  15 Jul 15 23:41 umad15
0 crw------- 1 root root 231,   2 Jul 15 23:41 umad2
0 crw------- 1 root root 231,   3 Jul 15 23:41 umad3
0 crw------- 1 root root 231,   4 Jul 15 23:41 umad4
0 crw------- 1 root root 231,   5 Jul 15 23:41 umad5
0 crw------- 1 root root 231,   6 Jul 15 23:41 umad6
0 crw------- 1 root root 231,   7 Jul 15 23:41 umad7
0 crw------- 1 root root 231,   8 Jul 15 23:41 umad8
0 crw------- 1 root root 231,   9 Jul 15 23:41 umad9
0 crw-rw-rw- 1 root root 231, 192 Jul 15 23:41 uverbs0
0 crw-rw-rw- 1 root root 231, 193 Jul 15 23:41 uverbs1
0 crw-rw-rw- 1 root root 231, 202 Jul 15 23:41 uverbs10
0 crw-rw-rw- 1 root root 231, 203 Jul 15 23:41 uverbs11
0 crw-rw-rw- 1 root root 231, 204 Jul 15 23:41 uverbs12
0 crw-rw-rw- 1 root root 231, 205 Jul 15 23:41 uverbs13
0 crw-rw-rw- 1 root root 231, 206 Jul 15 23:41 uverbs14
0 crw-rw-rw- 1 root root 231, 207 Jul 15 23:41 uverbs15
0 crw-rw-rw- 1 root root 231, 194 Jul 15 23:41 uverbs2
0 crw-rw-rw- 1 root root 231, 195 Jul 15 23:41 uverbs3
0 crw-rw-rw- 1 root root 231, 196 Jul 15 23:41 uverbs4
0 crw-rw-rw- 1 root root 231, 197 Jul 15 23:41 uverbs5
0 crw-rw-rw- 1 root root 231, 198 Jul 15 23:41 uverbs6
0 crw-rw-rw- 1 root root 231, 199 Jul 15 23:41 uverbs7
0 crw-rw-rw- 1 root root 231, 200 Jul 15 23:41 uverbs8
0 crw-rw-rw- 1 root root 231, 201 Jul 15 23:41 uverbs9

root@mn-ipvlan-1:/workspace# ip -br a
lo               UNKNOWN        127.0.0.1/8 ::1/128 
net0@if4         UNKNOWN        192.168.100.2/24 fe80::ae3a:e200:1bc:9132/64 
net1@if5         UNKNOWN        192.168.101.2/24 fe80::ae3a:e200:1bc:9133/64 
net2@if6         UNKNOWN        192.168.102.2/24 fe80::ae3a:e200:1bc:9152/64 
net3@if7         UNKNOWN        192.168.103.2/24 fe80::ae3a:e200:1bc:9153/64 
net4@if8         UNKNOWN        192.168.104.2/24 fe80::ae3a:e200:1bc:9162/64 
net5@if9         UNKNOWN        192.168.105.2/24 fe80::ae3a:e200:1bc:9163/64 
net6@if10        UNKNOWN        192.168.106.2/24 fe80::ae3a:e200:1bc:9142/64 
net7@if11        UNKNOWN        192.168.107.2/24 fe80::ae3a:e200:1bc:9143/64 
net8@if12        UNKNOWN        192.168.108.2/24 fe80::ae3a:e200:1bc:91a2/64 
net9@if13        UNKNOWN        192.168.109.2/24 fe80::ae3a:e200:1bc:91a3/64 
net10@if14       UNKNOWN        192.168.110.2/24 fe80::ae3a:e200:1bc:9182/64 
net11@if15       UNKNOWN        192.168.111.2/24 fe80::ae3a:e200:1bc:9183/64 
net12@if16       UNKNOWN        192.168.112.2/24 fe80::ae3a:e200:1bc:9172/64 
net13@if17       UNKNOWN        192.168.113.2/24 fe80::ae3a:e200:1bc:9173/64 
net14@if18       UNKNOWN        192.168.114.2/24 fe80::ae3a:e200:1bc:9192/64 
net15@if19       UNKNOWN        192.168.115.2/24 fe80::ae3a:e200:1bc:9193/64 
eth0@if49        UP             10.151.1.36/32 fe80::44f4:6eff:fec0:af57/64 

root@mn-ipvlan-1:/workspace# ibv_devinfo -v | grep GID
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9132, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9132, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6402, RoCE v1
			GID[  3]:		::ffff:192.168.100.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9132, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9132, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6406, RoCE v1
			GID[  7]:		::ffff:192.168.100.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9132, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9132, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9133, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9133, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6502, RoCE v1
			GID[  3]:		::ffff:192.168.101.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9133, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9133, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6506, RoCE v1
			GID[  7]:		::ffff:192.168.101.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9133, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9133, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9152, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9152, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6602, RoCE v1
			GID[  3]:		::ffff:192.168.102.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9152, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9152, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6606, RoCE v1
			GID[  7]:		::ffff:192.168.102.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9152, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9152, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9153, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9153, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6702, RoCE v1
			GID[  3]:		::ffff:192.168.103.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9153, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9153, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6706, RoCE v1
			GID[  7]:		::ffff:192.168.103.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9153, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9153, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9162, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9162, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6802, RoCE v1
			GID[  3]:		::ffff:192.168.104.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9162, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9162, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6806, RoCE v1
			GID[  7]:		::ffff:192.168.104.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9162, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9162, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9163, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9163, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6902, RoCE v1
			GID[  3]:		::ffff:192.168.105.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9163, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9163, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6906, RoCE v1
			GID[  7]:		::ffff:192.168.105.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9163, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9163, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9142, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9142, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6a02, RoCE v1
			GID[  3]:		::ffff:192.168.106.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9142, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9142, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6a06, RoCE v1
			GID[  7]:		::ffff:192.168.106.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9142, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9142, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9143, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9143, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6b02, RoCE v1
			GID[  3]:		::ffff:192.168.107.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9143, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9143, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6b06, RoCE v1
			GID[  7]:		::ffff:192.168.107.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9143, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9143, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:91a2, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:91a2, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6c02, RoCE v1
			GID[  3]:		::ffff:192.168.108.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:91a2, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:91a2, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6c06, RoCE v1
			GID[  7]:		::ffff:192.168.108.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:91a2, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:91a2, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:91a3, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:91a3, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6d02, RoCE v1
			GID[  3]:		::ffff:192.168.109.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:91a3, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:91a3, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6d06, RoCE v1
			GID[  7]:		::ffff:192.168.109.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:91a3, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:91a3, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9182, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9182, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6e02, RoCE v1
			GID[  3]:		::ffff:192.168.110.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9182, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9182, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6e06, RoCE v1
			GID[  7]:		::ffff:192.168.110.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9182, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9182, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9183, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9183, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:6f02, RoCE v1
			GID[  3]:		::ffff:192.168.111.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9183, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9183, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:6f06, RoCE v1
			GID[  7]:		::ffff:192.168.111.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9183, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9183, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9172, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9172, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:7002, RoCE v1
			GID[  3]:		::ffff:192.168.112.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9172, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9172, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:7006, RoCE v1
			GID[  7]:		::ffff:192.168.112.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9172, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9172, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9173, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9173, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:7102, RoCE v1
			GID[  3]:		::ffff:192.168.113.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9173, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9173, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:7106, RoCE v1
			GID[  7]:		::ffff:192.168.113.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9173, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9173, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9192, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9192, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:7202, RoCE v1
			GID[  3]:		::ffff:192.168.114.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9192, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9192, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:7206, RoCE v1
			GID[  7]:		::ffff:192.168.114.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9192, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9192, RoCE v2
			GID[  0]:		fe80:0000:0000:0000:ac3a:e2ff:febc:9193, RoCE v1
			GID[  1]:		fe80::ac3a:e2ff:febc:9193, RoCE v2
			GID[  2]:		0000:0000:0000:0000:0000:ffff:c0a8:7302, RoCE v1
			GID[  3]:		::ffff:192.168.115.2, RoCE v2
			GID[  4]:		fe80:0000:0000:0000:ae3a:e200:01bc:9193, RoCE v1
			GID[  5]:		fe80::ae3a:e200:1bc:9193, RoCE v2
			GID[  6]:		0000:0000:0000:0000:0000:ffff:c0a8:7306, RoCE v1
			GID[  7]:		::ffff:192.168.115.6, RoCE v2
			GID[  8]:		fe80:0000:0000:0000:ae3a:e200:02bc:9193, RoCE v1
			GID[  9]:		fe80::ae3a:e200:2bc:9193, RoCE v2

root@mn-ipvlan-1:/workspace# ip route
default via 10.151.1.62 dev eth0 mtu 1500 
10.151.1.62 dev eth0 scope link 
192.168.100.0/24 dev net0 proto kernel scope link src 192.168.100.2 
192.168.101.0/24 dev net1 proto kernel scope link src 192.168.101.2 
192.168.102.0/24 dev net2 proto kernel scope link src 192.168.102.2 
192.168.103.0/24 dev net3 proto kernel scope link src 192.168.103.2 
192.168.104.0/24 dev net4 proto kernel scope link src 192.168.104.2 
192.168.105.0/24 dev net5 proto kernel scope link src 192.168.105.2 
192.168.106.0/24 dev net6 proto kernel scope link src 192.168.106.2 
192.168.107.0/24 dev net7 proto kernel scope link src 192.168.107.2 
192.168.108.0/24 dev net8 proto kernel scope link src 192.168.108.2 
192.168.109.0/24 dev net9 proto kernel scope link src 192.168.109.2 
192.168.110.0/24 dev net10 proto kernel scope link src 192.168.110.2 
192.168.111.0/24 dev net11 proto kernel scope link src 192.168.111.2 
192.168.112.0/24 dev net12 proto kernel scope link src 192.168.112.2 
192.168.113.0/24 dev net13 proto kernel scope link src 192.168.113.2 
192.168.114.0/24 dev net14 proto kernel scope link src 192.168.114.2 
192.168.115.0/24 dev net15 proto kernel scope link src 192.168.115.2

root@mn-ipvlan-1:/workspace# for fabric in {0..15}; do
  subnet=$((100 + fabric))
  for pod in {1..6}; do
    ip="192.168.${subnet}.${pod}"
    ping -c 1 -W 1 "$ip" >/dev/null 2>&1 \
      && echo "OK   $ip" \
      || echo "FAIL $ip"
  done
done
OK   192.168.100.1
OK   192.168.100.2
OK   192.168.100.3
OK   192.168.100.4
OK   192.168.100.5
OK   192.168.100.6
OK   192.168.101.1
OK   192.168.101.2
OK   192.168.101.3
OK   192.168.101.4
OK   192.168.101.5
OK   192.168.101.6
OK   192.168.102.1
OK   192.168.102.2
OK   192.168.102.3
OK   192.168.102.4
OK   192.168.102.5
OK   192.168.102.6
OK   192.168.103.1
OK   192.168.103.2
OK   192.168.103.3
OK   192.168.103.4
OK   192.168.103.5
OK   192.168.103.6
OK   192.168.104.1
OK   192.168.104.2
OK   192.168.104.3
OK   192.168.104.4
OK   192.168.104.5
OK   192.168.104.6
OK   192.168.105.1
OK   192.168.105.2
OK   192.168.105.3
OK   192.168.105.4
OK   192.168.105.5
OK   192.168.105.6
OK   192.168.106.1
OK   192.168.106.2
OK   192.168.106.3
OK   192.168.106.4
OK   192.168.106.5
OK   192.168.106.6
OK   192.168.107.1
OK   192.168.107.2
OK   192.168.107.3
OK   192.168.107.4
OK   192.168.107.5
OK   192.168.107.6
OK   192.168.108.1
OK   192.168.108.2
OK   192.168.108.3
OK   192.168.108.4
OK   192.168.108.5
OK   192.168.108.6
OK   192.168.109.1
OK   192.168.109.2
OK   192.168.109.3
OK   192.168.109.4
OK   192.168.109.5
OK   192.168.109.6
OK   192.168.110.1
OK   192.168.110.2
OK   192.168.110.3
OK   192.168.110.4
OK   192.168.110.5
OK   192.168.110.6
OK   192.168.111.1
OK   192.168.111.2
OK   192.168.111.3
OK   192.168.111.4
OK   192.168.111.5
OK   192.168.111.6
OK   192.168.112.1
OK   192.168.112.2
OK   192.168.112.3
OK   192.168.112.4
OK   192.168.112.5
OK   192.168.112.6
OK   192.168.113.1
OK   192.168.113.2
OK   192.168.113.3
OK   192.168.113.4
OK   192.168.113.5
OK   192.168.113.6
OK   192.168.114.1
OK   192.168.114.2
OK   192.168.114.3
OK   192.168.114.4
OK   192.168.114.5
OK   192.168.114.6
OK   192.168.115.1
OK   192.168.115.2
OK   192.168.115.3
OK   192.168.115.4
OK   192.168.115.5
OK   192.168.115.6

root@mn-ipvlan-1:/workspace# arp | sort -k4
Address                  HWtype  HWaddress           Flags Mask            Iface
10.151.1.62              ether   22:cb:01:cc:ce:a3   C                     eth0
192.168.100.1            ether   ae:3a:e2:d2:0e:b2   C                     net0
192.168.100.3            ether   ae:3a:e2:cf:df:06   C                     net0
192.168.100.4            ether   ae:3a:e2:d2:0e:b2   C                     net0
192.168.100.5            ether   ae:3a:e2:cf:df:06   C                     net0
192.168.100.6            ether   ae:3a:e2:bc:91:32   C                     net0
192.168.101.1            ether   ae:3a:e2:d2:0e:b3   C                     net1
192.168.101.3            ether   ae:3a:e2:cf:df:07   C                     net1
192.168.101.4            ether   ae:3a:e2:d2:0e:b3   C                     net1
192.168.101.5            ether   ae:3a:e2:cf:df:07   C                     net1
192.168.101.6            ether   ae:3a:e2:bc:91:33   C                     net1
192.168.110.1            ether   ae:3a:e2:d2:0f:02   C                     net10
192.168.110.3            ether   ae:3a:e2:cf:df:56   C                     net10
192.168.110.4            ether   ae:3a:e2:d2:0f:02   C                     net10
192.168.110.5            ether   ae:3a:e2:cf:df:56   C                     net10
192.168.110.6            ether   ae:3a:e2:bc:91:82   C                     net10
192.168.111.1            ether   ae:3a:e2:d2:0f:03   C                     net11
192.168.111.3            ether   ae:3a:e2:cf:df:57   C                     net11
192.168.111.4            ether   ae:3a:e2:d2:0f:03   C                     net11
192.168.111.5            ether   ae:3a:e2:cf:df:57   C                     net11
192.168.111.6            ether   ae:3a:e2:bc:91:83   C                     net11
192.168.112.1            ether   ae:3a:e2:d2:0e:f2   C                     net12
192.168.112.3            ether   ae:3a:e2:cf:df:46   C                     net12
192.168.112.4            ether   ae:3a:e2:d2:0e:f2   C                     net12
192.168.112.5            ether   ae:3a:e2:cf:df:46   C                     net12
192.168.112.6            ether   ae:3a:e2:bc:91:72   C                     net12
192.168.113.1            ether   ae:3a:e2:d2:0e:f3   C                     net13
192.168.113.3            ether   ae:3a:e2:cf:df:47   C                     net13
192.168.113.4            ether   ae:3a:e2:d2:0e:f3   C                     net13
192.168.113.5            ether   ae:3a:e2:cf:df:47   C                     net13
192.168.113.6            ether   ae:3a:e2:bc:91:73   C                     net13
192.168.114.1            ether   ae:3a:e2:d2:0f:12   C                     net14
192.168.114.3            ether   ae:3a:e2:cf:df:66   C                     net14
192.168.114.4            ether   ae:3a:e2:d2:0f:12   C                     net14
192.168.114.5            ether   ae:3a:e2:cf:df:66   C                     net14
192.168.114.6            ether   ae:3a:e2:bc:91:92   C                     net14
192.168.115.1            ether   ae:3a:e2:d2:0f:13   C                     net15
192.168.115.3            ether   ae:3a:e2:cf:df:67   C                     net15
192.168.115.4            ether   ae:3a:e2:d2:0f:13   C                     net15
192.168.115.5            ether   ae:3a:e2:cf:df:67   C                     net15
192.168.115.6            ether   ae:3a:e2:bc:91:93   C                     net15
192.168.102.1            ether   ae:3a:e2:d2:0e:d2   C                     net2
192.168.102.3            ether   ae:3a:e2:cf:df:26   C                     net2
192.168.102.4            ether   ae:3a:e2:d2:0e:d2   C                     net2
192.168.102.5            ether   ae:3a:e2:cf:df:26   C                     net2
192.168.102.6            ether   ae:3a:e2:bc:91:52   C                     net2
192.168.103.1            ether   ae:3a:e2:d2:0e:d3   C                     net3
192.168.103.3            ether   ae:3a:e2:cf:df:27   C                     net3
192.168.103.4            ether   ae:3a:e2:d2:0e:d3   C                     net3
192.168.103.5            ether   ae:3a:e2:cf:df:27   C                     net3
192.168.103.6            ether   ae:3a:e2:bc:91:53   C                     net3
192.168.104.1            ether   ae:3a:e2:d2:0e:e2   C                     net4
192.168.104.3            ether   ae:3a:e2:cf:df:36   C                     net4
192.168.104.4            ether   ae:3a:e2:d2:0e:e2   C                     net4
192.168.104.5            ether   ae:3a:e2:cf:df:36   C                     net4
192.168.104.6            ether   ae:3a:e2:bc:91:62   C                     net4
192.168.105.1            ether   ae:3a:e2:d2:0e:e3   C                     net5
192.168.105.3            ether   ae:3a:e2:cf:df:37   C                     net5
192.168.105.4            ether   ae:3a:e2:d2:0e:e3   C                     net5
192.168.105.5            ether   ae:3a:e2:cf:df:37   C                     net5
192.168.105.6            ether   ae:3a:e2:bc:91:63   C                     net5
192.168.106.1            ether   ae:3a:e2:d2:0e:c2   C                     net6
192.168.106.3            ether   ae:3a:e2:cf:df:16   C                     net6
192.168.106.4            ether   ae:3a:e2:d2:0e:c2   C                     net6
192.168.106.5            ether   ae:3a:e2:cf:df:16   C                     net6
192.168.106.6            ether   ae:3a:e2:bc:91:42   C                     net6
192.168.107.1            ether   ae:3a:e2:d2:0e:c3   C                     net7
192.168.107.3            ether   ae:3a:e2:cf:df:17   C                     net7
192.168.107.4            ether   ae:3a:e2:d2:0e:c3   C                     net7
192.168.107.5            ether   ae:3a:e2:cf:df:17   C                     net7
192.168.107.6            ether   ae:3a:e2:bc:91:43   C                     net7
192.168.108.1            ether   ae:3a:e2:d2:0f:22   C                     net8
192.168.108.3            ether   ae:3a:e2:cf:df:76   C                     net8
192.168.108.4            ether   ae:3a:e2:d2:0f:22   C                     net8
192.168.108.5            ether   ae:3a:e2:cf:df:76   C                     net8
192.168.108.6            ether   ae:3a:e2:bc:91:a2   C                     net8
192.168.109.1            ether   ae:3a:e2:d2:0f:23   C                     net9
192.168.109.3            ether   ae:3a:e2:cf:df:77   C                     net9
192.168.109.4            ether   ae:3a:e2:d2:0f:23   C                     net9
192.168.109.5            ether   ae:3a:e2:cf:df:77   C                     net9
192.168.109.6            ether   ae:3a:e2:bc:91:a3   C                     net9
```

</details>
</br>


### RDMA Access Validation

** Pod: mn-ipvlan-1 to Pod: mn-ipvlan-0 **
 
``` bash
# mn-ipvlan-0
# kubectl exec -it mn-ipvlan-0 -- /bin/bash
root@mn-ipvlan-0:/workspace# 
    NCCL_DEBUG=INFO \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_IB_QPS_PER_CONNECTION=8 \
    NCCL_P2P_CHUNKSIZE=16777216 \
    MASTER_ADDR=10.151.0.245 \
    MASTER_PORT=12356 \
    WORLD_SIZE=2 \
    RANK=0 \
    NCCL_IB_HCA=mlx5_0 \
    GPU_ID=0 \
    python3 p2p_bw_test.py

# mn-ipvlan-1
# kubectl exec -it mn-ipvlan-1 -- /bin/bash
root@mn-ipvlan-1:/workspace#
    NCCL_DEBUG=INFO \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_IB_QPS_PER_CONNECTION=8 \
    NCCL_P2P_CHUNKSIZE=16777216 \
    MASTER_ADDR=10.151.0.245 \
    MASTER_PORT=12356 \
    WORLD_SIZE=2 \
    RANK=1 \
    NCCL_IB_HCA=mlx5_0 \
    GPU_ID=0 \
    python3 p2p_bw_test.py
```

## NCCL Tests

### Environment

``` bash
# kubectl get nodes -o wide
NAME                       STATUS   ROLES    AGE   VERSION   INTERNAL-IP   EXTERNAL-IP       OS-IMAGE                       KERNEL-VERSION                  CONTAINER-RUNTIME
rs-mn-b300x8-pool-37orfo   Ready    <none>   14m   v1.36.0   10.132.0.19   165.245.192.129   Debian GNU/Linux 13 (trixie)   6.12.90+deb13.1-amd64 (amd64)   containerd://2.2.3
rs-mn-b300x8-pool-37x4e3   Ready    <none>   8h    v1.36.0   10.132.0.5    165.245.193.214   Debian GNU/Linux 13 (trixie)   6.12.90+deb13.1-amd64 (amd64)   containerd://2.2.3
rs-mn-b300x8-pool-37x4e8   Ready    <none>   8h    v1.36.0   10.132.0.4    165.245.193.217   Debian GNU/Linux 13 (trixie)   6.12.90+deb13.1-amd64 (amd64)   containerd://2.2.3
rs-mn-b300x8-pool-37x4en   Ready    <none>   8h    v1.36.0   10.132.0.6    165.245.193.212   Debian GNU/Linux 13 (trixie)   6.12.90+deb13.1-amd64 (amd64)   containerd://2.2.3
rs-mn-cpu-pool-37xu9d      Ready    <none>   19h   v1.36.0   10.132.0.13   165.245.199.252   Debian GNU/Linux 13 (trixie)   6.12.90+deb13.1-amd64 (amd64)   containerd://2.2.3
```

###  MPI Operator Installation

``` bash
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
mpi-operator   mpi-operator-65474ddc85-lx8kz              1/1     Running   0             67s
```

### AllReduce Tests

<details>
<summary>1 B300x8 DOKS Worker</summary>

``` bash
# kubectl apply -f nvidia-mn-doks/nccl-test-1-node-b300x8.yaml
mpijob.kubeflow.org/mpi-multus-b300x8 created

# kubectl delete mpijob mpi-multus-b300x8

# kubectl get pods -o wide
NAME                               READY   STATUS    RESTARTS   AGE     IP             NODE                       NOMINATED NODE   READINESS GATES
mpi-multus-b300x8-launcher-b8v4h   1/1     Running   0          3m45s   10.133.0.9     rs-mn-cpu-pool-37xu9d      <none>           <none>
mpi-multus-b300x8-worker-0         1/1     Running   0          3m45s   10.133.0.226   rs-mn-b300x8-pool-37x4e3   <none>           <none>

# kubectl get mpijob
NAME                AGE
mpi-multus-b300x8   4m25s

# kubectl get pods -o wide
NAME                               READY   STATUS      RESTARTS   AGE    IP           NODE                    NOMINATED NODE   READINESS GATES
mpi-multus-b300x8-launcher-zbgmc   0/1     Completed   0          6m7s   10.133.0.9   rs-mn-cpu-pool-37xu9d   <none>           <none>

# kubectl describe mpijob mpi-multus-b300x8
# kubectl logs mpi-multus-b300x8-worker-0 -c mpi-worker
# kubectl exec -it mpi-multus-b300x8-worker-0 -- /bin/bash

# kubectl logs -f mpi-multus-b300x8-launcher-b8v4h
Warning: Permanently added '[mpi-multus-b300x8-worker-0.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 100 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    305 on mpi-multus-b300x8-worker-0 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  1 Group  0 Pid    306 on mpi-multus-b300x8-worker-0 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank  2 Group  0 Pid    307 on mpi-multus-b300x8-worker-0 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank  3 Group  0 Pid    308 on mpi-multus-b300x8-worker-0 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank  4 Group  0 Pid    312 on mpi-multus-b300x8-worker-0 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank  5 Group  0 Pid    314 on mpi-multus-b300x8-worker-0 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank  6 Group  0 Pid    316 on mpi-multus-b300x8-worker-0 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank  7 Group  0 Pid    321 on mpi-multus-b300x8-worker-0 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
NCCL version 2.29.7+cuda13.2
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    26.80    0.00    0.00      0    26.98    0.00    0.00      0
          16             4     float     sum      -1    25.08    0.00    0.00      0    25.19    0.00    0.00      0
          32             8     float     sum      -1    27.06    0.00    0.00      0    25.25    0.00    0.00      0
          64            16     float     sum      -1    28.87    0.00    0.00      0    31.25    0.00    0.00      0
         128            32     float     sum      -1    33.37    0.00    0.01      0    30.92    0.00    0.01      0
         256            64     float     sum      -1    31.23    0.01    0.01      0    30.98    0.01    0.01      0
         512           128     float     sum      -1    31.42    0.02    0.03      0    31.29    0.02    0.03      0
        1024           256     float     sum      -1    31.64    0.03    0.06      0    31.63    0.03    0.06      0
        2048           512     float     sum      -1    32.58    0.06    0.11      0    31.80    0.06    0.11      0
        4096          1024     float     sum      -1    34.45    0.12    0.21      0    32.33    0.13    0.22      0
        8192          2048     float     sum      -1    32.79    0.25    0.44      0    33.03    0.25    0.43      0
       16384          4096     float     sum      -1    32.71    0.50    0.88      0    35.28    0.46    0.81      0
       32768          8192     float     sum      -1    34.75    0.94    1.65      0    33.31    0.98    1.72      0
       65536         16384     float     sum      -1    34.40    1.91    3.33      0    34.26    1.91    3.35      0
      131072         32768     float     sum      -1    34.78    3.77    6.60      0    34.75    3.77    6.60      0
      262144         65536     float     sum      -1    35.77    7.33   12.83      0    35.51    7.38   12.92      0
      524288        131072     float     sum      -1    36.65   14.30   25.03      0    36.53   14.35   25.12      0
     1048576        262144     float     sum      -1    36.69   28.58   50.01      0    37.08   28.28   49.48      0
     2097152        524288     float     sum      -1    43.00   48.77   85.35      0    42.68   49.13   85.98      0
     4194304       1048576     float     sum      -1    64.15   65.38  114.42      0    63.57   65.98  115.46      0
     8388608       2097152     float     sum      -1    82.29  101.94  178.40      0    82.92  101.16  177.03      0
    16777216       4194304     float     sum      -1    113.3  148.02  259.04      0    111.6  150.28  262.99      0
    33554432       8388608     float     sum      -1    176.1  190.51  333.40      0    174.4  192.39  336.68      0
    67108864      16777216     float     sum      -1    284.7  235.71  412.48      0    283.8  236.46  413.81      0
   134217728      33554432     float     sum      -1    401.6  334.20  584.85      0    400.3  335.25  586.70      0
   268435456      67108864     float     sum      -1    718.5  373.61  653.81      0    719.5  373.07  652.87      0
   536870912     134217728     float     sum      -1   1338.4  401.14  701.99      0   1337.1  401.51  702.65      0
  1073741824     268435456     float     sum      -1   2576.0  416.83  729.45      0   2573.7  417.19  730.09      0
  2147483648     536870912     float     sum      -1   4592.5  467.60  818.30      0   4594.8  467.37  817.90      0
  4294967296    1073741824     float     sum      -1   9064.7  473.81  829.18      0   9070.7  473.50  828.63      0
  8589934592    2147483648     float     sum      -1    17990  477.48  835.59      0    17980  477.75  836.05      0
 17179869184    4294967296     float     sum      -1    35756  480.47  840.83      0    35756  480.47  840.82      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 233.857 
#
# Collective test concluded: all_reduce_perf
```

</details>
</br>

<details>
<summary>2 B300x8 DOKS Workers</summary>

``` bash
# kubectl apply -f nvidia-mn-doks/nccl-test-2-nodes-b300x8.yaml
mpijob.kubeflow.org/mpi-multus-b300x8 created

# kubectl delete mpijob mpi-multus-b300x8

# kubectl get pods -o wide
NAME                               READY   STATUS    RESTARTS   AGE   IP             NODE                       NOMINATED NODE   READINESS GATES
mpi-multus-b300x8-launcher-w942n   1/1     Running   0          55s   10.133.0.105   rs-mn-cpu-pool-37xu9d      <none>           <none>
mpi-multus-b300x8-worker-0         1/1     Running   0          55s   10.133.0.211   rs-mn-b300x8-pool-37x4e3   <none>           <none>
mpi-multus-b300x8-worker-1         1/1     Running   0          55s   10.133.1.188   rs-mn-b300x8-pool-37x4e8   <none>           <none>

# kubectl logs -f mpi-multus-b300x8-launcher-w942n
Warning: Permanently added '[mpi-multus-b300x8-worker-1.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-0.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 100 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    163 on mpi-multus-b300x8-worker-0 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  1 Group  0 Pid    164 on mpi-multus-b300x8-worker-0 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank  2 Group  0 Pid    165 on mpi-multus-b300x8-worker-0 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank  3 Group  0 Pid    166 on mpi-multus-b300x8-worker-0 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank  4 Group  0 Pid    169 on mpi-multus-b300x8-worker-0 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank  5 Group  0 Pid    172 on mpi-multus-b300x8-worker-0 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank  6 Group  0 Pid    174 on mpi-multus-b300x8-worker-0 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank  7 Group  0 Pid    178 on mpi-multus-b300x8-worker-0 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank  8 Group  0 Pid    163 on mpi-multus-b300x8-worker-1 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  9 Group  0 Pid    164 on mpi-multus-b300x8-worker-1 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 10 Group  0 Pid    165 on mpi-multus-b300x8-worker-1 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 11 Group  0 Pid    166 on mpi-multus-b300x8-worker-1 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 12 Group  0 Pid    168 on mpi-multus-b300x8-worker-1 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 13 Group  0 Pid    172 on mpi-multus-b300x8-worker-1 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 14 Group  0 Pid    175 on mpi-multus-b300x8-worker-1 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 15 Group  0 Pid    179 on mpi-multus-b300x8-worker-1 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
NCCL version 2.29.7+cuda13.2
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    38.93    0.00    0.00      0    39.60    0.00    0.00      0
          16             4     float     sum      -1    38.75    0.00    0.00      0    38.26    0.00    0.00      0
          32             8     float     sum      -1    38.46    0.00    0.00      0    37.94    0.00    0.00      0
          64            16     float     sum      -1    39.04    0.00    0.00      0    53.69    0.00    0.00      0
         128            32     float     sum      -1    39.15    0.00    0.01      0    39.34    0.00    0.01      0
         256            64     float     sum      -1    39.98    0.01    0.01      0    40.56    0.01    0.01      0
         512           128     float     sum      -1    40.28    0.01    0.02      0    40.22    0.01    0.02      0
        1024           256     float     sum      -1    40.58    0.03    0.05      0    40.65    0.03    0.05      0
        2048           512     float     sum      -1    42.38    0.05    0.09      0    42.34    0.05    0.09      0
        4096          1024     float     sum      -1    44.71    0.09    0.17      0    44.57    0.09    0.17      0
        8192          2048     float     sum      -1    45.48    0.18    0.34      0    46.53    0.18    0.33      0
       16384          4096     float     sum      -1    46.12    0.36    0.67      0    46.11    0.36    0.67      0
       32768          8192     float     sum      -1    48.19    0.68    1.27      0    47.94    0.68    1.28      0
       65536         16384     float     sum      -1    49.91    1.31    2.46      0    49.87    1.31    2.46      0
      131072         32768     float     sum      -1    51.46    2.55    4.78      0    51.72    2.53    4.75      0
      262144         65536     float     sum      -1    69.07    3.80    7.12      0    69.07    3.80    7.12      0
      524288        131072     float     sum      -1    72.32    7.25   13.59      0    72.77    7.20   13.51      0
     1048576        262144     float     sum      -1    79.58   13.18   24.70      0    78.77   13.31   24.96      0
     2097152        524288     float     sum      -1    94.21   22.26   41.74      0    95.15   22.04   41.33      0
     4194304       1048576     float     sum      -1    126.2   33.25   62.34      0    126.1   33.27   62.38      0
     8388608       2097152     float     sum      -1    138.1   60.76  113.93      0    137.2   61.16  114.67      0
    16777216       4194304     float     sum      -1    240.3   69.81  130.90      0    240.8   69.67  130.64      0
    33554432       8388608     float     sum      -1    283.4  118.42  222.03      0    282.4  118.81  222.77      0
    67108864      16777216     float     sum      -1    860.3   78.01  146.27      0    344.4  194.84  365.33      0
   134217728      33554432     float     sum      -1    544.3  246.57  462.32      0    590.4  227.33  426.24      0
   268435456      67108864     float     sum      -1    858.8  312.57  586.07      0    833.5  322.07  603.88      0
   536870912     134217728     float     sum      -1   1454.1  369.22  692.29      0   1434.2  374.33  701.86      0
  1073741824     268435456     float     sum      -1   2536.1  423.38  793.83      0   2704.2  397.07  744.50      0
  2147483648     536870912     float     sum      -1   4727.4  454.27  851.75      0   4776.3  449.61  843.02      0
  4294967296    1073741824     float     sum      -1   9495.7  452.31  848.07      0   9201.1  466.79  875.23      0
  8589934592    2147483648     float     sum      -1    18436  465.93  873.61      0    18204  471.88  884.77      0
 17179869184    4294967296     float     sum      -1    36106  475.82  892.17      0    36235  474.12  888.97      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 214.588 
#
# Collective test concluded: all_reduce_perf
```

</details>
</br>

<details>
<summary>2 B300x8 DOKS Workers - 8 Rails </summary>


``` bash
# kubectl apply -f nvidia-mn-doks/nccl-test-2-nodes-b300x-8rails.yaml
# kubectl delete mpijob mpi-multus-b300x8

# kubectl logs -f mpi-multus-b300x8-launcher-w988l
Warning: Permanently added '[mpi-multus-b300x8-worker-0.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-1.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 100 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    166 on mpi-multus-b300x8-worker-0 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  1 Group  0 Pid    167 on mpi-multus-b300x8-worker-0 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank  2 Group  0 Pid    168 on mpi-multus-b300x8-worker-0 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank  3 Group  0 Pid    169 on mpi-multus-b300x8-worker-0 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank  4 Group  0 Pid    172 on mpi-multus-b300x8-worker-0 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank  5 Group  0 Pid    175 on mpi-multus-b300x8-worker-0 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank  6 Group  0 Pid    177 on mpi-multus-b300x8-worker-0 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank  7 Group  0 Pid    182 on mpi-multus-b300x8-worker-0 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank  8 Group  0 Pid    164 on mpi-multus-b300x8-worker-1 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  9 Group  0 Pid    165 on mpi-multus-b300x8-worker-1 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 10 Group  0 Pid    166 on mpi-multus-b300x8-worker-1 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 11 Group  0 Pid    167 on mpi-multus-b300x8-worker-1 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 12 Group  0 Pid    169 on mpi-multus-b300x8-worker-1 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 13 Group  0 Pid    173 on mpi-multus-b300x8-worker-1 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 14 Group  0 Pid    175 on mpi-multus-b300x8-worker-1 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 15 Group  0 Pid    180 on mpi-multus-b300x8-worker-1 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
NCCL version 2.29.7+cuda13.2
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    39.63    0.00    0.00      0    39.54    0.00    0.00      0
          16             4     float     sum      -1    39.29    0.00    0.00      0    40.56    0.00    0.00      0
          32             8     float     sum      -1    49.80    0.00    0.00      0    39.75    0.00    0.00      0
          64            16     float     sum      -1    40.70    0.00    0.00      0    39.01    0.00    0.00      0
         128            32     float     sum      -1    40.21    0.00    0.01      0    40.75    0.00    0.01      0
         256            64     float     sum      -1    41.60    0.01    0.01      0    40.63    0.01    0.01      0
         512           128     float     sum      -1    42.09    0.01    0.02      0    41.16    0.01    0.02      0
        1024           256     float     sum      -1    42.66    0.02    0.05      0    41.44    0.02    0.05      0
        2048           512     float     sum      -1    42.85    0.05    0.09      0    43.02    0.05    0.09      0
        4096          1024     float     sum      -1    45.17    0.09    0.17      0    44.82    0.09    0.17      0
        8192          2048     float     sum      -1    47.04    0.17    0.33      0    47.05    0.17    0.33      0
       16384          4096     float     sum      -1    47.67    0.34    0.64      0    47.14    0.35    0.65      0
       32768          8192     float     sum      -1    49.26    0.67    1.25      0    49.08    0.67    1.25      0
       65536         16384     float     sum      -1    50.34    1.30    2.44      0    49.78    1.32    2.47      0
      131072         32768     float     sum      -1    54.22    2.42    4.53      0    54.06    2.42    4.55      0
      262144         65536     float     sum      -1    70.23    3.73    7.00      0    68.67    3.82    7.16      0
      524288        131072     float     sum      -1    77.18    6.79   12.74      0    75.73    6.92   12.98      0
     1048576        262144     float     sum      -1    128.0    8.19   15.36      0    89.61   11.70   21.94      0
     2097152        524288     float     sum      -1    640.1    3.28    6.14      0    131.2   15.99   29.98      0
     4194304       1048576     float     sum      -1    128.5   32.64   61.19      0    126.9   33.06   61.99      0
     8388608       2097152     float     sum      -1    143.0   58.66  109.99      0    141.8   59.16  110.93      0
    16777216       4194304     float     sum      -1    183.7   91.32  171.23      0    179.8   93.33  175.00      0
    33554432       8388608     float     sum      -1    265.4  126.41  237.02      0    263.2  127.50  239.06      0
    67108864      16777216     float     sum      -1    560.8  119.67  224.38      0    397.6  168.79  316.48      0
   134217728      33554432     float     sum      -1    634.1  211.68  396.89      0    632.8  212.11  397.70      0
   268435456      67108864     float     sum      -1    999.6  268.53  503.50      0    999.9  268.46  503.37      0
   536870912     134217728     float     sum      -1   1678.9  319.78  599.59      0   1688.5  317.96  596.17      0
  1073741824     268435456     float     sum      -1   3064.9  350.33  656.87      0   3104.7  345.85  648.47      0
  2147483648     536870912     float     sum      -1   5926.9  362.33  679.36      0   5881.5  365.12  684.60      0
  4294967296    1073741824     float     sum      -1    11297  380.19  712.86      0    11274  380.95  714.29      0
  8589934592    2147483648     float     sum      -1    22435  382.89  717.92      0    22350  384.34  720.63      0
 17179869184    4294967296     float     sum      -1    44373  387.17  725.94      0    44509  385.99  723.72      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 184.712 
#
# Collective test concluded: all_reduce_perf
```

</details>
</br>


<details>
<summary>3 B300x8 DOKS Workers</summary>


``` bash
# kubectl apply -f nvidia-mn-doks/nccl-test-3-nodes-b300x8.yaml
mpijob.kubeflow.org/mpi-multus-b300x8 created

# kubectl delete mpijob mpi-multus-b300x8

# # kubectl get pods -o wide
NAME                               READY   STATUS    RESTARTS   AGE   IP             NODE                       NOMINATED NODE   READINESS GATES
mpi-multus-b300x8-launcher-grbll   1/1     Running   0          29s   10.133.0.47    rs-mn-cpu-pool-37xu9d      <none>           <none>
mpi-multus-b300x8-worker-0         1/1     Running   0          29s   10.133.0.252   rs-mn-b300x8-pool-37x4e3   <none>           <none>
mpi-multus-b300x8-worker-1         1/1     Running   0          29s   10.133.1.121   rs-mn-b300x8-pool-37x4en   <none>           <none>
mpi-multus-b300x8-worker-2         1/1     Running   0          29s   10.133.1.200   rs-mn-b300x8-pool-37x4e8   <none>           <none>

# kubectl logs -f mpi-multus-b300x8-launcher-grbll
Warning: Permanently added '[mpi-multus-b300x8-worker-2.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-1.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-0.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 100 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    163 on mpi-multus-b300x8-worker-0 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  1 Group  0 Pid    164 on mpi-multus-b300x8-worker-0 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank  2 Group  0 Pid    165 on mpi-multus-b300x8-worker-0 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank  3 Group  0 Pid    166 on mpi-multus-b300x8-worker-0 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank  4 Group  0 Pid    169 on mpi-multus-b300x8-worker-0 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank  5 Group  0 Pid    172 on mpi-multus-b300x8-worker-0 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank  6 Group  0 Pid    176 on mpi-multus-b300x8-worker-0 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank  7 Group  0 Pid    179 on mpi-multus-b300x8-worker-0 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank  8 Group  0 Pid    167 on mpi-multus-b300x8-worker-1 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  9 Group  0 Pid    168 on mpi-multus-b300x8-worker-1 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 10 Group  0 Pid    169 on mpi-multus-b300x8-worker-1 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 11 Group  0 Pid    170 on mpi-multus-b300x8-worker-1 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 12 Group  0 Pid    172 on mpi-multus-b300x8-worker-1 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 13 Group  0 Pid    176 on mpi-multus-b300x8-worker-1 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 14 Group  0 Pid    178 on mpi-multus-b300x8-worker-1 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 15 Group  0 Pid    181 on mpi-multus-b300x8-worker-1 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank 16 Group  0 Pid    166 on mpi-multus-b300x8-worker-2 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank 17 Group  0 Pid    167 on mpi-multus-b300x8-worker-2 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 18 Group  0 Pid    168 on mpi-multus-b300x8-worker-2 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 19 Group  0 Pid    169 on mpi-multus-b300x8-worker-2 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 20 Group  0 Pid    171 on mpi-multus-b300x8-worker-2 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 21 Group  0 Pid    176 on mpi-multus-b300x8-worker-2 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 22 Group  0 Pid    178 on mpi-multus-b300x8-worker-2 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 23 Group  0 Pid    181 on mpi-multus-b300x8-worker-2 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
NCCL version 2.29.7+cuda13.2
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    50.15    0.00    0.00      0    49.20    0.00    0.00      0
          16             4     float     sum      -1    48.41    0.00    0.00      0    48.75    0.00    0.00      0
          32             8     float     sum      -1    48.79    0.00    0.00      0    48.76    0.00    0.00      0
          64            16     float     sum      -1    48.98    0.00    0.00      0    49.80    0.00    0.00      0
         128            32     float     sum      -1    49.88    0.00    0.00      0    50.48    0.00    0.00      0
         256            64     float     sum      -1    61.85    0.00    0.01      0    50.28    0.01    0.01      0
         512           128     float     sum      -1    63.72    0.01    0.02      0    50.25    0.01    0.02      0
        1024           256     float     sum      -1    50.76    0.02    0.04      0    50.60    0.02    0.04      0
        2048           512     float     sum      -1    67.26    0.03    0.06      0    53.02    0.04    0.07      0
        4096          1024     float     sum      -1    57.76    0.07    0.14      0    57.41    0.07    0.14      0
        8192          2048     float     sum      -1    58.92    0.14    0.27      0    58.84    0.14    0.27      0
       16384          4096     float     sum      -1    60.65    0.27    0.52      0    59.63    0.27    0.53      0
       32768          8192     float     sum      -1    62.05    0.53    1.01      0    73.03    0.45    0.86      0
       65536         16384     float     sum      -1    63.95    1.02    1.96      0    63.31    1.04    1.98      0
      131072         32768     float     sum      -1    87.35    1.50    2.88      0    86.42    1.52    2.91      0
      262144         65536     float     sum      -1    90.20    2.91    5.57      0    89.88    2.92    5.59      0
      524288        131072     float     sum      -1    95.18    5.51   10.56      0    94.26    5.56   10.66      0
     1048576        262144     float     sum      -1    101.5   10.33   19.79      0    101.4   10.35   19.83      0
     2097152        524288     float     sum      -1    116.8   17.96   34.42      0    116.5   18.01   34.51      0
     4194304       1048576     float     sum      -1    160.7   26.11   50.04      0    157.2   26.68   51.13      0
     8388608       2097152     float     sum      -1    173.7   48.28   92.54      0    172.4   48.65   93.25      0
    16777216       4194304     float     sum      -1    341.4   49.14   94.19      0    338.8   49.51   94.90      0
    33554432       8388608     float     sum      -1    392.3   85.54  163.95      0    405.5   82.76  158.62      0
    67108864      16777216     float     sum      -1    454.9  147.53  282.77      0    431.4  155.57  298.17      0
   134217728      33554432     float     sum      -1    617.7  217.29  416.48      0    623.6  215.24  412.54      0
   268435456      67108864     float     sum      -1   1024.3  262.07  502.30      0   1019.5  263.30  504.66      0
   536870912     134217728     float     sum      -1   1642.0  326.97  626.69      0   1635.6  328.24  629.12      0
  1073741824     268435456     float     sum      -1   3149.4  340.94  653.47      0   3187.6  336.85  645.63      0
  2147483648     536870912     float     sum      -1   6295.6  341.11  653.79      0   6152.3  349.05  669.02      0
  4294967296    1073741824     float     sum      -1    12147  353.58  677.70      0    12222  351.42  673.55      0
  8589934592    2147483648     float     sum      -1    24182  355.21  680.83      0    24481  350.88  672.53      0
 17179869184    4294967296     float     sum      -1    48021  357.76  685.70      0    48109  357.10  684.45      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 176.917 
#
# Collective test concluded: all_reduce_perf
```

</details>
</br>

<details>
<summary>4 B300x8 DOKS Workers</summary>

``` bash
# kubectl apply -f nvidia-mn-doks/nccl-test-4-nodes-b300x8.yaml
mpijob.kubeflow.org/mpi-multus-b300x8 created

# kubectl delete mpijob mpi-multus-b300x8

## kubectl get pods -o wide
NAME                               READY   STATUS    RESTARTS   AGE   IP             NODE                       NOMINATED NODE   READINESS GATES
mpi-multus-b300x8-launcher-2hgxf    1/1     Running   0          18s   10.133.0.51    rs-mn-cpu-pool-37xu9d      <none>           <none>
mpi-multus-b300x8-worker-0         1/1     Running   0          18s   10.133.1.99    rs-mn-b300x8-pool-37x4en   <none>           <none>
mpi-multus-b300x8-worker-1         0/1     Running   0          18s   10.133.0.203   rs-mn-b300x8-pool-37x4e3   <none>           <none>
mpi-multus-b300x8-worker-2         0/1     Running   0          18s   10.133.2.121   rs-mn-b300x8-pool-37orfo   <none>           <none>
mpi-multus-b300x8-worker-3         0/1     Running   0          18s   10.133.1.216   rs-mn-b300x8-pool-37x4e8   <none>           <none>

# kubectl logs -f mpi-multus-b300x8-launcher-2hgxf 
Warning: Permanently added '[mpi-multus-b300x8-worker-1.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-3.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-2.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-0.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 100 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    165 on mpi-multus-b300x8-worker-0 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  1 Group  0 Pid    166 on mpi-multus-b300x8-worker-0 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank  2 Group  0 Pid    167 on mpi-multus-b300x8-worker-0 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank  3 Group  0 Pid    168 on mpi-multus-b300x8-worker-0 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank  4 Group  0 Pid    171 on mpi-multus-b300x8-worker-0 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank  5 Group  0 Pid    174 on mpi-multus-b300x8-worker-0 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank  6 Group  0 Pid    177 on mpi-multus-b300x8-worker-0 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank  7 Group  0 Pid    181 on mpi-multus-b300x8-worker-0 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank  8 Group  0 Pid    163 on mpi-multus-b300x8-worker-1 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  9 Group  0 Pid    164 on mpi-multus-b300x8-worker-1 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 10 Group  0 Pid    165 on mpi-multus-b300x8-worker-1 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 11 Group  0 Pid    166 on mpi-multus-b300x8-worker-1 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 12 Group  0 Pid    169 on mpi-multus-b300x8-worker-1 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 13 Group  0 Pid    172 on mpi-multus-b300x8-worker-1 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 14 Group  0 Pid    175 on mpi-multus-b300x8-worker-1 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 15 Group  0 Pid    178 on mpi-multus-b300x8-worker-1 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank 16 Group  0 Pid    164 on mpi-multus-b300x8-worker-2 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank 17 Group  0 Pid    165 on mpi-multus-b300x8-worker-2 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 18 Group  0 Pid    166 on mpi-multus-b300x8-worker-2 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 19 Group  0 Pid    167 on mpi-multus-b300x8-worker-2 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 20 Group  0 Pid    169 on mpi-multus-b300x8-worker-2 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 21 Group  0 Pid    173 on mpi-multus-b300x8-worker-2 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 22 Group  0 Pid    175 on mpi-multus-b300x8-worker-2 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 23 Group  0 Pid    178 on mpi-multus-b300x8-worker-2 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank 24 Group  0 Pid    116 on mpi-multus-b300x8-worker-3 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank 25 Group  0 Pid    117 on mpi-multus-b300x8-worker-3 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 26 Group  0 Pid    118 on mpi-multus-b300x8-worker-3 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 27 Group  0 Pid    119 on mpi-multus-b300x8-worker-3 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 28 Group  0 Pid    121 on mpi-multus-b300x8-worker-3 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 29 Group  0 Pid    124 on mpi-multus-b300x8-worker-3 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 30 Group  0 Pid    127 on mpi-multus-b300x8-worker-3 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 31 Group  0 Pid    132 on mpi-multus-b300x8-worker-3 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
NCCL version 2.29.7+cuda13.2
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    49.90    0.00    0.00      0    49.26    0.00    0.00      0
          16             4     float     sum      -1    49.79    0.00    0.00      0    48.71    0.00    0.00      0
          32             8     float     sum      -1    49.40    0.00    0.00      0    49.51    0.00    0.00      0
          64            16     float     sum      -1    49.69    0.00    0.00      0    49.54    0.00    0.00      0
         128            32     float     sum      -1    50.05    0.00    0.00      0    49.87    0.00    0.00      0
         256            64     float     sum      -1    50.43    0.01    0.01      0    50.84    0.01    0.01      0
         512           128     float     sum      -1    51.22    0.01    0.02      0    52.08    0.01    0.02      0
        1024           256     float     sum      -1    51.66    0.02    0.04      0    51.90    0.02    0.04      0
        2048           512     float     sum      -1    53.76    0.04    0.07      0    54.09    0.04    0.07      0
        4096          1024     float     sum      -1    59.00    0.07    0.13      0    58.43    0.07    0.14      0
        8192          2048     float     sum      -1    60.52    0.14    0.26      0    59.64    0.14    0.27      0
       16384          4096     float     sum      -1    61.69    0.27    0.51      0    61.14    0.27    0.52      0
       32768          8192     float     sum      -1    63.21    0.52    1.00      0    62.91    0.52    1.01      0
       65536         16384     float     sum      -1    66.95    0.98    1.90      0    66.77    0.98    1.90      0
      131072         32768     float     sum      -1    70.05    1.87    3.63      0    68.23    1.92    3.72      0
      262144         65536     float     sum      -1    93.18    2.81    5.45      0    92.42    2.84    5.50      0
      524288        131072     float     sum      -1    95.29    5.50   10.66      0    95.22    5.51   10.67      0
     1048576        262144     float     sum      -1    106.5    9.85   19.08      0    120.5    8.70   16.86      0
     2097152        524288     float     sum      -1    153.3   13.68   26.50      0    118.3   17.72   34.34      0
     4194304       1048576     float     sum      -1    163.8   25.60   49.60      0    166.8   25.14   48.71      0
     8388608       2097152     float     sum      -1    246.3   34.05   65.98      0    237.1   35.39   68.56      0
    16777216       4194304     float     sum      -1    444.4   37.75   73.14      0    439.9   38.14   73.90      0
    33554432       8388608     float     sum      -1    458.8   73.14  141.71      0    461.8   72.67  140.79      0
    67108864      16777216     float     sum      -1    540.1  124.26  240.76      0    539.6  124.36  240.94      0
   134217728      33554432     float     sum      -1    677.3  198.17  383.95      0    739.7  181.44  351.55      0
   268435456      67108864     float     sum      -1   1060.1  253.22  490.62      0   1063.9  252.31  488.85      0
   536870912     134217728     float     sum      -1   1904.9  281.83  546.05      0   1982.4  270.82  524.72      0
  1073741824     268435456     float     sum      -1   3213.4  334.15  647.41      0   3251.3  330.25  639.86      0
  2147483648     536870912     float     sum      -1   6197.9  346.49  671.32      0   6218.1  345.36  669.14      0
  4294967296    1073741824     float     sum      -1    12307  348.98  676.15      0    12265  350.18  678.47      0
  8589934592    2147483648     float     sum      -1    24324  353.14  684.21      0    24406  351.95  681.91      0
 17179869184    4294967296     float     sum      -1    48392  355.01  687.84      0    48354  355.29  688.38      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 168.732 
#
# Collective test concluded: all_reduce_perf
```

</details>
</br>

<details>
<summary>4 B300x8 DOKS Workers - 1 Rail </summary>

``` bash
# kubectl apply -f nvidia-mn-doks/nccl-test-4-nodes-b300x8-1rail.yaml
# kubectl delete mpijob mpi-multus-b300x8

# kubectl logs -f mpi-multus-b300x8-launcher-4fjjx
Warning: Permanently added '[mpi-multus-b300x8-worker-3.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-2.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-0.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-1.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 100 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    164 on mpi-multus-b300x8-worker-0 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  1 Group  0 Pid    165 on mpi-multus-b300x8-worker-0 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank  2 Group  0 Pid    166 on mpi-multus-b300x8-worker-0 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank  3 Group  0 Pid    167 on mpi-multus-b300x8-worker-0 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank  4 Group  0 Pid    171 on mpi-multus-b300x8-worker-0 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank  5 Group  0 Pid    173 on mpi-multus-b300x8-worker-0 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank  6 Group  0 Pid    177 on mpi-multus-b300x8-worker-0 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank  7 Group  0 Pid    179 on mpi-multus-b300x8-worker-0 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank  8 Group  0 Pid    164 on mpi-multus-b300x8-worker-1 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  9 Group  0 Pid    165 on mpi-multus-b300x8-worker-1 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 10 Group  0 Pid    166 on mpi-multus-b300x8-worker-1 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 11 Group  0 Pid    167 on mpi-multus-b300x8-worker-1 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 12 Group  0 Pid    170 on mpi-multus-b300x8-worker-1 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 13 Group  0 Pid    174 on mpi-multus-b300x8-worker-1 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 14 Group  0 Pid    177 on mpi-multus-b300x8-worker-1 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 15 Group  0 Pid    180 on mpi-multus-b300x8-worker-1 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank 16 Group  0 Pid    166 on mpi-multus-b300x8-worker-2 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank 17 Group  0 Pid    167 on mpi-multus-b300x8-worker-2 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 18 Group  0 Pid    168 on mpi-multus-b300x8-worker-2 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 19 Group  0 Pid    169 on mpi-multus-b300x8-worker-2 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 20 Group  0 Pid    171 on mpi-multus-b300x8-worker-2 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 21 Group  0 Pid    175 on mpi-multus-b300x8-worker-2 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 22 Group  0 Pid    178 on mpi-multus-b300x8-worker-2 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 23 Group  0 Pid    182 on mpi-multus-b300x8-worker-2 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank 24 Group  0 Pid    168 on mpi-multus-b300x8-worker-3 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank 25 Group  0 Pid    169 on mpi-multus-b300x8-worker-3 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 26 Group  0 Pid    170 on mpi-multus-b300x8-worker-3 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 27 Group  0 Pid    171 on mpi-multus-b300x8-worker-3 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 28 Group  0 Pid    174 on mpi-multus-b300x8-worker-3 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 29 Group  0 Pid    178 on mpi-multus-b300x8-worker-3 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 30 Group  0 Pid    181 on mpi-multus-b300x8-worker-3 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 31 Group  0 Pid    184 on mpi-multus-b300x8-worker-3 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
NCCL version 2.29.7+cuda13.2
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    51.52    0.00    0.00      0    66.95    0.00    0.00      0
          16             4     float     sum      -1    50.04    0.00    0.00      0    50.04    0.00    0.00      0
          32             8     float     sum      -1    49.38    0.00    0.00      0    49.89    0.00    0.00      0
          64            16     float     sum      -1    49.73    0.00    0.00      0    65.92    0.00    0.00      0
         128            32     float     sum      -1    54.42    0.00    0.00      0    61.17    0.00    0.00      0
         256            64     float     sum      -1    51.37    0.00    0.01      0    51.32    0.00    0.01      0
         512           128     float     sum      -1    51.64    0.01    0.02      0    51.56    0.01    0.02      0
        1024           256     float     sum      -1    52.84    0.02    0.04      0    51.92    0.02    0.04      0
        2048           512     float     sum      -1    53.76    0.04    0.07      0    54.84    0.04    0.07      0
        4096          1024     float     sum      -1    58.95    0.07    0.13      0    59.21    0.07    0.13      0
        8192          2048     float     sum      -1    62.37    0.13    0.25      0    61.71    0.13    0.26      0
       16384          4096     float     sum      -1    62.82    0.26    0.51      0    62.13    0.26    0.51      0
       32768          8192     float     sum      -1    72.72    0.45    0.87      0    71.31    0.46    0.89      0
       65536         16384     float     sum      -1    90.11    0.73    1.41      0    88.37    0.74    1.44      0
      131072         32768     float     sum      -1    96.71    1.36    2.63      0    96.18    1.36    2.64      0
      262144         65536     float     sum      -1    109.9    2.39    4.62      0    109.6    2.39    4.63      0
      524288        131072     float     sum      -1    141.1    3.72    7.20      0    158.0    3.32    6.43      0
     1048576        262144     float     sum      -1    205.2    5.11    9.90      0    214.7    4.88    9.46      0
     2097152        524288     float     sum      -1    327.0    6.41   12.43      0    323.8    6.48   12.55      0
     4194304       1048576     float     sum      -1    334.8   12.53   24.27      0    334.1   12.55   24.32      0
     8388608       2097152     float     sum      -1    448.3   18.71   36.25      0    447.9   18.73   36.28      0
    16777216       4194304     float     sum      -1    728.8   23.02   44.60      0    728.7   23.02   44.61      0
    33554432       8388608     float     sum      -1   1429.5   23.47   45.48      0   1429.4   23.47   45.48      0
    67108864      16777216     float     sum      -1   2849.5   23.55   45.63      0   2849.7   23.55   45.63      0
   134217728      33554432     float     sum      -1   5339.1   25.14   48.71      0   5336.7   25.15   48.73      0
   268435456      67108864     float     sum      -1    10640   25.23   48.88      0    10632   25.25   48.92      0
   536870912     134217728     float     sum      -1    21296   25.21   48.84      0    21307   25.20   48.82      0
  1073741824     268435456     float     sum      -1    42589   25.21   48.85      0    42583   25.22   48.85      0
  2147483648     536870912     float     sum      -1    85214   25.20   48.83      0    85191   25.21   48.84      0
  4294967296    1073741824     float     sum      -1   170313   25.22   48.86      0   170397   25.21   48.84      0
  8589934592    2147483648     float     sum      -1   340593   25.22   48.86      0   340339   25.24   48.90      0
 17179869184    4294967296     float     sum      -1   680561   25.24   48.91      0   680288   25.25   48.93      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 19.5831 
#
# Collective test concluded: all_reduce_perf
```

</details>
</br>

<details>
<summary>4 B300x8 DOKS Workers - 2 Rails </summary>

``` bash
# kubectl apply -f nvidia-mn-doks/nccl-test-4-nodes-b300x8-2rails.yaml
# kubectl delete mpijob mpi-multus-b300x8

# kubectl logs -f mpi-multus-b300x8-launcher-rgnt6 
Warning: Permanently added '[mpi-multus-b300x8-worker-1.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-2.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-0.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
Warning: Permanently added '[mpi-multus-b300x8-worker-3.mpi-multus-b300x8.default.svc]:2222' (ED25519) to the list of known hosts.
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 100 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    164 on mpi-multus-b300x8-worker-0 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  1 Group  0 Pid    165 on mpi-multus-b300x8-worker-0 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank  2 Group  0 Pid    166 on mpi-multus-b300x8-worker-0 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank  3 Group  0 Pid    167 on mpi-multus-b300x8-worker-0 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank  4 Group  0 Pid    169 on mpi-multus-b300x8-worker-0 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank  5 Group  0 Pid    174 on mpi-multus-b300x8-worker-0 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank  6 Group  0 Pid    177 on mpi-multus-b300x8-worker-0 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank  7 Group  0 Pid    180 on mpi-multus-b300x8-worker-0 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank  8 Group  0 Pid    167 on mpi-multus-b300x8-worker-1 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank  9 Group  0 Pid    168 on mpi-multus-b300x8-worker-1 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 10 Group  0 Pid    169 on mpi-multus-b300x8-worker-1 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 11 Group  0 Pid    170 on mpi-multus-b300x8-worker-1 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 12 Group  0 Pid    173 on mpi-multus-b300x8-worker-1 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 13 Group  0 Pid    176 on mpi-multus-b300x8-worker-1 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 14 Group  0 Pid    178 on mpi-multus-b300x8-worker-1 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 15 Group  0 Pid    183 on mpi-multus-b300x8-worker-1 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank 16 Group  0 Pid    164 on mpi-multus-b300x8-worker-2 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank 17 Group  0 Pid    165 on mpi-multus-b300x8-worker-2 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 18 Group  0 Pid    166 on mpi-multus-b300x8-worker-2 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 19 Group  0 Pid    167 on mpi-multus-b300x8-worker-2 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 20 Group  0 Pid    169 on mpi-multus-b300x8-worker-2 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 21 Group  0 Pid    173 on mpi-multus-b300x8-worker-2 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 22 Group  0 Pid    175 on mpi-multus-b300x8-worker-2 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 23 Group  0 Pid    180 on mpi-multus-b300x8-worker-2 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
#  Rank 24 Group  0 Pid    163 on mpi-multus-b300x8-worker-3 device  0 [0000:85:00] NVIDIA B300 SXM6 PC
#  Rank 25 Group  0 Pid    164 on mpi-multus-b300x8-worker-3 device  1 [0000:8d:00] NVIDIA B300 SXM6 PC
#  Rank 26 Group  0 Pid    165 on mpi-multus-b300x8-worker-3 device  2 [0000:95:00] NVIDIA B300 SXM6 PC
#  Rank 27 Group  0 Pid    166 on mpi-multus-b300x8-worker-3 device  3 [0000:9d:00] NVIDIA B300 SXM6 PC
#  Rank 28 Group  0 Pid    168 on mpi-multus-b300x8-worker-3 device  4 [0000:a5:00] NVIDIA B300 SXM6 PC
#  Rank 29 Group  0 Pid    172 on mpi-multus-b300x8-worker-3 device  5 [0000:ad:00] NVIDIA B300 SXM6 PC
#  Rank 30 Group  0 Pid    175 on mpi-multus-b300x8-worker-3 device  6 [0000:b5:00] NVIDIA B300 SXM6 PC
#  Rank 31 Group  0 Pid    178 on mpi-multus-b300x8-worker-3 device  7 [0000:bd:00] NVIDIA B300 SXM6 PC
NCCL version 2.29.7+cuda13.2
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    50.81    0.00    0.00      0    50.40    0.00    0.00      0
          16             4     float     sum      -1    49.00    0.00    0.00      0    48.98    0.00    0.00      0
          32             8     float     sum      -1    49.09    0.00    0.00      0    49.64    0.00    0.00      0
          64            16     float     sum      -1    49.77    0.00    0.00      0    50.12    0.00    0.00      0
         128            32     float     sum      -1    50.27    0.00    0.00      0    50.49    0.00    0.00      0
         256            64     float     sum      -1    50.98    0.01    0.01      0    51.00    0.01    0.01      0
         512           128     float     sum      -1    51.25    0.01    0.02      0    52.02    0.01    0.02      0
        1024           256     float     sum      -1    52.40    0.02    0.04      0    52.47    0.02    0.04      0
        2048           512     float     sum      -1    54.80    0.04    0.07      0    55.23    0.04    0.07      0
        4096          1024     float     sum      -1    59.80    0.07    0.13      0    58.81    0.07    0.13      0
        8192          2048     float     sum      -1    60.22    0.14    0.26      0    59.54    0.14    0.27      0
       16384          4096     float     sum      -1    61.83    0.26    0.51      0    60.88    0.27    0.52      0
       32768          8192     float     sum      -1    65.56    0.50    0.97      0    64.71    0.51    0.98      0
       65536         16384     float     sum      -1    67.23    0.97    1.89      0    67.21    0.98    1.89      0
      131072         32768     float     sum      -1    77.51    1.69    3.28      0    75.47    1.74    3.36      0
      262144         65536     float     sum      -1    92.43    2.84    5.50      0    91.62    2.86    5.54      0
      524288        131072     float     sum      -1    101.3    5.18   10.03      0    100.1    5.24   10.15      0
     1048576        262144     float     sum      -1    115.5    9.08   17.59      0    115.3    9.10   17.63      0
     2097152        524288     float     sum      -1    149.7   14.01   27.15      0    149.6   14.02   27.16      0
     4194304       1048576     float     sum      -1    223.8   18.74   36.31      0    223.1   18.80   36.43      0
     8388608       2097152     float     sum      -1    563.3   14.89   28.85      0    414.9   20.22   39.17      0
    16777216       4194304     float     sum      -1    459.8   36.49   70.70      0    430.8   38.94   75.45      0
    33554432       8388608     float     sum      -1    499.1   67.23  130.26      0    474.6   70.69  136.97      0
    67108864      16777216     float     sum      -1    714.1   93.98  182.09      0    662.4  101.32  196.30      0
   134217728      33554432     float     sum      -1   1018.6  131.76  255.29      0    977.7  137.28  265.98      0
   268435456      67108864     float     sum      -1   1604.8  167.27  324.08      0   1619.0  165.80  321.24      0
   536870912     134217728     float     sum      -1   2790.2  192.41  372.80      0   2791.6  192.32  372.62      0
  1073741824     268435456     float     sum      -1   5389.3  199.23  386.02      0   5387.4  199.31  386.16      0
  2147483648     536870912     float     sum      -1    10725  200.24  387.96      0    10722  200.30  388.07      0
  4294967296    1073741824     float     sum      -1    21521  199.57  386.67      0    21454  200.20  387.88      0
  8589934592    2147483648     float     sum      -1    42825  200.58  388.63      0    42781  200.79  389.02      0
 17179869184    4294967296     float     sum      -1    85418  201.13  389.68      0    85533  200.86  389.16      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 107.173 
#
# Collective test concluded: all_reduce_perf
```

</details>
</br>

## Documentation and Resources