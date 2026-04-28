# Multi-Node Environment Setup for MI350x8 GPU Droplets

This guide provides a basic introduction and step-by-step instructions for manually setting up a multi-node environment for testing and applications using MI350x8 GPU Droplets.

## GPU Network Introduction

All droplets have Internet access via `eth0` at 10 Gb/s and VPC connectivity via `eth1` at 25 Gb/s, while the GPU network supports inter-node GPU-to-GPU communication via [RDMA/RoCE](https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/how-to/roce-network-config.html) for distributed training and inference workloads.

A single GPU Fabric can support up to 122 homogeneous nodes (MI325x8 or MI350x8) after reserving 6 nodes as spares. These nodes can be assigned across multiple customers, with VLANs used to isolate and segregate their traffic.

The GPU Fabric is rail-only at the moment and consists of 8 independent forwarding planes. Each plane connects at 400 Gb/s to [an NVIDIA Mellanox ConnectX-7 NIC](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/network/nic-installation.html#nvidia-mellanox-cx-7-400gx1) on MI325x8 node or [an AMD Pensando Pollara NIC](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/network/nic-installation.html#amd-pensando-pollara-400-ai-nic) on MI350X nodes. NICs across different planes are isolated and cannot communicate with each other.

<img src="mi350x8-mn/mi350x8-mn-droplet.png" alt="GPU Network" width="800" height="400"/>

The 8 RDMA/RoCE NICs in GPU Droplets are configured and managed as `eth2–eth9`. We must define 8 IP address segments—one per forwarding plane—and assign a unique IP from the correct segment to each NIC, as misconfiguration can easily cause issues. 

The `eth2–eth9` can be treated as standard Ethernet interfaces, supported by the kernel TCP/IP stack in GPU Droplets (Ubuntu 24.04). You can run HTTP or ping over their IP addresses, and traffic can be managed via iptables.

When RDMA/RoCE applications run over the 8 NICs using RoCEv2 (UDP over IP), traffic is handled directly by the NICs, bypassing the kernel TCP/IP stack. As a result, the RX and TX packet counters for `eth2–eth9` do not increment, even though the traffic is transmitted successfully.

RDMA/RoCE applications still require an out-of-band TCP/IP channel for bootstrap/initialization and coordination, such as exchanging rank or connection information. On GPU Droplets, this channel is provided by `eth1` interface.

For GPU-to-GPU communication, the preferred paths are [XGMI](https://instinct.docs.amd.com/projects/virt-drv/en/latest/userguides/XGMI_configuration.html) (AMD Infinity Fabric) for intra-node transfers and 400 Gigabit Ethernet with RDMA for inter-node transfers. PCIe and shared memory (intra-node) and TCP/IP (inter-node, eth1) serve only as fallback options. Numerous [RCCL environment variables](https://rocm.docs.amd.com/projects/rccl/en/develop/api-reference/env-variables.html) can be configured to control these communication paths, and they should be carefully considered.

**The following summarizes the different communication paths, their roles and traffic in distributed AI/ML applications on top of GPU Droplets:**

| Interface / Path | Bandwidth | Role |
|------------------|----------|------|
| eth0 | 10 Gb/s | SSH; <br>HTTP APIs;<br> Downloads and uploads (container images, models, datasets, libraries, and updates) |
| eth1 | 25 Gb/s | NFS (models/checkpoints, datasets, logs, and metrics); <br>HTTP APIs for distributed inference across Router, Prefiller, and Decoder;<br> Bootstrap/initialization and coordination for RDMA/RoCE; <br>Fallback path for RDMA/RoCE |
| eth2–eth9 | 400 Gb/s each | RDMA/RoCE for inter-node GPU-to-GPU communication (used by RCCL);<br> Training traffic (gradients, parameters, optimizer states, activations);<br> Inference traffic (activations, KV cache) |
| XGMI (not user-configurable) | 7×128 GB/s per MI325 GPU<br> 7×153.6 GB/s per MI350 GPU   | Intra-node GPU-to-GPU communication (used by NCCL/RCCL);<br>  Training traffic (gradients, activations, optimizer states);<br> Inference traffic (activations, KV cache) |

## Environment Setup and Basic Verification

Ensure that you have at least 2 nodes in the same GPU Fabric assigned to your account.

Provision 2 GPU droplets using doctl:

``` shell
doctl compute size list | grep gpu 
doctl compute ssh-key list
doctl vpcs list

# Get the current default AI/ML image for AMD GPU Droplets 
doctl compute image list --public | grep AMD | grep base


doctl compute droplet create rs-mi350x8-mn-test1 --image 221919586 --region sfo2 --size gpu-mi350x8-2304gb-fabric-contracted --enable-monitoring --ssh-keys 55531023

doctl compute droplet create rs-mi350x8-mn-test2 --image 221919586 --region sfo2 --size gpu-mi350x8-2304gb-fabric-contracted --enable-monitoring --ssh-keys 55531023
```

SSH into the droplets and perform a basic check:

``` shell
uname -r
cat /etc/os-release
dkms status

# Check if the AMD GPU kernel driver is currently loaded and active
lsmod | grep amdgpu 

amd-smi version
amd-smi list 
amd-smi xgmi
amd-smi topology

rocm-smi
rocm-smi --showtopo
rocminfo

# DO Agent
systemctl status do-agent 

# AMD Metrics Exporter
amd-metrics-exporter --version
systemctl status amd-metrics-exporter
netstat -tunap | amd-metrics-exporter
curl http://127.0.0.1:5000/metrics

# Basic Tools
apt update
apt install net-tools
```

Check the network configuration for the GPU droplets:

| GPU Droplet    | rs-mi350x8-mn-test1  | rs-mi350x8-mn-test2  | 
|----------------|----------------|-----------------|
| eth0-internet  | 143.110.201.31 | 178.128.11.166 |
| eth1-vpc       | 10.120.0.9/20  | 10.120.0.10/20 | 


## GPU Network Setup and Basic Check

Refer to GPU network IP allocation files ([node1-eth-ip.txt](mi350x8-mn/node1-eth-ip.txt) and [node2-eth-ip.txt](mi350x8-mn/node2-eth-ip.txt)), and configure the GPU network interface controllers on each GPU droplet:

| GPU Droplet    | rs-mi350x8-mn-test1  | rs-mi350x8-mn-test2  | 
|----------------|----------------|-----------------|
| eth2  | 192.168.52.101/24 | 192.168.52.102/24  | 
| eth3  | 192.168.53.101/24 | 192.168.53.102/24  | 
| eth4  | 192.168.54.101/24 | 192.168.54.102/24  | 
| eth5  | 192.168.55.101/24 | 192.168.55.102/24  |
| eth6  | 192.168.56.101/24 | 192.168.56.102/24  | 
| eth7  | 192.168.57.101/24 | 192.168.57.102/24  | 
| eth8  | 192.168.58.101/24 | 192.168.58.102/24  | 
| eth9  | 192.168.59.101/24 | 192.168.59.102/24  | 

``` shell
nano /etc/netplan/50-cloud-init.yaml 

# Copy the contents of the IP allocation file
netplan apply 
```

Perform a basic check:

``` shell
# Provides an overview of all network interfaces and their IPs
ip -br a

# Shows detailed configuration and status information for each interface.
ifconfig

# Lists all RDMA devices exposed by the kernel
ls /sys/class/infiniband/

# Provides detailed hardware-level information about each RDMA device’s capabilities and configuration.
ibv_devices
ibv_devinfo -v | grep GID

# Displays the mapping between RDMA devices and network interfaces along with link status
rdma link

# Ping the IPs of the remote eth0/eth1/eth2~9 interfaces.
ping xxx 
```

## Single Node Check - Details

```
root@rs-mi350x8-mn-test1:~# uname -r
6.8.0-106-generic

root@rs-mi350x8-mn-test1:~# cat /etc/os-release
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo

root@rs-mi350x8-mn-test1:~# dkms status
amdgpu/6.14.14-2226257.24.04, 6.8.0-106-generic, x86_64: installed
amdgpu/6.14.14-2226257.24.04, 6.8.0-110-generic, x86_64: installed
ionic/25.08.4.004, 6.8.0-106-generic, x86_64: installed
ionic/25.08.4.004, 6.8.0-110-generic, x86_64: installed
mlnx-ofed-kernel/24.10.OFED.24.10.3.2.5.1, 6.8.0-106-generic, x86_64: installed
mlnx-ofed-kernel/24.10.OFED.24.10.3.2.5.1, 6.8.0-110-generic, x86_64: installed

root@rs-mi350x8-mn-test1:~# amd-smi version
AMDSMI Tool: 26.0.2+39589fda | AMDSMI Library version: 26.0.2 | ROCm version: 7.0.2 | amdgpu version: 6.14.14 | amd_hsmp version: N/A

root@rs-mi350x8-mn-test1:~# amd-metrics-exporter --version
Version : v1.4.0-31
BuildDate: 2025-10-03T05:40:37-0700
GitCommit: fe12ca42
Deployment: debian package deployment

root@rs-mi350x8-mn-test1:~# amd-smi firmware 
GPU: 0
    FW_LIST:
        FW 0:
            FW_ID: CP_MEC1
            FW_VERSION: 34
        FW 1:
            FW_ID: CP_MEC2
            FW_VERSION: 34
        FW 2:
            FW_ID: RLC
            FW_VERSION: 42
        FW 3:
            FW_ID: SDMA0
            FW_VERSION: 12
        FW 4:
            FW_ID: SDMA1
            FW_VERSION: 12
        FW 5:
            FW_ID: VCN
            FW_VERSION: 09.10.90.11
        FW 6:
            FW_ID: RLC_RESTORE_LIST_GPM_MEM
            FW_VERSION: 4
        FW 7:
            FW_ID: RLC_RESTORE_LIST_SRM_MEM
            FW_VERSION: 4
        FW 8:
            FW_ID: RLC_RESTORE_LIST_CNTL
            FW_VERSION: 4
        FW 9:
            FW_ID: TA_RAS
            FW_VERSION: 1B.45.00.09
        FW 10:
            FW_ID: TA_XGMI
            FW_VERSION: 20.00.00.14
        FW 11:
            FW_ID: PM
            FW_VERSION: 04.86.15.106
        FW 12:
            FW_ID: PLDM_BUNDLE
            FW_VERSION: 01.25.17.07
GPU: 1

root@rs-mi350x8-mn-test1:~# amd-smi
+------------------------------------------------------------------------------+
| AMD-SMI 26.0.2+39589fda      amdgpu version: 6.14.14  ROCm version: 7.0.2    |
| Platform: Linux Guest                                                        |
|-------------------------------------+----------------------------------------|
| BDF                        GPU-Name | Mem-Uti   Temp   UEC       Power-Usage |
| GPU  HIP-ID  OAM-ID  Partition-Mode | GFX-Uti    Fan               Mem-Usage |
|=====================================+========================================|
| 0000:83:00.0 AMD Instinct MI350X VF | 0 %      60 °C   0          268/1000 W |
|   0       0       6        SPX/NPS1 | 11 %       N/A           285/294592 MB |
|-------------------------------------+----------------------------------------|
| 0000:8b:00.0 AMD Instinct MI350X VF | 0 %      63 °C   0          265/1000 W |
|   1       1       7        SPX/NPS1 | 9 %        N/A           285/294592 MB |
|-------------------------------------+----------------------------------------|
| 0000:93:00.0 AMD Instinct MI350X VF | 0 %      60 °C   0          268/1000 W |
|   2       2       5        SPX/NPS1 | 9 %        N/A           285/294592 MB |
|-------------------------------------+----------------------------------------|
| 0000:9b:00.0 AMD Instinct MI350X VF | 0 %      65 °C   0          267/1000 W |
|   3       3       4        SPX/NPS1 | 11 %       N/A           285/294592 MB |
|-------------------------------------+----------------------------------------|
| 0000:a3:00.0 AMD Instinct MI350X VF | 0 %      61 °C   0          266/1000 W |
|   4       4       2        SPX/NPS1 | 9 %        N/A           285/294592 MB |
|-------------------------------------+----------------------------------------|
| 0000:ab:00.0 AMD Instinct MI350X VF | 0 %      65 °C   0          271/1000 W |
|   5       5       3        SPX/NPS1 | 9 %        N/A           285/294592 MB |
|-------------------------------------+----------------------------------------|
| 0000:b3:00.0 AMD Instinct MI350X VF | 0 %      60 °C   0          261/1000 W |
|   6       6       1        SPX/NPS1 | 11 %       N/A           285/294592 MB |
|-------------------------------------+----------------------------------------|
| 0000:bb:00.0 AMD Instinct MI350X VF | 0 %      63 °C   0          257/1000 W |
|   7       7       0        SPX/NPS1 | 0 %        N/A           285/294592 MB |
+-------------------------------------+----------------------------------------+
+------------------------------------------------------------------------------+
| Processes:                                                                   |
|  GPU        PID  Process Name          GTT_MEM  VRAM_MEM  MEM_USAGE     CU % |
|==============================================================================|
|    0       4236  N/A                     0.0 B     0.0 B      0.0 B    0.0 % |
|    1       4236  N/A                     0.0 B     0.0 B      0.0 B    0.0 % |
|    2       4236  N/A                     0.0 B     0.0 B      0.0 B    0.0 % |
|    3       4236  N/A                     0.0 B     0.0 B      0.0 B    0.0 % |
|    4       4236  N/A                     0.0 B     0.0 B      0.0 B    0.0 % |
|    5       4236  N/A                     0.0 B     0.0 B      0.0 B    0.0 % |
|    6       4236  N/A                     0.0 B     0.0 B      0.0 B    0.0 % |
|    7       4236  N/A                     0.0 B     0.0 B      0.0 B    0.0 % |
+------------------------------------------------------------------------------+
Process Name may require elevated permissions.

root@rs-mi350x8-mn-test1:~# rocm-smi
============================================ ROCm System Management Interface ============================================
====================================================== Concise Info ======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK   MCLK     Fan  Perf  PwrCap   VRAM%  GPU%  
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                   
==========================================================================================================================
0       2     0x75b0,   29921  60.0°C      268.0W    NPS1, SPX, 0        38Mhz  2000Mhz  0%   auto  1000.0W  0%     11%   
1       3     0x75b0,   45798  63.0°C      265.0W    NPS1, SPX, 0        38Mhz  2000Mhz  0%   auto  1000.0W  0%     9%    
2       4     0x75b0,   47340  60.0°C      268.0W    NPS1, SPX, 0        37Mhz  2000Mhz  0%   auto  1000.0W  0%     9%    
3       5     0x75b0,   32491  65.0°C      266.0W    NPS1, SPX, 0        38Mhz  2000Mhz  0%   auto  1000.0W  0%     11%   
4       6     0x75b0,   44280  61.0°C      266.0W    NPS1, SPX, 0        38Mhz  2000Mhz  0%   auto  1000.0W  0%     9%    
5       7     0x75b0,   27391  65.0°C      270.0W    NPS1, SPX, 0        38Mhz  2000Mhz  0%   auto  1000.0W  0%     9%    
6       8     0x75b0,   24821  60.0°C      260.0W    NPS1, SPX, 0        46Mhz  2000Mhz  0%   auto  1000.0W  0%     11%   
7       9     0x75b0,   42738  63.0°C      257.0W    NPS1, SPX, 0        38Mhz  2000Mhz  0%   auto  1000.0W  0%     0%    
==========================================================================================================================
================================================== End of ROCm SMI Log ===================================================

root@rs-mi350x8-mn-test1:~# rocm-smi --showtopo
============================ ROCm System Management Interface ============================
================================ Weight between two GPUs =================================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7         
GPU0   0            15           15           15           15           15           15           15           
GPU1   15           0            15           15           15           15           15           15           
GPU2   15           15           0            15           15           15           15           15           
GPU3   15           15           15           0            15           15           15           15           
GPU4   15           15           15           15           0            15           15           15           
GPU5   15           15           15           15           15           0            15           15           
GPU6   15           15           15           15           15           15           0            15           
GPU7   15           15           15           15           15           15           15           0            
================================= Hops between two GPUs ==================================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7         
GPU0   0            1            1            1            1            1            1            1            
GPU1   1            0            1            1            1            1            1            1            
GPU2   1            1            0            1            1            1            1            1            
GPU3   1            1            1            0            1            1            1            1            
GPU4   1            1            1            1            0            1            1            1            
GPU5   1            1            1            1            1            0            1            1            
GPU6   1            1            1            1            1            1            0            1            
GPU7   1            1            1            1            1            1            1            0            
=============================== Link Type between two GPUs ===============================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7         
GPU0   0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         
GPU1   XGMI         0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         
GPU2   XGMI         XGMI         0            XGMI         XGMI         XGMI         XGMI         XGMI         
GPU3   XGMI         XGMI         XGMI         0            XGMI         XGMI         XGMI         XGMI         
GPU4   XGMI         XGMI         XGMI         XGMI         0            XGMI         XGMI         XGMI         
GPU5   XGMI         XGMI         XGMI         XGMI         XGMI         0            XGMI         XGMI         
GPU6   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         0            XGMI         
GPU7   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         0            
======================================= Numa Nodes =======================================
GPU[0]		: (Topology) Numa Node: 0
GPU[0]		: (Topology) Numa Affinity: 0
GPU[1]		: (Topology) Numa Node: 0
GPU[1]		: (Topology) Numa Affinity: 0
GPU[2]		: (Topology) Numa Node: 0
GPU[2]		: (Topology) Numa Affinity: 0
GPU[3]		: (Topology) Numa Node: 0
GPU[3]		: (Topology) Numa Affinity: 0
GPU[4]		: (Topology) Numa Node: 1
GPU[4]		: (Topology) Numa Affinity: 1
GPU[5]		: (Topology) Numa Node: 1
GPU[5]		: (Topology) Numa Affinity: 1
GPU[6]		: (Topology) Numa Node: 1
GPU[6]		: (Topology) Numa Affinity: 1
GPU[7]		: (Topology) Numa Node: 1
GPU[7]		: (Topology) Numa Affinity: 1
================================== End of ROCm SMI Log ===================================
```





