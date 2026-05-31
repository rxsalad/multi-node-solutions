# Multi-Node Environment Setup for DOKS with AMD GPUs

This guide provides a basic introduction and step-by-step instructions for setting up a multi-node DOKS environment for testing and applications using AMD GPU nodes, including MI325x8 or MI350x8.

## GPU Network Introduction

All DOKS workers have Internet access via `eth0` at 10 Gb/s and VPC connectivity via `eth1` at 25 Gb/s, while the GPU network supports inter-node GPU-to-GPU communication via [RDMA/RoCE](https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/how-to/roce-network-config.html) for distributed training and inference workloads.

A single GPU Fabric can support up to 122 homogeneous nodes (MI325x8 or MI350x8) after reserving 6 nodes as spares. These nodes can be assigned across multiple customers, with VLANs used to isolate and segregate their traffic.

The GPU Fabric is rail-only at the moment and consists of 8 independent forwarding planes. Each plane connects at 400 Gb/s to [an NVIDIA Mellanox ConnectX-7 NIC](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/network/nic-installation.html#nvidia-mellanox-cx-7-400gx1) on MI325x8 node or [an AMD Pensando Pollara NIC](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/network/nic-installation.html#amd-pensando-pollara-400-ai-nic) on MI350X nodes. NICs across different planes are isolated and cannot communicate with each other.

**GPU Network with MI350x8 nodes:**

<img src="amd-mn-doks/amd-mn-doks.png" alt="GPU Network" width="900" height="350"/>

The 8 RDMA/RoCE NICs in DOKS workers are configured and managed as `fabric0–fabric7`, with an IPv6 addresses automatically assigned to each inferace.

The `fabric0–fabric7` can be treated as standard Ethernet interfaces, supported by the kernel TCP/IP stack in DOKS workers (Debian 13). You can run HTTP or ping over their IP addresses, and traffic can be managed via iptables.

When RDMA/RoCE applications run over the 8 NICs using RoCEv2 (UDP over IP), traffic is handled directly by the NICs, bypassing the kernel TCP/IP stack. As a result, the RX and TX packet counters for `fabric0–fabric7` do not increment, even though the traffic is transmitted successfully.

**Addtional Details:** RDMA/RoCE applications require access to both the RDMA subsystem and the associated Ethernet interfaces. First, applications must be able to access the RDMA character devices under `/dev/infiniband`. These devices provide the verbs interface used to create and manage RDMA resources such as queue pairs, memory regions, completion queues, and other RDMA objects.They also require access to the corresponding Ethernet interfaces, which provide the network context necessary for communication, including IP configuration, MTU settings, VLAN membership, and QoS parameters. Although data transfer occurs through the RDMA data path, the Ethernet interfaces remain essential for address resolution (for example, GID resolution), route selection, and endpoint discovery.

**Out-of-Band TCP/IP Channel (Control Plane)**

RDMA/RoCE applications still require an out-of-band TCP/IP channel for bootstrap/initialization and coordination, such as exchanging rank or connection information. In DOKS, there are several ways to provide this channel:

**Scenario 1:** Pods over DOKS Workers

In typical DOKS deployments, Pods running on DOKS workers use their own `eth0` interface to communicate with other Pods, Services, Nodes and the Internet. This interface can also serve as the out-of-band TCP channel for RDMA/RoCE applications. In this model, the RDMA network devices (`fabric0–fabric7`) are moved from the host network namespace into the Pod network namepsace. Once transfered, these devices are no longer visible or accessible from the host network namespace and are available exclusively within the Pod.

**Addtional Details:** [DOKS uses Cilium CNI with VPC-native routing](https://docs.digitalocean.com/products/kubernetes/details/features/#vpc-native-networking), enabling Pod-to-Pod,Pod-to-Service, Pod-to-Node/External traffic to be forwarded directly through the Linux kernel networking stack without overlay encapsulation. Pod-to-Pod traffic is routed natively across the VPC and does not require NAT. Cilium leverages eBPF for routing, service load balancing, network policy enforcement, and NAT where necessary (for example, egress traffic), eliminating the need for kube-proxy and replacing traditional iptables/IPVS-based service handling with an eBPF-based datapath, while the Cilium agent operates solely in the control plane and is not involved in packet forwarding.

**Scenario 2:** Pods Sharing Host Network

For Pods configured with the host network namespace (where interfaces such as `eth0` ,`eth1` and `fabric0–fabric7` are shared), the `eth1` interface is used as the dedicated out-of-band TCP/IP channel.

This setup is similar to a non-DOKS environment, where containers are launched with direct access to GPUs and RDMA/RoCE devices on the host. For additional details, please refer to [the link](https://github.com/rxsalad/multi-node-solutions/blob/main/mi350x8-mn-pytorch-testing.MD#how-to-access-rdmaroce-nics-within-containers).

**Intra-node and Inter-node GPU Communication (Data Plane)**

For GPU-to-GPU communication, the preferred paths are [XGMI](https://instinct.docs.amd.com/projects/virt-drv/en/latest/userguides/XGMI_configuration.html) (AMD Infinity Fabric) for intra-node transfers and 400 Gigabit Ethernet with RDMA for inter-node transfers. PCIe and shared memory (intra-node) and TCP/IP (inter-node, eth1) serve only as fallback options. Numerous [RCCL environment variables](https://rocm.docs.amd.com/projects/rccl/en/develop/api-reference/env-variables.html) can be configured to control these communication paths, and they should be carefully considered.

**The following summarizes the different communication paths, their roles and traffic in distributed AI/ML applications on DOKS:**

| Interface / Path | Bandwidth | Role |
|------------------|----------|------|
| Scenario 1: Pod eth0 (Cilium CNI with VPC-native routing), <br> for Pod-to-Pod/Service/Node/External traffic | 10~25 Gb/s | SSH; <br>HTTP APIs for serving user traffic;<br> Downloads and uploads (models, datasets, libraries, and updates); <br>HTTP APIs for distributed inference across Router, Prefiller, and Decoder;<br> Bootstrap/initialization and coordination for RDMA/RoCE; <br>Fallback path for RDMA/RoCE |
| Scenario 2: Host eth0 | 10 Gb/s | SSH; <br>HTTP APIs for serving user traffic;<br> Downloads and uploads (container images, models, datasets, libraries, and updates) |
| Scenario 2: Host eth1 | 25 Gb/s | NFS (models/checkpoints, datasets, logs, and metrics), exposed to Pods through bind mounts; <br>HTTP APIs for distributed inference across Router, Prefiller, and Decoder;<br> Bootstrap/initialization and coordination for RDMA/RoCE; <br>Fallback path for RDMA/RoCE |
| fabric0–fabric7 | 400 Gb/s each | RDMA/RoCE for inter-node GPU-to-GPU communication (used by RCCL);<br> Training traffic (gradients, parameters, optimizer states, activations);<br> Inference traffic (activations, KV cache) |
| XGMI (not user-configurable) | 7×128 GB/s per MI325 GPU<br> 7×153.6 GB/s per MI350 GPU   | Intra-node GPU-to-GPU communication (used by RCCL);<br>  Training traffic (gradients, activations, optimizer states);<br> Inference traffic (activations, KV cache) |

## Environment Setup 

Ensure that you have at least 2 nodes in the same GPU Fabric assigned to your account.

Provision a DOKS cluster with 1 CPU worker and 2 MI350x8 workers using doctl:

``` bash
doctl vpcs list
doctl compute size list | grep gpu 
doctl compute size list | grep gpu 
doctl kubernetes cluster list

# MI350x8, default VPC
# Use a fabric-enabled size; otherwise, RDMA/RoCE will not be enabled.
doctl kubernetes cluster create rs-amd-sfo2-test \
  --region sfo2 \
  --version 1.34.1-do.2 \
  --node-pool "name=rs-mn-cpu-pool;size=s-8vcpu-16gb;count=1" \
  --node-pool "name=rs-mn-mi350x8-pool;size=gpu-mi350x8-2304gb-fabric-contracted;count=2" \
  --tag "rs-test"

doctl kubernetes cluster get rs-amd-sfo2-test -o json
doctl kubernetes cluster delete rs-amd-sfo2-test

# MI350x8 + MI325x8
doctl kubernetes cluster create rs-amd-tor1-test \
  --region tor1 \
  --vpc-uuid d7489d54-76ff-47b6-b214-2162e69c52a2 \
  --version 1.34.1-do.2 \
  --node-pool "name=rs-mn-cpu-pool;size=s-4vcpu-8gb-amd;count=1" \
  --node-pool "name=rs-mn-mi325x8-pool;size=gpu-mi325x8-2048gb-fabric-contracted;count=2" \
  --node-pool "name=rs-mn-mi350x8-pool;size=gpu-mi350x8-2304gb-fabric-contracted;count=2" \
  --tag "rs-test"
``` 

**Notes:** The memory information shown in the GPU instance size refers to the physical host server. Since a DOKS worker runs as a virtual machine, the usable memory inside the VM is smaller. You can verify the actual available memory within the VM (1.2 TiB for MI325x8 instances and 2.0 TiB for MI350x8 instances).

The 3 workers have been created and are now managed by DOKS.

``` bash
# doctl compute droplet list

ID           Name                             Public IPv4        Private IPv4    Public IPv6                Memory     VCPUs    Disk    Region    Image                                 VPC UUID                                Status    Tags                                                       Features                                       Volumes  
574372194    rs-mn-cpu-pool-381iam            167.71.156.33      10.120.0.6                                 16384      8        320     sfo2      Debian do-kube-1.34.1-do.2            996966ce-4eb8-4ed5-9bf2-7ff1688b5825    active    k8s,k8s:b8945e9f-2a50-469d-9eb6-7b0d3ad9f614,k8s:worker    private_networking                             
574372195    rs-mn-mi350x8-pool-381iaq        159.65.102.82      10.120.0.3                                 2097152    192      2046    sfo2      Debian do-kube-1.34.1-do.2-gpu-amd    996966ce-4eb8-4ed5-9bf2-7ff1688b5825    active    k8s,k8s:b8945e9f-2a50-469d-9eb6-7b0d3ad9f614,k8s:worker    droplet_agent,private_networking               
574372196    rs-mn-mi350x8-pool-381iaf        159.89.134.60      10.120.0.11                                2097152    192      2046    sfo2      Debian do-kube-1.34.1-do.2-gpu-amd    996966ce-4eb8-4ed5-9bf2-7ff1688b5825    active    k8s,k8s:b8945e9f-2a50-469d-9eb6-7b0d3ad9f614,k8s:worker    droplet_agent,private_networking      
```

## DOKS Worker Check (Host OS)

There are 2 ways to access DOKS workers to check system information and troubleshoot issues:

- **Privileged DaemonSet approach:** Deploy a privileged DaemonSet that schedules a pod on each node, then use `kubectl exec` to access the pod on the target node. Refer to [doks-debug](https://github.com/digitalocean/doks-debug) for details.

- **Web Console approach:** Use the Do web console. This requires a firewall rule allowing TCP port 22 on the target DOKS workers. You can use [tag-based matching](https://docs.digitalocean.com/products/networking/firewalls/details/features/), as all DOKS workers are labeled with the **k8s:worker** tag. You may also add your public SSH keys via the web console to enable future SSH access without using the web console.

### GPU Check

Perform a basic system and AMD GPU check on the DOKS GPU workers:

``` bash
uname -r
cat /etc/os-release
dkms status

# Check if the AMD GPU kernel driver is currently loaded and active.
lsmod | grep amdgpu 

amd-smi version
amd-smi list 
amd-smi xgmi
amd-smi topology

rocm-smi
rocm-smi --showtopo
rocminfo

# Basic Tools
apt update
apt install net-tools
```

### RDMA/RoCE Check

Run the following commands on the DOKS GPU workers to verify network connectivity and RDMA/RoCE configuration.

``` bash
# Provides an overview of all network interfaces and their IPs.
ip -br a

# Shows detailed configuration and status information for each interface.
ifconfig

# Lists all RDMA devices exposed by the kernel
ls /sys/class/infiniband/

# Provides detailed hardware-level information about each RDMA device’s capabilities and configuration.
ibv_devices
ibv_devinfo -v | grep GID

# Displays the mapping between RDMA devices and network interfaces along with link status.
rdma link

# Ping the remote eth0 and eth1 addresses
ping xxx 

# Extract the IPv6 address assigned to each fabric interface.
for i in {0..7}; do
  echo "fabric$i:"
  ip -6 addr show dev fabric$i | grep inet6
done
```

| DOKS Worker    | rs-mn-mi350x8-pool-381iaf  | rs-mn-mi350x8-pool-381iaq  | 
|----------------|----------------|-----------------|
| eth0-internet  | 159.89.134.60 | 159.65.102.82 |
| eth1-vpc       | 10.120.0.11/20  | 10.120.0.3/20 | 
| fabric0  | fe80::490:81ff:fe49:c440/64 | fe80::490:81ff:fe4b:e670/64  | 
| fabric1  | fe80::490:81ff:fe49:488/64 | fe80::490:81ff:fe4b:ca20/64 | 
| fabric2  | fe80::490:81ff:fe49:18f8/64 | fe80::490:81ff:fe4c:1a90/64  | 
| fabric3  | fe80::490:81ff:fe4a:6fb8/64 | fe80::490:81ff:fe4b:fc78/64 |
| fabric4  | fe80::490:81ff:fe49:85d0/64 | fe80::490:81ff:fe4c:1790/64  | 
| fabric5  | fe80::490:81ff:fe49:cbc0/64 | fe80::490:81ff:fe4c:f98/64 | 
| fabric6  | fe80::490:81ff:fe49:d280/64 | fe80::490:81ff:fe4c:1610/64 | 
| fabric7  | fe80::490:81ff:fe49:1eb0/64  | fe80::490:81ff:fe4b:f510/64  | 


Each RDMA/RoCE fabric interface is assigned a [link-local IPv6 address](https://www.geeksforgeeks.org/computer-networks/link-local-address/) (**fe80::/10**). This is a common approach in HPC and AI clusters, where link-local addressing provides deterministic interface binding and reliable neighbor discovery for RDMA communication. Some large-scale RoCEv2 deployments may additionally assign routable IPv4 or IPv6 addresses to fabric interfaces to support broader networking, operational, and management requirements.

To verify fabric connectivity, test communication between the fabric interfaces on two worker nodes. Because link-local IPv6 addresses are scoped to a specific interface and are not routable, the source interface must be explicitly specified using the `-I` option.

``` bash
# rs-mn-mi350x8-pool-381iaf to rs-mn-mi350x8-pool-381iaq 

ping -c 1 -6 -I fabric0 fe80::490:81ff:fe4b:e670
...
ping -c 1 -6 -I fabric7 fe80::490:81ff:fe49:1eb0
```

The destination link-local IPv6 address does not need to belong to the same fabric interface number on the peer node. Connectivity succeeds as long as the destination address is reachable on the Layer 2 network associated with the selected local fabric interface.

``` bash
# rs-mn-mi350x8-pool-381iaf to rs-mn-mi350x8-pool-381iaq 

ping -c 1 -6 -I fabric7 fe80::490:81ff:fe4b:e670
...
ping -c 1 -6 -I fabric0 fe80::490:81ff:fe49:1eb0
```

### VPC Native Routing Check

Check host routing table on DOKS CPU and GPU workers:

``` bash
root@rs-mn-mi350x8-pool-381iaf:~# ip route

# Default Route
default via 159.89.128.1 dev eth0 proto static 

# VPC Peering Routes through the VPC router (10.120.0.1)
10.0.0.0/8 via 10.120.0.1 dev eth1 metric 101 mtu 1500 
172.16.0.0/12 via 10.120.0.1 dev eth1 metric 101 mtu 1500 
192.168.0.0/16 via 10.120.0.1 dev eth1 metric 101 mtu 1500 

# Additional VPC-local network segment via eth0 
10.46.0.0/16 dev eth0 scope link 

# VPC Node Network via eth1 
10.120.0.0/20 dev eth1 proto kernel scope link src 10.120.0.11 

# Pod Network
10.121.0.0/25 via 10.120.0.6 dev eth1 proto kernel                          # to 128 Pods on rs-mn-cpu-pool-381iam 
10.121.0.128/25 via 10.120.0.3 dev eth1 proto kernel                        # to 128 Pods on rs-mn-mi350x8-pool-381iaq
10.121.1.0/25 via 10.121.1.67 dev cilium_host proto kernel src 10.121.1.67  # to 128 Pods on the local node - rs-mn-mi350x8-pool-381iaf

# Local pod interface route (CNI endpoint for host-pod communication)
10.121.1.67 dev cilium_host proto kernel scope link 

# Public Internet subnet 
159.89.128.0/20 dev eth0 proto kernel scope link src 159.89.134.60 
```

The routing table shows a VPC-native Kubernetes networking setup where eth0 provides public internet access and eth1 serves as the VPC underlay. Private RFC1918 traffic (10/8, 172.16/12, 192.168/16) is routed through the VPC gateway (10.120.0.1). Pod networking is VPC-native, with each node pool’s pod CIDR routed directly over the VPC network via static next-hop routes, enabling inter-node pod communication without overlay encapsulation. Local pod traffic is handled through the cilium_host interface, while Cilium manages pod-to-node integration.


The routing table shows a VPC-native Kubernetes networking setup where `eth0` provides public internet access and `eth1` serves as the VPC underlay. Private RFC1918 traffic (10/8, 172.16/12, 192.168/16) is routed through the VPC gateway (10.120.0.1). Pod networking is VPC-native, with each node pool’s pod CIDR routed directly over the VPC network via static next-hop routes, enabling inter-node pod communication without overlay encapsulation. Local pod traffic is handled through the cilium_host interface, while Cilium manages pod-to-node integration.

The DOKS service subnet (ClusterIP range, e.g. 10.120.32.0/19) does not appear in the host routing table because it is not a routable network at L3. Instead, it is a virtual service CIDR implemented in the dataplane (Cilium eBPF and/or kube-proxy iptables/IPVS rules). Service IPs are translated in-kernel to backend pod endpoints rather than being forwarded via the Linux routing table, so no ip route entries are installed for this range.




## DOKS Cluster Check

``` bash
# kubectl get nodes -o wide

NAME                        STATUS   ROLES    AGE     VERSION   INTERNAL-IP   EXTERNAL-IP     OS-IMAGE                       KERNEL-VERSION        CONTAINER-RUNTIME
rs-mn-cpu-pool-381iam       Ready    <none>   3h47m   v1.34.1   10.120.0.6    167.71.156.33   Debian GNU/Linux 13 (trixie)   6.12.48+deb13-amd64   containerd://1.7.28
rs-mn-mi350x8-pool-381iaf   Ready    <none>   3h41m   v1.34.1   10.120.0.11   159.89.134.60   Debian GNU/Linux 13 (trixie)   6.12.48+deb13-amd64   containerd://1.7.28
rs-mn-mi350x8-pool-381iaq   Ready    <none>   3h41m   v1.34.1   10.120.0.3    159.65.102.82   Debian GNU/Linux 13 (trixie)   6.12.48+deb13-amd64   containerd://1.7.28


# kubectl get daemonset -ALL
NAMESPACE     NAME                                        DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                                                                                                  AGE     L
kube-system   amd-gpu-device-plugin                       2         2         2       2            2           doks.digitalocean.com/gpu-brand=amd,kubernetes.io/os=linux                                                     3h47m   
kube-system   cilium                                      3         3         3       3            3           kubernetes.io/os=linux                                                                                         3h51m   
kube-system   cpc-bridge-proxy-ebpf                       3         3         3       3            3           kubernetes.io/os=linux                                                                                         3h48m   
kube-system   csi-do-node                                 3         3         3       3            3           kubernetes.io/os=linux                                                                                         3h48m   
kube-system   do-node-agent                               1         1         1       1            1           kubernetes.io/os=linux                                                                                         3h48m   
kube-system   do-node-agent-amd-device-metrics-exporter   2         2         2       2            2           doks.digitalocean.com/gpu-brand=amd,kubernetes.io/os=linux                                                     3h48m   
kube-system   do-node-agent-nvidia-dcgm-exporter          0         0         0       0            0           doks.digitalocean.com/gpu-brand=nvidia,doks.digitalocean.com/nvidia-dcgm-enabled=true,kubernetes.io/os=linux   3h48m   
kube-system   doks-telemetry-config-reloader              3         3         3       3            3           kubernetes.io/os=linux                                                                                         3h52m   
kube-system   rdma-shared-dp-ds                           2         2         2       2            2           doks.digitalocean.com/is-gpu-fabric-connected=true,kubernetes.io/os=linux                                      3h47m   

# kubectl get pods -ALL
NAMESPACE     NAME                                              READY   STATUS    RESTARTS   AGE     L
kube-system   amd-gpu-device-plugin-2l72n                       1/1     Running   0          3h42m   
kube-system   amd-gpu-device-plugin-wf9gr                       1/1     Running   0          3h42m   
kube-system   cilium-hvrqg                                      2/2     Running   0          3h48m   
kube-system   cilium-mbz2g                                      2/2     Running   0          3h42m   
kube-system   cilium-nm9wd                                      2/2     Running   0          3h42m   
kube-system   coredns-7c475d69-4ztgp                            1/1     Running   0          3h46m   
kube-system   coredns-7c475d69-8t9ft                            1/1     Running   0          3h46m   
kube-system   cpc-bridge-proxy-ebpf-5v27z                       1/1     Running   0          3h42m   
kube-system   cpc-bridge-proxy-ebpf-n9x7k                       1/1     Running   0          3h42m   
kube-system   cpc-bridge-proxy-ebpf-pk5b2                       1/1     Running   0          3h47m   
kube-system   csi-do-node-nmvc2                                 2/2     Running   0          3h42m   
kube-system   csi-do-node-wvczt                                 2/2     Running   0          3h42m   
kube-system   csi-do-node-zjxwb                                 2/2     Running   0          3h46m   
kube-system   do-node-agent-74z6r                               1/1     Running   0          3h46m   
kube-system   do-node-agent-amd-device-metrics-exporter-9p7pr   3/3     Running   0          3h42m   
kube-system   do-node-agent-amd-device-metrics-exporter-c5qvx   3/3     Running   0          3h42m   
kube-system   doks-telemetry-config-reloader-4dp8p              1/1     Running   0          3h42m   
kube-system   doks-telemetry-config-reloader-96t4n              1/1     Running   0          3h42m   
kube-system   doks-telemetry-config-reloader-x5gsn              1/1     Running   0          3h48m   
kube-system   hubble-relay-7d6bc4d6f4-htj5k                     1/1     Running   0          3h50m   
kube-system   hubble-ui-b95c9f464-kgqnf                         2/2     Running   0          3h47m   
kube-system   konnectivity-agent-54996dbbb4-xxxm8               1/1     Running   0          3h47m   
kube-system   konnectivity-agent-54996dbbb4-zlpns               1/1     Running   0          3h47m   
kube-system   rdma-shared-dp-ds-6pzt6                           1/1     Running   0          3h42m   
kube-system   rdma-shared-dp-ds-mfhxs                           1/1     Running   0          3h42m   
```






