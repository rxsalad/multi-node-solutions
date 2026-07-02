import re
import subprocess
from collections import defaultdict

def get_fabric_ips():
    """Return {fabricX: ip}."""
    output = subprocess.run( ["ip", "-br", "a"], capture_output=True, text=True, check=True, ).stdout
    cidr_re = re.compile(r"([0-9a-fA-F:.]+)/\d+")
    fabrics = {}

    for line in output.splitlines():
        if not line.startswith("fabric"):
            continue
        parts = line.split()
        iface = parts[0].split("@")[0]
        for field in parts[2:]:
            m = cidr_re.match(field)
            if m:
                fabrics[iface] = m.group(1)
                break

    return fabrics

def get_hca_gids():
    output = subprocess.run(
        ["ibv_devinfo", "-v"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    hca_re = re.compile(r"^\s*hca_id:\s*(\S+)")
    gid_re = re.compile(r"^\s*GID\[\s*(\d+)\]:\s*(\S+)")
    hcas = defaultdict(dict)
    current = None

    for line in output.splitlines():
        if m := hca_re.match(line):
            current = m.group(1)
            continue

        if current and (m := gid_re.match(line)):
            hcas[current][int(m.group(1))] = m.group(2)

    return dict(hcas)


def mapping():
    hca_to_fabric = {}
    fabric_ips = get_fabric_ips()
    hca_gids = get_hca_gids()
    for hca, gids in sorted(hca_gids.items()):
        gid_lookup = { gid.lower(): idx for idx, gid in gids.items() }
        for fabric, ip in sorted(fabric_ips.items()):
            for gid, idx in gid_lookup.items():
                if ip.lower() in gid:
                    #print(f"  {fabric}: GID index {idx}")
                    hca_to_fabric[hca] = (fabric, idx)
                    break

    return hca_to_fabric

if __name__ == "__main__":

    hca_to_fabric_gid = mapping()
    print(hca_to_fabric_gid)

    #fabric_to_hca_gid = { fabric: (hca, gid) for hca, (fabric, gid) in hca_to_fabric_gid.items()}
    #print(fabric_to_hca_gid)

"""
{'mlx5_0': ('fabric0', 1), 'mlx5_1': ('fabric1', 1), 'mlx5_2': ('fabric2', 1), 'mlx5_3': ('fabric3', 1), 'mlx5_4': ('fabric4', 1), 'mlx5_5': ('fabric5', 1), 'mlx5_6': ('fabric6', 1), 'mlx5_7': ('fabric7', 1)}

{'fabric0': ('mlx5_0', 1), 'fabric1': ('mlx5_1', 1), 'fabric2': ('mlx5_2', 1), 'fabric3': ('mlx5_3', 1), 'fabric4': ('mlx5_4', 1), 'fabric5': ('mlx5_5', 1), 'fabric6': ('mlx5_6', 1), 'fabric7': ('mlx5_7', 1)}
"""