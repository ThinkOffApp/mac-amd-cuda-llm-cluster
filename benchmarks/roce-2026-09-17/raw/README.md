# Unfiltered RoCE capture, 17 Sep 2026

Every file is a program's complete stdout+stderr with the exact command and
timestamps in its header. **Nothing here is filtered.** The file one level up,
`../ib_write_bw-raw.log`, is a *summary* despite its name: it was produced with
`sed`/`grep` in the pipeline, which codexmb caught. It is kept for continuity,
not as the raw record.

| file | what |
|---|---|
| `A-client-single.txt` | single function, client, full header: `GID index : 5`, `Link type : Ethernet`, both address vectors |
| `A-server-single.txt` | the server side of that same run |
| `B-client-0.txt`, `B-client-1.txt` | **both PCIe functions concurrently**, client side, one file per process, complete journals |
| `B-server-0.txt`, `B-server-1.txt` | the server side of that run |
| `B-client-f0/f1`, `B-server-f0/f1` | a **second, independent** concurrent pass. Different iteration counts, same 92.57 result, so this is a repeat rather than a duplicate |
| `gids-asus1.txt`, `gids-asus2.txt` | the full GID table per index plus `ibv_devinfo`, so index 5 is verifiable rather than asserted |
| `env-gid-c1.txt`, `env-gid-c2.txt`, `gids-.txt` | **empty.** Failed capture attempts, kept rather than deleted |

## Results

```
single function   108.32  108.33  108.33 Gb/s
both concurrent   92.57 + 92.57   and   92.56 + 92.56   = 185.1 Gb/s
```

`GID index 5` on both boxes is the RoCEv2 entry carrying the IPv4-mapped fabric
address, `ndev=enp1s0f0np0`:

```
asus1  gid[5] 0000:...:ffff:c0a8:640a  type=RoCE v2  ndev=enp1s0f0np0   (192.168.100.10)
asus2  gid[5] 0000:...:ffff:c0a8:640b  type=RoCE v2  ndev=enp1s0f0np0   (192.168.100.11)
```

## What this does NOT establish

- `BW peak` reads 0.00 in duration mode; only the average is a reading.
- RDMA write between idle boxes is not inference traffic.
- The concurrent passes ran while both machines were downloading model weights
  at ~15-18 MB/s each. That is small beside a 185 Gbit/s fabric and it is on a
  different NIC, but it is a condition, so it is stated.
- **Nothing here shows the configuration surviving a reboot.** Neither box has
  rebooted since the fabric was configured.
