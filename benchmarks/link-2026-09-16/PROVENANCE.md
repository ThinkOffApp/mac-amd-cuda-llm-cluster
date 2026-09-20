# The Mac↔Spark 10 GbE link, measured (16 Sep 2026)

> **TRANSPORT: TCP over Ethernet. Not RDMA, and not capable of it** — the Mac has no
> ConnectX or other RDMA NIC; `en12` is a Thunderbolt-to-10GBASE-T adapter. See
> [`TRANSPORT.md`](../../TRANSPORT.md). These are the numbers an RDMA transport on this
> pair would have to beat, not an RDMA result.

Link: MacBook `en12` **10.10.10.1** ↔ Spark 1 (ASUS Ascent GX10) **10.10.10.2**, direct
cable, no switch. MacBook Pro M5 Max, Darwin 25.6.0 (read from the iperf3 `system_info`
field in the committed JSON, so the host is checkable rather than asserted).

## Bandwidth — MEASURED by us

`iperf3 3.21`, TCP, 4 parallel streams, 128 KiB block, 10 s, one run each direction.

| direction | sender | receiver | retransmits |
|---|---:|---:|---:|
| Mac → Spark (`iperf3-20260916T180103Z.json`) | **9.420 Gbit/s** | 9.407 Gbit/s | **0** |
| Spark → Mac (`…-reverse.json`, `-R`) | **9.424 Gbit/s** | 9.413 Gbit/s | **0** |

Symmetric, at line rate, with zero retransmits. **Bandwidth is not the limitation on this
link.** Latency is, which is the next section and the reason this directory exists.

## Latency — MEASURED by us, and read the caveats before quoting

TCP ping-pong, `TCP_NODELAY` set, **3000 samples per size**, round-trip microseconds, one
sample per line in the committed `.txt` files. Recomputed from those raw files for this
commit rather than carried over from an earlier summary.

| payload | min | p10 | **median** | p90 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 B | 100.0 | 195.2 | **403.0** | 999.3 | 1047.9 | 7105.5 | 497.6 |
| 4 KiB | 102.8 | 222.1 | **244.0** | 801.5 | 1143.4 | 6180.6 | 413.2 |
| 64 KiB | 234.9 | 354.2 | **810.8** | 1157.3 | 1484.0 | 4336.0 | 776.2 |

**Three things about this table that a single median would hide.**

1. **The 64 B series is bimodal and non-stationary — do not quote one number for it.**
   Its first 200 samples have a median of **213 µs**; the whole 3000 have a median of
   **403 µs**; 500-sample windows across the run read 407 / 407 / 412 / 574 / 393 / 319.
   An earlier internal summary of ours circulated **229 µs** for this size, which is
   consistent with an early window and **is not reproducible from the committed file**.
   The committed file is the record; the 229 figure is withdrawn.
2. **64 B is *slower* than 4 KiB at the median**, which is backwards for a wire and points
   at host-side scheduling or coalescing rather than at the cable. The *floors* agree
   (100.0 vs 102.8 µs), so the transmission cost is the same; only the upper mass differs.
3. **4 KiB and 64 KiB are stable.** 500-sample window medians vary by under 5 % across the
   run for both. Those two rows are safe to quote; the 64 B row is not.

**Provenance gap, stated rather than papered over:** the ping-pong files carry no header —
no command line, no direction, no host. Their timestamps place them 16 minutes after the
iperf3 runs on the same link, and the payload sizes match the filenames, but **the
direction (Mac→Spark or Spark→Mac) is UNKNOWN** and we are not going to guess it. Future
captures in this repo should carry the header the RoCE captures in
[`benchmarks/roce-2026-09-17/raw/`](../roce-2026-09-17/raw/) already do.

An ICMP figure (300 pings, min 0.264 / avg 0.862 ms) has circulated internally alongside
these. **No artifact for it exists in this repo**, so it is not reproduced here. Separate
same-week ICMP spot checks on this link read 0.667–0.976 ms average, which is the same
order but is a different measurement.

## The contrast this repo exists to give — MEASURED by us vs NOT OURS

Ash Hart reports, for [MCDMA](https://github.com/ashhart/MCDMA), 4 KiB QD1 medians of
**7.6 µs WRITE** and **6.0 µs READ**, Mac→Spark. **Those are his measurements under his
conditions on his hardware; we did not run them and cannot vouch for them.** Ours is the
4 KiB row above.

| | our 10 GbE TCP, 4 KiB | MCDMA RDMA, 4 KiB QD1 (NOT OURS) |
|---|---:|---:|
| median | 244.0 µs round trip | 7.6 µs WRITE / 6.0 µs READ |

**The ratio depends on a boundary we cannot resolve from the published figures**, so here
is the honest range instead of one number. If his figure is a round trip like ours, the
gap is **≈32×**. If his is a one-way or half-round-trip figure and ours is halved to match,
it is **≈16×**. Either way the answer is **one to one-and-a-half orders of magnitude**, and
even our *best* sample (102.8 µs, once in 3000) is still more than 13× his median.

**Why that number is the most useful thing in this directory.** Bandwidth on this link is
already at line rate with zero retransmits, so a faster cable buys nothing. Per-exchange
latency is where the two transports differ by more than an order of magnitude, and it is
the term that multiplies by the number of crossings per token. A layer split crosses the
link roughly **once per token** and survives this latency; tensor parallelism crosses it
**once per layer per token** and does not. That is the whole practical consequence of this
table, and it is why the split results elsewhere in this repo look the way they do.

## Files

| file | what |
|---|---|
| `iperf3-20260916T180103Z.json` | forward bandwidth, complete iperf3 JSON including per-interval and `system_info` |
| `iperf3-20260916T180103Z-reverse.json` | the `-R` run |
| `pingpong-64B-20260916T181734Z.txt` | 3000 round-trip µs samples, 64 B |
| `pingpong-4096B-20260916T181736Z.txt` | 3000 round-trip µs samples, 4 KiB |
| `pingpong-65536B-20260916T181738Z.txt` | 3000 round-trip µs samples, 64 KiB |
