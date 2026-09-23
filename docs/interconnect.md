# Interconnect: the links on the bench and the three paths in detail

Per-link measurements (Thunderbolt, 10 GbE, 200 GbE RoCE), the latency correction, and the detail behind Paths 1 and 3 of the README's interconnect table. Path 2 (the Plyisty adapter) stays in the [README](../README.md#path-2-the-plyisty-adapter-which-we-own).

> Moved verbatim from the [README](../README.md) on 23 Sep 2026 to keep the README short.
> Nothing was cut. Where the text says "above", "below" or "this README", it refers to
> the README as it was before the move; the [README's last section](../README.md#details-and-all-numbers)
> lists where each part now lives.

## The three links on the bench

From the README section "Two desks, two halves of this repo".

Both boxes are visible in the fleet view on the screen behind, `gx10-6678` and `gx10-e6a8`, each
reporting CPU, memory and its **own** critical temperature rather than a guessed scale: 49.1 °C and
46.8 °C against the 104.8 °C every ACPI zone on a GX10 declares, beside the Strix Halo box at 73 °C
against its 110. The 5-inch panel on top is the Sparks' console, the e-ink tablet on the left is the
same fleet on a phone-class screen.

Three links. **Two of them have a bandwidth measurement in this repository; the
Thunderbolt one does not.**

| link | what it carries | measured |
|---|---|---|
| **Thunderbolt** | MacBook ↔ Strix Halo, Helsinki | **not measured here.** The [M5² card](../benchmarks/m5squared-card.png) reports inference throughput *across* this link, which is not the same as the link's bandwidth |
| **10 GbE** | MacBook ↔ Spark 1, direct cable | **9.42 Gbit/s** both ways, 0 retransmits ([iperf3](../benchmarks/link-2026-09-16/)); 8.7 Gbit/s by file transfer. **TCP, never RDMA.** Every GB10 split number on this page crossed it |
| **200 GbE** | Spark 1 ↔ Spark 2, one QSFP56 DAC | **185 Gbit/s** RDMA write, 93 % of line rate — synthetic, idle boxes, **and not used at all by the llama.cpp RPC runs here** (see the three-box topology note) |

Latency, not bandwidth, is what separates the 10 GbE link from an RDMA one: 4 KiB TCP
ping-pong median **244 µs** round trip against MCDMA's reported 7.6 µs.
[`benchmarks/link-2026-09-16/`](../benchmarks/link-2026-09-16/).

**The 200 GbE figure has a trap in it.** A single RDMA stream reaches 108.33 Gbit/s, about half the
port, and stays there. The full 185 only appears when **both PCIe functions of that one port are
driven at once** — 92.56 Gbit/s each, concurrently. Configure one interface, measure it, and you
will conclude the cable does 108 and move on, which is why NVIDIA's playbook has you address every
interface of a populated port rather than one.

```
ib_write_bw -d rocep1s0f0   -F --report_gbits -D 10                      108.33  (x2 passes)
ib_write_bw -d rocep1s0f0   -p 18515  +  -d roceP2p1s0f0 -p 18516        92.56 + 92.56  (x2 passes)
```

ConnectX-7, firmware 28.45.4028 both ends, MTU 1500, `Link type: Ethernet`, `GID index: 5` — the
RoCEv2 entry carrying the IPv4-mapped fabric address on `enp1s0f0np0` at each end, so the device and
GID selection is checkable rather than asserted. `BW peak` reads 0.00 in duration mode, so only the
average is a real number.

Complete per-process stdout and stderr, with the exact command and timestamps in every header, and
the full GID tables from both boxes:
[`benchmarks/roce-2026-09-17/raw/`](../benchmarks/roce-2026-09-17/raw/). The concurrent configuration
was run twice, independently, and agreed. Failed capture attempts are kept in that directory as
empty files rather than deleted.

This is RDMA write bandwidth between two idle machines, **not** what an inference run achieves over
the same wire. NCCL will be lower, and that is the figure that matters for a model split across both
Sparks. It has not been measured yet.

**One thing the photo makes easy to misread:** each Spark shows *two* 200 GbE interfaces
(`enp1s0f0np0` and `enP2p1s0f0np0`), and that is one physical QSFP port presented as two PCIe
functions, not two cables. NVIDIA states it plainly — "Each QSFP port appears as two independent
Linux Ethernet interfaces". There is exactly one cable between the boxes.


## Latency against RDMA, and the 20 Sep 2026 correction

From the README section "Related upstream work".

For scale, our own transport over the direct 10G cable measures a 4 KiB TCP
ping-pong median of **244 µs round trip** (3000 samples, best sample 103 µs)
against Ash Hart's 6–8 µs RDMA figures. The two figures may not share a
measurement boundary, so the honest gap is a range: **≈32×** if his is also a
round trip, **≈16×** if his is one-way and ours is halved to match. Either way it
is one to one-and-a-half orders of magnitude, and that gap is the reason both
projects exist. Raw samples and the full distribution:
[`benchmarks/link-2026-09-16/`](../benchmarks/link-2026-09-16/).

*Correction, 20 Sep 2026:* an earlier revision of this paragraph gave 350 µs and
"≈175 µs per one-way hop, a 20–30× gap". That figure does not reproduce from the
raw capture, which is now committed — recomputing the same file gives 244 µs. The
paragraph also implied a stable per-hop latency; the 64 B series in particular is
bimodal and drifts within a single run, so a single median is the wrong summary
for it. See that directory before quoting any latency from this repo.

## The interconnect paths in detail

From the README section "The interconnect: three paths, and what actually limits each one". The summary table and Path 2 are in the [README](../README.md#the-three-paths-at-a-glance).

### Path 1: the self-assembled adapter

The cheapest route, and not ours — it is the build
[Benjamin Ostrov](https://github.com/b-ostrov/MelonDMA) has put together: an
[ADT-Link USB4-to-PCIe adapter](https://www.adt.link/product/UT4G.html), a second-hand
Mellanox ConnectX-4, a power supply you provide yourself, a 3D-printed frame of his own
design, and a cable. He measures **28 Gbit/s** on it today, on a PCIe Gen3 adapter —
**his measurement, on his hardware, not ours.** The adapter's controller is an ASMedia
ASM2464PD on the UT3G and an ASM2464PDX on the UT4G; both are PCIe Gen4 x4, so the
upstream link is the same either way.

Prices vary in how firm they are, so each says which it is:

| component | price | how firm |
|---|---|---|
| ADT-Link USB4-to-PCIe adapter | 109 EUR AliExpress / 129 DFRobot / 170 Amazon | dated lookup, 15 Sep 2026 |
| the same adapter's UT4G variant | **unknown** | out of stock, no price shown |
| Mellanox ConnectX-4, second-hand | ~26 EUR (about $30) | **one** eBay listing — indicative, not a market survey |
| ATX power supply, user-supplied | 30-60 EUR | **estimate, not looked up** |
| 3D-printed frame | — | self-designed |
| direct-attach cable or transceiver | 20-30 EUR | **estimate, not looked up** |

### Path 3: the Helios enclosure and ConnectX-5, ordered

An [OWC Mercury Helios 5S](https://www.owc.com/solutions/mercury-helios-5s) — a
Thunderbolt 5 enclosure — holding a Mellanox
[MCX516A-CDAT](https://docs.nvidia.com/networking/display/connectx5en/specifications),
a ConnectX-5 Ex with dual 100-gigabit ports on PCIe Gen4 x16. **Over 800 EUR for the
two together, again a price actually paid.**

**Nothing here has been measured. The hardware has not arrived, and no benchmark in
this repository involves it.** What follows is arithmetic on published specifications,
not a result of ours.

The card is rated for 200 gigabits per second across its two ports together, but only
in a full-width slot. The Helios slot is **x16 mechanically and x4 electrically**, and
OWC quote the enclosure at up to about 6000 MB/s. On this path, then, the **enclosure
sets the ceiling rather than the card** — worth knowing before reading the card's
headline figure as something we expect to reach.

### What the money actually buys

The prices above span roughly four to one, and the speeds do not. Because all three
paths are capped by the Thunderbolt tunnel rather than by the card, the real comparison
is about **200-300 EUR for roughly 32 gigabits against over 800 EUR for roughly 48.**
Paying three to four times as much does not return three to four times the throughput,
and it very definitely does not buy 100 gigabits against 25.

What the extra money does buy is a Thunderbolt 5 tunnel rather than a Thunderbolt 4
one, a finished enclosure with its own power and cooling rather than a bare board and a
supply sitting on the desk, and a card that is not the component holding the link back.

**One detail worth noticing on the way past: the network card is the cheapest thing in
the build.** A ConnectX-4 at about 26 EUR sits inside an adapter that costs four to six
times more than the card does. Second-hand enterprise networking is nearly free. The
Thunderbolt bridge needed to get it onto a Mac is where the money actually goes.

**The interconnect plan is a comparison, not a replacement:** the 25-gigabit ConnectX-4
route we already have, measured against the 100-gigabit ConnectX-5 route once it lands.
Neither arm of that comparison has been run.

## Open question: can an ADT-Link Gen4 adapter replace the Helios 5S?

From the README section "What we could not test" (17 Sep 2026 split measurements).

**Open question, not yet measured: can an ADT-Link Gen4 adapter replace the Helios 5S?**
Note these are not the same kind of product. The Helios 5S is a finished enclosure with a case,
integrated power and cooling, and three TB5 ports. The ADT-Link is a bare adapter board plus a
Thunderbolt cable and a bare PCIe slot: its own documentation says "prepare the power supply
according to the power of the graphics card", so an external ATX PSU is the user's problem, and
the page states nothing about a case, cooling, or 75 W slot power. For a ~15-25 W ConnectX card
that is workable but it is a loose board and a PSU on the desk, not a swap of like for like.
The price difference should be read with that in mind.
Raised by [Benjamin Ostrov](https://github.com/b-ostrov/MelonDMA) on 20 Sep 2026, on the reasoning
that an ADT-Link is far cheaper. We have **not** benchmarked either enclosure, and what follows is
vendor and controller ceilings chained together, not an A/B:

| | host tunnel | PCIe | vendor figure |
|---|---|---|---|
| OWC Mercury Helios 5S | Thunderbolt 5, 80 Gb/s | 4.0 x4 electrical (x16 mechanical) | up to 6000 MB/s ≈ 48 Gb/s |
| ADT-Link UT3G / UT4G | USB4 Gen3x2, 40 Gb/s (32 on a TB3/TB4 host) | 4.0 x4 | ≈30.5 Gb/s usable on USB4v1 |

Both are PCIe Gen4 x4 on the card side, so on paper the difference is entirely upstream: the
ADT-Link's ASMedia ASM2464PDX is a USB4 Gen3x2 controller, while the Helios is Thunderbolt 5. On a
TB4 host the two should converge; on a TB5 host the Helios should win. An M5 Max MacBook Pro
reports Thunderbolt buses at up to 120 Gb/s, so the TB5 path is available to test.

For scale, the card itself is not the constraint in either case: an
[MCX516A-CDAT](https://docs.nvidia.com/networking/display/connectx5en/specifications) is
dual-port 100GbE on PCIe Gen4 **x16**, 200 Gb/s aggregate. Through either x4 path it cannot reach
even one full port.

**The experiment that would settle it**, in the order it should be run:

1. **First establish that the interconnect is the bottleneck at all.** Per the paragraph above,
   we have never measured inference wire utilisation. If it is not the limit, the enclosure
   question is moot and the money is better spent elsewhere. This step can be done today with the
   existing 10 GbE setup and costs nothing.
2. Same card, same host, same cable, same model, same workload; only the adapter changes.
   Report iperf3 both directions, a ping-pong RTT sweep (64 B / 4 KiB / 64 KiB), and an inference
   arm, each with repetitions and the raw traces.
3. Record the host's negotiated Thunderbolt mode per run. A TB5 enclosure on a port that
   negotiated TB3 is a different experiment from the one intended, and that is easy to do by
   accident.
4. State the PCIe link width and speed the card actually trains at, read from the host, not from
   the enclosure's spec sheet. Connector width is not electrical width, and electrical width is
   not what a given host negotiates.

Until at least step 1 exists, no purchase of either is justified by anything in this repository.
