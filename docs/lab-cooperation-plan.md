# How the lab's machines work together (plan, 28 November 2026)

A plan, not a record of what runs: nothing below is set up yet unless it says so. It describes how the local AI
machines in one lab network share models and work, so that a request from any agent or person reaches the right
machine, and big models can use more than one.

## The machines

| Machine | Chip | Memory | Fast link | Role in the plan |
|---|---|---|---|---|
| Four ASUS GX10 (DGX Spark class) | NVIDIA GB10, sm_121 | 128 GB unified each | ConnectX-7 into the switch | serving, and pairs that split one big model |
| Bosgame M5 | AMD Strix Halo (Radeon 8060S) | 128 GB unified | SFP56 into the switch | serving with HIP, and host of the RTX 5090 |
| NVIDIA RTX 5090 | sm_120 | 32 GB | Thunderbolt 5 to its host | fastest single-user replies; travels with a MacBook today |
| Mac mini | Apple silicon | | LAN | Metal and MLX serving |
| NAS | | storage | LAN | backups and one copy of every model |
| MikroTik CRS812 | | | QSFP56 and SFP56 ports | the fast fabric between the machines |

The 5090 already runs from a MacBook over Thunderbolt (see the
[recipe](../recipes/inkling-2-mini-rtx5090-mac/README.md)); in the lab it moves into a Thunderbolt enclosure on the M5,
under Linux.

## Five layers

1. **One network, two speeds.** Every machine is on the ordinary LAN for management and remote work. The GB10s and
   the M5 are also on the CRS812 fabric, which carries model traffic. Every machine has a fixed name and address,
   kept in one inventory that the dashboard reads.

2. **One front door for models.** Every machine serves its models through TensorFold's OpenAI-compatible API. One
   router in front knows which machine serves which model, and agents and people talk only to the router.
   - It chooses by model, revision and settings first, then by health, then by load; never by "idle" alone.
   - If a machine fails, it retries elsewhere only before any output has streamed, and never repeats an action a
     tool already took.

3. **Big models across machines.** Two GB10s split one large model. We have served one model
   across two GB10s over 10 GbE
   ([two-Spark GLM serve](measurements-2026-09-16-18.md#the-two-spark-glm-serve-read-the-transport-per-run)); over the
   fast fabric, and with four GB10s as one group, it is not measured yet. Reading the
   prompt on a GB10 and generating on a Mac stays experimental while the question of exact output is open upstream.

4. **The RTX 5090 as the fast single-user engine.** On the M5 under Linux it serves the interactive assistant with the
   lowest delay, and gives the same-card comparison between the Mac route and Linux: the token ids of one fixed prompt
   set, which must be bit-identical (the comparison kit is on the [linux-compare branch](https://github.com/ThinkOffApp/mac-amd-cuda-llm-cluster/tree/linux-compare/linux-compare)). Its speed behind the M5 is
   not measured yet.

5. **Shared storage and memory.** Model weights live once on the NAS, are copied to each machine and checked by
   SHA-256 before use. The shared notes store, which lets one agent's lesson reach the others, is reachable from
   every machine. The Linux machines get regular backups to the NAS.

## Rules that keep it safe

- One GPU job per card at a time, behind a lock that every agent checks first.
- Each machine has one owner agent, named in the inventory, so two agents never contend for a card.
- Nothing leaves the lab network unless a person asks for it; every agent has its own key.

## Order of work

Each step is useful on its own.

1. Fixed names and addresses for every machine in the inventory; cable and health-check the fourth GX10.
2. The router, with every serving machine behind it; the agents switched over to it.
3. The RTX 5090 into its enclosure on the M5; the Linux token comparison; the assistant on it.
4. NAS backups for the Linux machines, and the shared model store.
5. GB10 pairs, then a group of four, for big models.

## Open questions

- Which travelling machines (MacBook with the 5090, mini PCs) join the lab network permanently.
- Which machine runs the router: one that never runs big jobs is the safer choice.
- Which model is served everywhere first.
