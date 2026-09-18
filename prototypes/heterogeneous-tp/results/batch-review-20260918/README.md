# Batch correctness review

Derived from ClaudeMM batch_tp.py at 271a9f9. Added solo mode, pinned checkpoint hash validation, EOS-aware counting, root-only last-position head, rank barriers, mandatory initial/warmup/timed references and global timing suppression on failure. See ../../BATCHING.md for the protocol.

Initial checks: CPU solo batches 1/4; two CPU ranks batch4; MPS solo batch4. Final source checks: two CPU ranks batches2/8/16; MPS solo batch8. Every normal case used 128 prompt tokens, 8 new tokens, 1 warmup and 2 recorded runs; both ranks exited0 where applicable. These short local runs validate correctness and reporting, not a hardware performance claim.

The final source adds an exact pinned-checkpoint hash guard and removes an unused import after the initial cases. pre-hash-guard-source.py preserves the exact earlier source (its SHA matches the initial receipts). Final-case source hashes match the published batch_tp.py. Edge fixtures were generated from the earlier source: sequence0 forced to EOS at its second token produces counts2/8/8/8, all one-token sequences produce no decode rate, and corruption introduced ONLY during timed runs makes both ranks exit1 with initial correctness true and timing rows withheld. See batch-edges and validate_batch_edges.py.

Physical Mini/M5 and Mac/Spark batch performance is still unverified here. This implementation is static batching, not communication/computation overlap.
