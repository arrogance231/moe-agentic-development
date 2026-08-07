# DeepSpeed adapter (planned)

## Purpose

The DeepSpeed adapter will generate and validate ZeRO + MoE configuration
files. Consuming the `moe-training` skill's outputs, it will produce DeepSpeed
configs covering ZeRO stages and MoE settings (expert parallelism, capacity,
all-to-all communication) and validate them before training.

## Status

Placeholder — implementation planned in Phase 3.
