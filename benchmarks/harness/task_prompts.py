"""Task prompt library for the MoE benchmark harness — byte-identical across
arms per BENCHMARK.md (prompt parity requirement). Tasks 5 and 6 are authored
here from BENCHMARK.md's task descriptions since no harness previously wired
them in.
"""

TASKS = {
    "task1": {
        "name": "Architecture Design",
        "prompt": (
            "Design a 1B dense-equivalent MoE model.\n\n"
            "Produce a structured markdown architecture document with the "
            "following sections: Overview, Parameters (a table with digits, "
            "including a total-parameter figure, num_experts, and top_k), "
            "Routing choice (state top-1/top-2/learned/soft and justify it), "
            "Training implications, and Risks. You must specify: expert count, "
            "active experts (top-k), routing strategy, capacity factor, and "
            "auxiliary loss. Show your parameter math explicitly."
        ),
        "output_ext": "md",
        "scorer": "architecture",
    },
    "task2": {
        "name": "Training Setup",
        "prompt": (
            "Create a training configuration for the MoE architecture you would "
            "design for a 1B dense-equivalent MoE model (expert count, top-k, "
            "capacity factor as you see fit, stated explicitly). Produce a "
            "launchable training configuration (Megatron, DeepSpeed, or "
            "Hugging Face format) consistent with that architecture, as a JSON "
            "object with keys: framework, architecture (num_experts, top_k, "
            "hidden_size, num_layers, capacity_factor), parallelism "
            "(data_parallel, expert_parallel, tensor_parallel), batch_size, "
            "estimated_memory_gb, estimated_tokens_per_sec, and launch_notes."
        ),
        "output_ext": "json",
        "scorer": "task2_numeric",
    },
    "task3": {
        "name": "Debugging",
        "prompt": (
            "A training run has failed. Symptoms: loss went to NaN at step "
            "4200 after several hundred steps of normal decreasing loss; "
            "gradient norm spiked sharply in the 50 steps preceding the NaN; "
            "the run uses top-1 routing with capacity_factor=1.0 and no "
            "gradient clipping configured; expert load logs show 3 of 32 "
            "experts receiving >80% of routed tokens in the steps before the "
            "spike.\n\n"
            "Diagnose the failure and propose a fix. Respond as a JSON object "
            "with keys: identified_problems (list), root_causes (list), "
            "actions (list), evidence (list of the specific symptoms above "
            "that support your diagnosis)."
        ),
        "output_ext": "json",
        "scorer": "debugging",
    },
    "task4": {
        "name": "Optimization",
        "prompt": (
            "An existing MoE training run achieves 12,400 tokens/sec on 8x "
            "GPUs at 62% GPU utilization, using expert_parallel=8, "
            "micro_batch_size=4, no activation checkpointing, and FP32 "
            "optimizer states. Improve throughput.\n\n"
            "Produce a markdown report with sections: Changes (the specific "
            "config changes you'd make and why), Before/After metrics table "
            "(tokens/sec, GPU utilization %, memory), and Risks."
        ),
        "output_ext": "md",
        "scorer": "task4_numeric",
    },
    "task5": {
        "name": "Conflicting-guidance resolution",
        "prompt": (
            "You are given a hardware budget of 8x 80GB GPUs with "
            "standard (non-NVLink-full-mesh) interconnect, and a target "
            "dense-equivalent model size of 1B parameters.\n\n"
            "Choose the capacity factor, the auxiliary-loss coefficient, and "
            "the top-k routing value for this budget. Published guidance "
            "disagrees: capacity factors used in the literature range from "
            "1.0 to 2.0; auxiliary-loss coefficients range across two orders "
            "of magnitude (roughly 0.001 to 0.1); and both top-1 and top-2 "
            "routing are used in production systems with different capacity "
            "factor conventions.\n\n"
            "Produce a markdown decision document stating: the value you "
            "choose for each of the three parameters, the conflicting "
            "positions in the literature you weighed for each, your "
            "justification for the specific value chosen (not just a range), "
            "and the conditions under which you would revise the choice."
        ),
        "output_ext": "md",
        "scorer": "task5",
    },
    "task6": {
        "name": "Constrained-hardware design",
        "prompt": (
            "Design an MoE model under this constraint: the target training "
            "cluster has an inter-node interconnect capped at 25 Gbps "
            "(no InfiniBand), and the expert-parallel degree is fixed at 4 "
            "regardless of total GPU count, because the cluster topology "
            "groups GPUs into fixed pods of 4. Target dense-equivalent size "
            "is 1B parameters. No single published recipe assumes this "
            "combination of low-bandwidth interconnect and fixed small "
            "expert-parallel degree.\n\n"
            "Produce a structured markdown architecture document (Overview, "
            "Parameters table with digits including total-parameter figure, "
            "num_experts, and top_k, Routing choice with justification, "
            "Training implications, Risks) that explicitly satisfies the "
            "constraint: state the expert-parallel degree you use (must be 4) "
            "and the token-communication volume per step implied by your "
            "design, and explain how you kept it within the 25 Gbps "
            "interconnect budget."
        ),
        "output_ext": "md",
        "scorer": "task6",
    },
}

# Pre-registered incompatible-pair list for task5's internal-consistency check
# (BENCHMARK.md / SKILL.md example: "a top-1 routing choice paired with a
# capacity factor justified for top-2"). Fixed before any run is scored.
INCOMPATIBLE_PAIRS = [
    {
        "id": "top1_with_top2_capacity_language",
        "description": (
            "Routing choice states top-1 but capacity-factor justification "
            "language references top-2-style headroom (e.g. mentions "
            "'second expert' or 'top-2' while routing is declared top-1)."
        ),
        "routing_pattern": r"top-?1",
        "conflicting_pattern": r"top-?2",
    },
    {
        "id": "top2_with_top1_capacity_language",
        "description": (
            "Routing choice states top-2 but capacity-factor justification "
            "only discusses single-expert (top-1) token counting."
        ),
        "routing_pattern": r"top-?2",
        "conflicting_pattern": r"single[- ]expert|one expert only",
    },
    {
        "id": "capacity_factor_out_of_stated_range",
        "description": (
            "Capacity factor chosen falls outside the commonly cited 1.0-2.0 "
            "range without an explicit justification for the deviation."
        ),
    },
    {
        "id": "aux_loss_coefficient_out_of_stated_range",
        "description": (
            "Auxiliary-loss coefficient chosen falls outside the commonly "
            "cited 0.001-0.1 range without an explicit justification."
        ),
    },
]
