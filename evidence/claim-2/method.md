# Method

The immutable output of source run fd92e2bc-ea94-4004-a820-62abf3e5e917 is replayed and independently checked. That parent run trained the paper's published 32x32 architecture for 20,000 steps over three deterministic seeds and enumerated terminal distributions exactly over all 1,024 states. The archived raw JSON has SHA-256 d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec.

Contract: Train the paper's neural ingredient GFNs, MOGFN, and HN-GFN on the 32x32 grid; evaluate exactly 128 simplex preferences for each k=2..5 over seeds 604, 1337, and 20260719. FALSIFY the exact printed ours=0.003 result only if 0.003 lies below the two-sided 95% t interval for every k. Independently require ours to beat both trained baselines in every seed/k cell and baseline aggregates to be within absolute L1 0.02 of the paper.

The fixed campaign reruns accepted regressions and independent checkers. Expensive raw training is provenance-linked and content-addressed rather than repeated in this cumulative node. Child branches vary committed code/config, never the command or locked environment.
