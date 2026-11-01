---
geometry:
  - landscape
  - margin=0.65in
fontsize: 10pt
---

# Pacman AI — Output Analysis Report

## 1. Observed performance

A short one-episode comparison was run with three ghosts, a 20-step episode cap, and a fixed maze seed. The results are a sample, not a statistically reliable ranking; ghost placement and game behavior remain stochastic.

| Agent | Kills | Score | Steps survived | Tiles travelled | Episode runtime | Timeouts |
|---|---:|---:|---:|---:|---:|---:|
| Minimax (depth 2) | 3 | 30 | 20 | 20 | 1.318 s | 0 |
| Alpha-Beta (max depth 3) | 2 | 20 | 20 | 18 | 2.401 s | 0 |
| MCTS (180 iterations, rollout 10) | 3 | 30 | 20 | 20 | 0.459 s | 0 |

On one five-ghost initial state, one observed decision took approximately 0.307 s for Minimax, 0.765 s for Alpha-Beta, and 0.166 s for MCTS. These are individual measurements rather than averages.

MCTS was the fastest in the short sample and tied for the most kills. Alpha-beta spent more time searching deeper and did not outperform Minimax in this single episode. More episodes and multiple seeds are needed before drawing a general performance conclusion. The `--assess` option runs all agents for a user-selected episode count and reports average kills, score, episode length, tiles travelled, and timeouts.
