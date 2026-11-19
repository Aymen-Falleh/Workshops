# Lab 3 · Lost robot in a maze — Instructor Notes

**Fundamentals of AI · 2-hour workshop · ~15 min slot**
Notebook: `lab3_lost_robot_maze.ipynb` (34 cells, runs top-to-bottom in Colab). No dataset — the 8×8 maze world is built in numpy/matplotlib.

## Colab upload steps

1. Colab → **File → Upload notebook** → `lab3_lost_robot_maze.ipynb`. CPU fine.
2. **Runtime → Run all** — everything is local compute; whole notebook finishes in well under a minute after install.
3. Two blanks (plain editable cells, each followed by a collapsed **"Show solution"**). `#@param` sliders/dropdown are the knobs in cells 1d and 4a.
4. `<details>` checkpoint blocks render in Colab (same as Labs 1–2).

## The world (for your reference)

- 8×8 grid, start (0,0) top-left, exit (7,7) bottom-right. 4 actions: up/down/left/right.
- Rewards: `step_penalty` per move (default −1), `trap_penalty` on trap cells (default −10, episode ends), `exit_reward` (default +100). Episode cap: 200 steps. `slip_prob` = chance the robot moves a random direction instead.
- Three layouts, all solvable (BFS shortest *safe* path avoiding walls+traps):
  - **classic**: 14 steps — has walls forcing a route plus 4 traps.
  - **shortcut**: 14 steps safe, but the natural diagonal corridor runs between 10 traps — with slip > 0 it's risky.
  - **new_maze**: 18 steps — different walls/traps for the transfer experiment.

## Expected behavior (measured on a real nbconvert run, 500 episodes default)

| Step | Expected result |
|---|---|
| §2 random agent (300 eps) | mean ~11 steps, **100% trapped**, 0% reach exit — random dies fast because traps are everywhere |
| §4 training (defaults) | last-50 mean reward ≈ **+85**, mean steps ≈ **14** — matches the BFS-optimal safe path (14) |
| §5 policy arrows | greedy arrows trace the safe corridor; replay reaches EXIT in ~14 steps |
| **6a** step_penalty=0 | mean steps jumps to ~**170**, greedy rollout hits the 200 cap and never exits — free moves → no incentive for efficiency |
| **6b** shortcut + slip 0.2 | policy stays safe (rollout ~18 steps, untrapped); then trap_penalty=−2 → greedy rollout goes **straight into a trap in 3 steps** — it takes the risk when the penalty is cheap |
| **6c** old Q on new_maze | old policy fails: 200 steps, no exit; after retrain: **18 steps**, success |

## Checkpoint answers (also in collapsed `<details>` blocks)

- **§2 — why so bad?** 256 state-action pairs and zero memory of what worked; random walk + traps = hopeless.
- **§3 — α and γ?** α = how fast new experience overwrites old guesses (α=1 forgets instantly, α→0 never learns). γ = how much future rewards count (γ≈0 makes it myopic; γ≈0.95 lets trap penalties propagate backward so it plans ahead).
- Epsilon decays linearly to 0.02 over training — explore early, exploit late.

## Blank solutions

- **Blank #1** (epsilon-greedy, 1 line):
  `return np.random.randint(4) if np.random.rand() < eps else int(np.argmax(Q[s]))`
- **Blank #2** (Q-update, 1 line):
  `Q[s, a] + alpha * (r + gamma * best_next - Q[s, a])`
  (assigned to `Q[s, a] = ...`; `best_next` is already computed as 0 when done)

## Runtime & verification

- Executed end-to-end via `nbconvert --execute` with both blanks filled: **no errors, no warnings**; all maze plots and the replay rendered.
- Total wall time ~40 s including pip install and ~4 Q-learning runs (each ≤800 episodes, ≤200k updates). Seeded (`np.random.seed(42)`) → repeatable.
- Verified: learned classic policy = BFS shortest safe path (14 steps); new_maze retrain = 18 steps = its BFS optimum.
