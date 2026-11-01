"""
student_assignment.py — AI assignment: Pacman-like game agents
==============================================================

Your task: implement three AI agents that play the Pacman-like game,
plus a shared evaluation function used by all of them.

Each agent receives the current game observation and a live PacmanGame
instance, and must return an Action.

Run this file to test:
    python student_assignment.py                 # human window
    python student_assignment.py minimax         # evaluate minimax (headless)
    python student_assignment.py alphabeta
    python student_assignment.py mcts
    python student_assignment.py alphabeta 10    # 10 episodes instead of 5
    python student_assignment.py alphabeta --depth 8   # override the depth
    python student_assignment.py run minimax     # watch agent play in window

Search depth / budget for every agent lives in the AGENT SETTINGS block
below — tune them there (or use --depth on the command line).

Actions (13 total — see the Action enum in game.py):
    Move:  UP, DOWN, LEFT, RIGHT
    Shoot: SHOOT_UP, SHOOT_DOWN, SHOOT_LEFT, SHOOT_RIGHT,
           SHOOT_UP_LEFT, SHOOT_UP_RIGHT, SHOOT_DOWN_LEFT, SHOOT_DOWN_RIGHT
    Stay:  STAY — player skips turn; ghosts still move.
           Returned automatically when your agent exceeds the 5-second time
           limit per move.
    A shot hits a ghost on the adjacent tile in that direction (diagonals
    included) and costs 1 ammo whether it hits or not. Diagonal shots can be
    turned off with GameConfig(diagonal_shooting=False).
    Always build your action list from game.get_legal_actions() rather than
    hard-coding it — shoot actions only appear when the player has ammo.

Useful game API:
    game.clone()             → independent copy for simulation (no window)
    game.get_state()         → GameState snapshot
    game.set_state(state)    → restore from snapshot
    game.get_legal_actions() → list[Action] valid this turn
    game.step(action)        → (obs, reward, done, info)
    game.done                → bool
    game.player.x/y/ammo/alive
    game.ghosts              → list[Ghost]  (each has .x .y .alive .behavior)

──────────────────────────────────────────────────────────────────────────────
Game State Reference  (everything evaluate_state / your agent can read)
──────────────────────────────────────────────────────────────────────────────

Player:
    p = game.player
    p.x, p.y            # current position (column, row)
    p.ammo              # shots remaining (starts at 1)
    p.alive             # bool
    p.tiles_traveled    # total tiles walked so far

Ghosts:
    game.ghosts                              # list[Ghost] — alive or dead
    alive_ghosts = [g for g in game.ghosts if g.alive]
    g.x, g.y, g.alive, g.behavior           # per ghost fields

    Ghost threat levels (GhostBehavior enum):
        GhostBehavior.RANDOM     # wanders — least dangerous
        GhostBehavior.FOLLOW     # BFS chases you directly
        GhostBehavior.LOOKAHEAD  # plans ahead — most dangerous

Walls:
    game.walls                   # set of (x, y) tuples — impassable tiles
    if (x, y) in game.walls: …  # check before moving

Grid size:
    game.config.grid_width   # columns
    game.config.grid_height  # rows

Ammo pickup timing (+1 ammo every tiles_per_ammo tiles walked, default 10):
    tiles_until_ammo = (
        game.config.tiles_per_ammo
        - p.tiles_traveled % game.config.tiles_per_ammo
    )
    # value == 1  → next step earns ammo
    # value == 10 → just earned ammo

Manhattan distance to a ghost (fast, no wall awareness):
    dist = abs(g.x - p.x) + abs(g.y - p.y)
    nearest = min(alive_ghosts, key=lambda g: abs(g.x-p.x)+abs(g.y-p.y))

Ghosts in shooting range (adjacent 8 tiles):
    in_range = [
        g for g in alive_ghosts
        if abs(g.x - p.x) <= 1 and abs(g.y - p.y) <= 1
        and (g.x != p.x or g.y != p.y)
    ]
    # shoot actions only appear in get_legal_actions() when p.ammo > 0

Reactive defence:
    If a ghost steps onto the player's tile and p.ammo > 0, the ghost dies
    automatically inside game.step() — no action needed.
──────────────────────────────────────────────────────────────────────────────
"""

import math
import multiprocessing as mp
import random
import sys
import time
from collections import deque

import game as game_module
from typing import Dict, List, Optional

from game import Action, GameConfig, GameState, GhostBehavior, PacmanGame


# ──────────────────────────────────────────────────────────────────────────────
# Time-limited agent wrapper
# ──────────────────────────────────────────────────────────────────────────────

MOVE_TIME_LIMIT = 5.0

_timeout_counter: Dict[str, int] = {}


def _worker(agent_name: str, obs: Dict, state: GameState,
            config: GameConfig, kwargs: dict, q: mp.SimpleQueue) -> None:
    game = PacmanGame(config=config, render=False)
    game.set_state(state)
    fn = _AGENT_FNS[agent_name]
    try:
        q.put(fn(obs, game, **kwargs))
    except Exception:
        q.put(Action.STAY)


def _timed(agent_name: str, obs: Dict, game: PacmanGame, **kwargs) -> Action:
    q = mp.SimpleQueue()
    p = mp.Process(
        target=_worker,
        args=(agent_name, obs, game.get_state(), game.config, kwargs, q),
        daemon=True,
    )
    p.start()
    p.join(MOVE_TIME_LIMIT)
    if p.is_alive():
        p.kill()
        p.join()
        _timeout_counter[agent_name] = _timeout_counter.get(agent_name, 0) + 1
        return Action.STAY
    try:
        return q.get()
    except Exception:
        return Action.STAY


# ──────────────────────────────────────────────────────────────────────────────
# AGENT SETTINGS — tune these
# ──────────────────────────────────────────────────────────────────────────────
# "depth" means plies: one ply = one game.step() (player acts, ghosts react).
# The player has up to 12 legal actions, so a full tree has ~12^depth nodes.

MINIMAX_DEPTH      = 2     # plain minimax: every extra ply ≈ 12× slower
ALPHABETA_DEPTH    = 3     # pruning makes much deeper search affordable

MCTS_ITERATIONS    = 180   # selection → backprop cycles per move
MCTS_ROLLOUT_DEPTH = 10    # steps per rollout (simulation phase)
MCTS_C             = 1.41  # UCB1 exploration constant (C in Q + C*sqrt(ln(N)/n))


# ──────────────────────────────────────────────────────────────────────────────
# TODO 1 — Scoring / evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_state(game: PacmanGame) -> float:
    p = game.player
    if not p.alive:
        return -10000.0 + game.score

    alive_ghosts = [g for g in game.ghosts if g.alive]
    killed = len(game.ghosts) - len(alive_ghosts)
    score = 1000.0 + killed * 250.0 + game.score * 2.0

    if not alive_ghosts:
        return score + 2500.0

    distances = [abs(g.x - p.x) + abs(g.y - p.y) for g in alive_ghosts]
    nearest = min(distances)
    weighted_danger = 0.0
    threat_weights = {
        GhostBehavior.RANDOM: 0.65,
        GhostBehavior.FOLLOW: 1.0,
        GhostBehavior.LOOKAHEAD: 1.3,
        GhostBehavior.DFS: 1.05,
    }
    for ghost, distance in zip(alive_ghosts, distances):
        weight = threat_weights.get(ghost.behavior, 1.0)
        weighted_danger += weight / (1.0 + distance)

    if p.ammo > 0:
        score += max(0.0, 9.0 - nearest) * 4.0
        score += 65.0 * sum(
            1 for g in alive_ghosts
            if max(abs(g.x - p.x), abs(g.y - p.y)) == 1
        )
    else:
        score += min(nearest, 12) * 5.0
        if nearest <= 1:
            score -= 180.0
        elif nearest == 2:
            score -= 55.0
    score -= weighted_danger * 36.0

    tiles_per_ammo = max(1, game.config.tiles_per_ammo)
    tiles_until_ammo = tiles_per_ammo - (p.tiles_traveled % tiles_per_ammo)
    score += min(p.ammo, 4) * 24.0
    score += (tiles_per_ammo - tiles_until_ammo) * 1.4

    free_neighbors = 0
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = p.x + dx, p.y + dy
        if (0 <= nx < game.config.grid_width
                and 0 <= ny < game.config.grid_height
                and (nx, ny) not in game.walls):
            free_neighbors += 1
    score += free_neighbors * 7.0

    frontier = deque([(p.x, p.y, 0)])
    visited = {(p.x, p.y)}
    local_space = 0
    while frontier:
        x, y, depth = frontier.popleft()
        local_space += 1
        if depth == 3:
            continue
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxt = (x + dx, y + dy)
            if (0 <= nxt[0] < game.config.grid_width
                    and 0 <= nxt[1] < game.config.grid_height
                    and nxt not in game.walls and nxt not in visited):
                visited.add(nxt)
                frontier.append((nxt[0], nxt[1], depth + 1))
    score += min(local_space, 25) * 1.5
    score += game.step_count * 0.15
    return score


def _apply_player_action(game: PacmanGame, action: Action) -> None:
    p = game.player
    reward = 0.0
    move_delta = game_module._MOVE_DELTA
    shoot_delta = game_module._SHOOT_DELTA

    if action in move_delta:
        dx, dy = move_delta[action]
        nx, ny = p.x + dx, p.y + dy
        if (0 <= nx < game.config.grid_width and 0 <= ny < game.config.grid_height
                and (nx, ny) not in game.walls):
            p.x, p.y = nx, ny
            p.tiles_traveled += 1
            if p.tiles_traveled % game.config.tiles_per_ammo == 0:
                p.ammo += 1
    elif action in shoot_delta and p.ammo > 0:
        dx, dy = shoot_delta[action]
        tx, ty = p.x + dx, p.y + dy
        hit = False
        for ghost in game.ghosts:
            if ghost.alive and (ghost.x, ghost.y) == (tx, ty):
                ghost.alive = False
                reward += 10.0
                hit = True
        p.ammo -= 1
        if hit:
            p.x, p.y = tx, ty
            p.tiles_traveled += 1
            if p.tiles_traveled % game.config.tiles_per_ammo == 0:
                p.ammo += 1
    game.score += reward


def _legal_ghost_tiles(game: PacmanGame, ghost) -> List[tuple]:
    options = [(ghost.x, ghost.y)]
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = ghost.x + dx, ghost.y + dy
        if (0 <= nx < game.config.grid_width and 0 <= ny < game.config.grid_height
                and (nx, ny) not in game.walls):
            options.append((nx, ny))
    return options


def _finish_ghost_phase(game: PacmanGame, proposals: Dict[int, tuple]) -> None:
    p = game.player
    alive = [i for i, ghost in enumerate(game.ghosts) if ghost.alive]
    order = sorted(alive, key=lambda i: (
        abs(game.ghosts[i].x - p.x) + abs(game.ghosts[i].y - p.y), i
    ))
    occupied = set()
    for index in order:
        ghost = game.ghosts[index]
        destination = proposals.get(index, (ghost.x, ghost.y))
        if destination not in occupied:
            ghost.x, ghost.y = destination
        occupied.add((ghost.x, ghost.y))

    reward = 0.0
    for ghost in game.ghosts:
        if ghost.alive and ghost.x == p.x and ghost.y == p.y:
            if p.ammo > 0:
                ghost.alive = False
                p.ammo -= 1
                reward += 10.0
            else:
                p.alive = False
    game.step_count += 1
    if not p.alive:
        reward -= 100.0
    game.score += reward
    if not p.alive or game.step_count >= game.config.max_steps:
        game.done = True


def _ghost_response_states(game: PacmanGame) -> List[PacmanGame]:
    alive_indices = [i for i, ghost in enumerate(game.ghosts) if ghost.alive]
    baseline = {}
    for index in alive_indices:
        ghost = game.ghosts[index]
        if ghost.behavior == GhostBehavior.RANDOM:
            options = list(ghost._neighbors(
                ghost.x, ghost.y, game.config.grid_width,
                game.config.grid_height, game.walls
            ))
            seed = (game.step_count + 1) * 1000003 + index * 9176 + ghost.x * 131 + ghost.y * 17
            baseline[index] = random.Random(seed).choice(options) if options else (ghost.x, ghost.y)
        elif ghost.behavior == GhostBehavior.DFS:
            baseline[index] = ghost._bfs_first_step(
                game.player.x, game.player.y, game.config, game.walls
            )
        else:
            baseline[index] = ghost.compute_move(
                game.player.x, game.player.y, game.config, game.walls
            )

    if not alive_indices:
        child = game.clone()
        _finish_ghost_phase(child, {})
        return [child]

    closest = sorted(
        alive_indices,
        key=lambda i: (
            abs(game.ghosts[i].x - game.player.x)
            + abs(game.ghosts[i].y - game.player.y), i
        ),
    )[:2]
    responses = []
    seen = set()
    for index in closest:
        for destination in _legal_ghost_tiles(game, game.ghosts[index]):
            proposals = dict(baseline)
            proposals[index] = destination
            child = game.clone()
            _finish_ghost_phase(child, proposals)
            key = tuple(
                (g.x, g.y, g.alive) for g in child.ghosts
            ) + (child.player.ammo, child.player.alive, child.step_count)
            if key not in seen:
                seen.add(key)
                responses.append(child)
    if not responses:
        child = game.clone()
        _finish_ghost_phase(child, baseline)
        responses.append(child)
    return responses


def _ordered_actions(game: PacmanGame) -> List[Action]:
    def priority(action):
        if action in game_module._SHOOT_DELTA:
            dx, dy = game_module._SHOOT_DELTA[action]
            target = (game.player.x + dx, game.player.y + dy)
            hit = any(g.alive and (g.x, g.y) == target for g in game.ghosts)
            return (0 if hit else 3, action.value)
        if action in game_module._MOVE_DELTA:
            return (1, action.value)
        return (4, action.value)
    return sorted(game.get_legal_actions(), key=priority)


# ──────────────────────────────────────────────────────────────────────────────
# Agent 1 — Minimax
# ──────────────────────────────────────────────────────────────────────────────

_SEARCH_DEADLINE = None


class _SearchTimeout(Exception):
    pass


def _check_search_time() -> None:
    if _SEARCH_DEADLINE is not None and time.perf_counter() >= _SEARCH_DEADLINE:
        raise _SearchTimeout


def minimax_agent(obs: Dict, game: PacmanGame,
                  depth: int = MINIMAX_DEPTH) -> Action:
    global _SEARCH_DEADLINE
    _SEARCH_DEADLINE = time.perf_counter() + 4.4
    best_action = Action.STAY
    best_value = -math.inf
    try:
        for action in _ordered_actions(game):
            _check_search_time()
            child = game.clone()
            _apply_player_action(child, action)
            value = _minimax(child, max(0, depth), maximising=False)
            if value > best_value:
                best_value, best_action = value, action
    except _SearchTimeout:
        pass
    finally:
        _SEARCH_DEADLINE = None
    return best_action


def _minimax(game: PacmanGame, depth: int, maximising: bool) -> float:
    _check_search_time()
    if depth <= 0 or game.done or not game.player.alive:
        return evaluate_state(game)

    if maximising:
        best = -math.inf
        for action in _ordered_actions(game):
            _check_search_time()
            child = game.clone()
            _apply_player_action(child, action)
            value = _minimax(child, depth, maximising=False)
            best = max(best, value)
        return best if best != -math.inf else evaluate_state(game)

    worst = math.inf
    responses = sorted(_ghost_response_states(game), key=evaluate_state)
    for response in responses:
        _check_search_time()
        value = _minimax(response, depth - 1, maximising=True)
        worst = min(worst, value)
    return worst if worst != math.inf else evaluate_state(game)


def alphabeta_agent(obs: Dict, game: PacmanGame,
                    depth: int = ALPHABETA_DEPTH) -> Action:
    global _SEARCH_DEADLINE
    _SEARCH_DEADLINE = time.perf_counter() + 4.5
    best_action = Action.STAY
    try:
        for current_depth in range(1, max(1, depth) + 1):
            iteration_action = best_action
            iteration_value = -math.inf
            alpha = -math.inf
            beta = math.inf
            try:
                for action in _ordered_actions(game):
                    _check_search_time()
                    child = game.clone()
                    _apply_player_action(child, action)
                    value = _alphabeta(
                        child, current_depth, alpha, beta, maximising=False
                    )
                    if value > iteration_value:
                        iteration_value = value
                        iteration_action = action
                    alpha = max(alpha, iteration_value)
                best_action = iteration_action
            except _SearchTimeout:
                break
    finally:
        _SEARCH_DEADLINE = None
    return best_action


def _alphabeta(game: PacmanGame, depth: int, alpha: float, beta: float,
               maximising: bool) -> float:
    _check_search_time()
    if depth <= 0 or game.done or not game.player.alive:
        return evaluate_state(game)

    if maximising:
        value = -math.inf
        for action in _ordered_actions(game):
            _check_search_time()
            child = game.clone()
            _apply_player_action(child, action)
            value = max(value, _alphabeta(
                child, depth, alpha, beta, maximising=False
            ))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value if value != -math.inf else evaluate_state(game)

    value = math.inf
    responses = sorted(_ghost_response_states(game), key=evaluate_state)
    for response in responses:
        _check_search_time()
        value = min(value, _alphabeta(
            response, depth - 1, alpha, beta, maximising=True
        ))
        beta = min(beta, value)
        if beta <= alpha:
            break
    return value if value != math.inf else evaluate_state(game)


# ──────────────────────────────────────────────────────────────────────────────
# TODO 2 — UCB1 score formula
# ──────────────────────────────────────────────────────────────────────────────

def ucb1_score(q: float, n: int, N: int, c: float) -> float:
    """
    Slide 42 — UCB1 formula:

        UCB1(node) = U(n)/N(n)  +  C × sqrt( log N(PARENT(n)) / N(n) )

    Translated to this function's parameters:
        q = U(n)/N(n)  — mean rollout value of this node  (total_value / visits)
        n = N(n)       — visit count of this node
        N = N(PARENT)  — visit count of the parent node
        c = C          — exploration constant

    Returns:
        float UCB1 score (higher = prefer this node).

    Notes:
      • n == 0 means the node is unvisited — return +infinity so it is always
        chosen before any visited sibling (forces every child to be tried once).
      • Called by MCTSNode.best_child() for every child in the selection step.

    TODO: replace the NotImplementedError with your implementation.
    """
    if n <= 0:
        return math.inf
    if N <= 1:
        exploration = 0.0
    else:
        exploration = c * math.sqrt(math.log(N) / n)
    return q + exploration


# ──────────────────────────────────────────────────────────────────────────────
# Agent 3 — Monte Carlo Tree Search with UCB1 (UCT)
# ──────────────────────────────────────────────────────────────────────────────

class MCTSNode:
    def __init__(self, state_snapshot: GameState,
                 parent: Optional['MCTSNode'] = None,
                 action_taken: Optional[Action] = None,
                 untried_actions: Optional[List[Action]] = None):
        self.state_snapshot = state_snapshot
        self.parent = parent
        self.action_taken = action_taken
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = list(untried_actions) if untried_actions is not None else []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c: float = 1.41) -> 'MCTSNode':
        if not self.children:
            raise ValueError("A node without children has no best child.")
        return max(
            self.children,
            key=lambda child: ucb1_score(
                child.mean_value(), child.visits, self.visits, c
            ),
        )

    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0


def _rollout_action(game: PacmanGame) -> Action:
    actions = game.get_legal_actions()
    p = game.player
    adjacent = []
    if p.ammo > 0:
        for action, (dx, dy) in game_module._SHOOT_DELTA.items():
            if action in actions:
                target = (p.x + dx, p.y + dy)
                for ghost in game.ghosts:
                    if ghost.alive and (ghost.x, ghost.y) == target:
                        adjacent.append(action)
    if adjacent:
        return random.choice(adjacent)

    moves = [a for a in actions if a in game_module._MOVE_DELTA]
    if not moves:
        return Action.STAY
    alive = [g for g in game.ghosts if g.alive]
    if not alive:
        return random.choice(moves)
    nearest = min(alive, key=lambda g: abs(g.x - p.x) + abs(g.y - p.y))
    if p.ammo <= 0:
        ranked = sorted(
            moves,
            key=lambda a: abs(
                (p.x + game_module._MOVE_DELTA[a][0]) - nearest.x
            ) + abs((p.y + game_module._MOVE_DELTA[a][1]) - nearest.y),
            reverse=True,
        )
        return random.choice(ranked[:min(2, len(ranked))])
    ranked = sorted(
        moves,
        key=lambda a: abs(
            (p.x + game_module._MOVE_DELTA[a][0]) - nearest.x
        ) + abs((p.y + game_module._MOVE_DELTA[a][1]) - nearest.y),
    )
    return random.choice(ranked[:min(2, len(ranked))])


def mcts_agent(obs: Dict, game: PacmanGame,
               iterations: int = MCTS_ITERATIONS,
               rollout_depth: int = MCTS_ROLLOUT_DEPTH,
               c: float = MCTS_C) -> Action:
    deadline = time.perf_counter() + 4.45
    root_actions = game.get_legal_actions()
    if not root_actions:
        return Action.STAY
    root = MCTSNode(game.get_state(), untried_actions=root_actions)
    simulation = game.clone()
    completed = 0
    for _ in range(max(1, iterations)):
        if time.perf_counter() >= deadline:
            break
        node = _mcts_select(root, simulation, c)
        if not simulation.done and node.untried_actions and time.perf_counter() < deadline:
            node = _mcts_expand(node, simulation)
        value = _mcts_simulate(simulation, rollout_depth, deadline=deadline)
        _mcts_backprop(node, value)
        completed += 1
    if not root.children:
        return root_actions[0]
    return max(root.children, key=lambda child: child.visits).action_taken or root_actions[0]


def _mcts_select(root: MCTSNode, game: PacmanGame, c: float = MCTS_C) -> MCTSNode:
    node = root
    game.set_state(node.state_snapshot)
    while node.is_fully_expanded() and node.children and not game.done:
        node = node.best_child(c)
        game.set_state(node.state_snapshot)
    return node


def _mcts_expand(node: MCTSNode, game: PacmanGame) -> MCTSNode:
    if not node.untried_actions:
        return node
    action = random.choice(node.untried_actions)
    node.untried_actions.remove(action)
    game.step(action)
    next_actions = [] if game.done else game.get_legal_actions()
    child = MCTSNode(
        game.get_state(), parent=node, action_taken=action,
        untried_actions=next_actions,
    )
    node.children.append(child)
    return child


def _mcts_simulate(game: PacmanGame, depth: int,
                   deadline: Optional[float] = None) -> float:
    for _ in range(max(0, depth)):
        if game.done:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        game.step(_rollout_action(game))
    return evaluate_state(game) / 1000.0


def _mcts_backprop(node: MCTSNode, value: float):
    current = node
    while current is not None:
        current.visits += 1
        current.total_value += value
        current = current.parent


# ──────────────────────────────────────────────────────────────────────────────
# Test harness
# ──────────────────────────────────────────────────────────────────────────────

_AGENT_FNS = {
    'minimax':   minimax_agent,
    'alphabeta': alphabeta_agent,
    'mcts':      mcts_agent,
}

_AGENTS = {
    'minimax':   lambda obs, g: _timed('minimax',   obs, g, depth=MINIMAX_DEPTH),
    'alphabeta': lambda obs, g: _timed('alphabeta', obs, g, depth=ALPHABETA_DEPTH),
    'mcts':      lambda obs, g: _timed('mcts',      obs, g,
                                       iterations=MCTS_ITERATIONS,
                                       rollout_depth=MCTS_ROLLOUT_DEPTH,
                                       c=MCTS_C),
}

# which setting `--depth N` overrides for each agent
_DEPTH_SETTING = {
    'minimax':   'MINIMAX_DEPTH',
    'alphabeta': 'ALPHABETA_DEPTH',
    'mcts':      'MCTS_ROLLOUT_DEPTH',
}


def evaluate_agent(agent_name: str,
                   num_episodes: int             = 5,
                   config: Optional[GameConfig]  = None) -> Dict:
    config = config or GameConfig(num_ghosts=5, max_steps=500)
    game   = PacmanGame(config=config, render=False)
    fn     = _AGENTS[agent_name]

    _timeout_counter[agent_name] = 0
    scores, lengths, kills, tiles = [], [], [], []
    for ep in range(num_episodes):
        before = _timeout_counter.get(agent_name, 0)
        obs    = game.reset()
        done   = False
        while not done:
            action = fn(obs, game)
            obs, _, done, info = game.step(action)
        ep_timeouts = _timeout_counter.get(agent_name, 0) - before
        scores.append(info['score'])
        lengths.append(info['step'])
        kills.append(config.num_ghosts - info['ghosts_alive'])
        tiles.append(info['tiles_traveled'])
        print(f"  episode {ep+1}: score={info['score']:.1f}  "
              f"steps={info['step']}  kills={config.num_ghosts - info['ghosts_alive']}  "
              f"tiles={info['tiles_traveled']}  timeouts={ep_timeouts}")

    total_timeouts = _timeout_counter.get(agent_name, 0)
    return {
        'agent':          agent_name,
        'avg_score':      sum(scores)  / len(scores),
        'avg_length':     sum(lengths) / len(lengths),
        'avg_kills':      sum(kills)   / len(kills),
        'avg_tiles':      sum(tiles)   / len(tiles),
        'total_timeouts': total_timeouts,
    }


def run_agent_in_window(agent_name: str,
                        config: Optional[GameConfig] = None) -> Dict:
    config = config or GameConfig(num_ghosts=5)
    game   = PacmanGame(config=config, render=True)
    fn     = _AGENTS[agent_name]
    _timeout_counter[agent_name] = 0

    def _agent_fn(obs):
        before = _timeout_counter.get(agent_name, 0)
        action = fn(obs, game)
        if _timeout_counter.get(agent_name, 0) > before:
            game.agent_timeouts += 1
        return action

    game.run(agent_fn=_agent_fn)
    kills = sum(1 for g in game.ghosts if not g.alive)
    return {'kills': kills, 'tiles_traveled': game.player.tiles_traveled,
            'timeouts': game.agent_timeouts}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _ask(prompt: str, options: list) -> int:
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"{prompt}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  Enter a number between 1 and {len(options)}.")


if __name__ == '__main__':
    headless      = '--headless'      in sys.argv
    no_time_limit = '--no-time-limit' in sys.argv
    assess        = '--assess'        in sys.argv

    if no_time_limit:
        _AGENTS['minimax']   = lambda obs, g: minimax_agent(obs, g, depth=MINIMAX_DEPTH)
        _AGENTS['alphabeta'] = lambda obs, g: alphabeta_agent(obs, g, depth=ALPHABETA_DEPTH)
        _AGENTS['mcts']      = lambda obs, g: mcts_agent(obs, g,
                                                         iterations=MCTS_ITERATIONS,
                                                         rollout_depth=MCTS_ROLLOUT_DEPTH,
                                                         c=MCTS_C)

    if assess:
        try:
            raw = input("Episodes per agent [5]: ").strip()
        except EOFError:
            raw = ''
        episodes = int(raw) if raw.isdigit() and int(raw) > 0 else 5
        all_agents = ['minimax', 'alphabeta', 'mcts']
        results = []
        for a in all_agents:
            print(f"\n--- {a} ({episodes} episodes) ---")
            results.append(evaluate_agent(a, num_episodes=episodes))
        col = 14
        print(f"\n{'agent':<{col}} {'avg_kills':>10} {'avg_tiles':>10} {'avg_score':>10} {'avg_steps':>10} {'timeouts':>10}")
        print('-' * (col + 54))
        for r in results:
            print(f"{r['agent']:<{col}} {r['avg_kills']:>10.2f} {r['avg_tiles']:>10.1f}"
                  f" {r['avg_score']:>10.1f} {r['avg_length']:>10.1f} {r['total_timeouts']:>10}")
        sys.exit(0)

    print("=== Pacman AI ===\n")

    agent_options = [
        'Human (you play)',
        f'Minimax            (depth={MINIMAX_DEPTH})',
        f'Alpha-Beta         (depth={ALPHABETA_DEPTH})',
        f'MCTS               (iter={MCTS_ITERATIONS}, rollout={MCTS_ROLLOUT_DEPTH})',
    ]
    agent_keys = ['human', 'minimax', 'alphabeta', 'mcts']
    agent = agent_keys[_ask("Select agent", agent_options)]

    if agent == 'human':
        print("\nControls: arrows/WASD move  Shift+dir shoot  R reset  Q quit")
        game = PacmanGame(render=True)
        game.run()
        kills = sum(1 for g in game.ghosts if not g.alive)
        print(f"\nkills: {kills}  tiles traveled: {game.player.tiles_traveled}")

    elif headless:
        raw = input("\nEpisodes [5]: ").strip()
        episodes = int(raw) if raw.isdigit() and int(raw) > 0 else 5
        print()
        stats = evaluate_agent(agent, num_episodes=episodes)
        print()
        for k, v in stats.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    else:
        stats = run_agent_in_window(agent)
        print(f"\nkills: {stats['kills']}  tiles traveled: {stats['tiles_traveled']}  timeouts: {stats['timeouts']}")
