from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import kgp
from kgp import Board

# -----------------------------
# Configuration / constants
# -----------------------------

EXACT, LOWER, UPPER = 0, 1, 2
PVS_EPS = 1e-6


@dataclass(frozen=True)
class SearchConfig:
    """Search and evaluation configuration."""
    # Soft compute budget. The server can still stop us at any time via stop_evt.
    time_limit_s: float = 0.90

    max_depth: int = 22

    endgame_stones_threshold: int = 24
    endgame_max_depth: int = 60

    # Aspiration window half-width around previous iteration score (in eval units).
    aspiration_window: float = 250.0

    # Heuristic weights (stores dominate).
    w_store: float = 100.0
    w_pits: float = 1.0
    w_extra_turn: float = 3.0
    w_capture: float = 2.0

    use_transposition: bool = True


@dataclass
class TTEntry:
    """Transposition table entry."""
    depth: int
    value: float
    flag: int
    best_move: Optional[int]


# -----------------------------
# Engine (persistent across moves)
# -----------------------------

class AlphaBetaBoardEngine:
    """
    Stateful search engine that can be reused across move requests.

    It keeps transposition table, killer moves and history heuristic across moves
    within a game. It resets those structures when a new game start position is detected.
    """

    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg

        # Persistent structures (within a game)
        self._tt: Dict[Tuple[int, ...], TTEntry] = {}
        self._killers: Dict[int, List[int]] = {}
        self._history: List[int] = [0] * 64

        # Per-search state
        self.stop_evt = None
        self._start_time = 0.0
        self.nodes = 0

        # Used to detect a new game
        self._last_was_start = False

    # ---- Public API ----

    def begin_search(self, board: Board, stop_evt) -> None:
        """Initialize per-search state and reset persistent data if a new game is detected."""
        self.stop_evt = stop_evt
        self._start_time = time.time()
        self.nodes = 0

        self._ensure_history_len(len(board.south_pits))

        is_start = _is_start_position(board)
        if is_start and not self._last_was_start:
            self._tt.clear()
            self._killers.clear()
            for i in range(len(self._history)):
                self._history[i] = 0
        self._last_was_start = is_start

    def iterative_best_moves(self, board: Board) -> Iterable[int]:
        """
        Yield improving best moves as depth increases.

        This is a generator; the caller may emit intermediate best moves to comply
        with the KGP protocol expectations.
        """
        legal = board.legal_moves(kgp.SOUTH)
        if not legal:
            return

        depth_cap = self._depth_cap(board)
        best_move = legal[0]
        best_score = -math.inf
        prev_score: Optional[float] = None

        for depth in range(1, depth_cap + 1):
            if self._timed_out():
                break

            if prev_score is None:
                alpha, beta = -math.inf, math.inf
            else:
                w = self.cfg.aspiration_window
                alpha, beta = prev_score - w, prev_score + w

            move, score = self._search_root(board, depth, alpha, beta)

            # Aspiration failure => re-search with full window.
            if (not self._timed_out()) and (prev_score is not None) and (score <= alpha or score >= beta):
                move, score = self._search_root(board, depth, -math.inf, math.inf)

            if self._timed_out():
                break

            prev_score = score
            if move in legal and (move != best_move or score > best_score):
                best_move, best_score = move, score
                yield best_move

    # ---- Root search ----

    def _search_root(self, board: Board, depth: int, alpha: float, beta: float) -> Tuple[int, float]:
        legal = board.legal_moves(kgp.SOUTH)
        best_move = legal[0]
        best_score = -math.inf

        for move in self._ordered_moves(board, ply=0):
            if self._timed_out():
                break

            child, repeat = board.sow(kgp.SOUTH, move, pure=True)
            if repeat:
                score = self._negamax(child, depth - 1, alpha, beta, ply=1)
            else:
                score = -self._negamax(self._swap_perspective(child), depth - 1, -beta, -alpha, ply=1)

            if score > best_score:
                best_score, best_move = score, move

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break

        return best_move, best_score

    # ---- Negamax with PVS ----

    def _negamax(self, board: Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        self.nodes += 1

        if self._timed_out():
            return self._evaluate(board)

        if depth <= 0 or board.is_final():
            return self._evaluate(board)

        key = self._tt_key(board)

        # TT probe (bounds)
        if self.cfg.use_transposition:
            entry = self._tt.get(key)
            if entry is not None and entry.depth >= depth:
                if entry.flag == EXACT:
                    return entry.value
                if entry.flag == LOWER:
                    alpha = max(alpha, entry.value)
                elif entry.flag == UPPER:
                    beta = min(beta, entry.value)
                if alpha >= beta:
                    return entry.value

        alpha0 = alpha
        best_val = -math.inf
        best_move: Optional[int] = None

        moves = self._ordered_moves(board, ply=ply)

        for i, move in enumerate(moves):
            if self._timed_out():
                break

            child, repeat = board.sow(kgp.SOUTH, move, pure=True)

            if repeat:
                val = self._pvs_same_side(child, depth, alpha, beta, ply, is_first=(i == 0))
            else:
                swapped = self._swap_perspective(child)
                val = -self._pvs_same_side(swapped, depth, -beta, -alpha, ply, is_first=(i == 0))

            if val > best_val:
                best_val, best_move = val, move

            alpha = max(alpha, val)
            if alpha >= beta:
                self._note_cutoff(move, depth, ply)
                break

        # Store TT entry
        if self.cfg.use_transposition and best_move is not None:
            if best_val <= alpha0:
                flag = UPPER
            elif best_val >= beta:
                flag = LOWER
            else:
                flag = EXACT
            self._tt[key] = TTEntry(depth=depth, value=best_val, flag=flag, best_move=best_move)

        return best_val

    def _pvs_same_side(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        *,
        is_first: bool,
    ) -> float:
        """PVS for a position where 'to-move is SOUTH' invariant already holds."""
        if is_first:
            return self._negamax(board, depth - 1, alpha, beta, ply=ply + 1)

        # Null window first, then widen if it improves alpha.
        val = self._negamax(board, depth - 1, alpha, alpha + PVS_EPS, ply=ply + 1)
        if (val > alpha) and (val < beta) and (not self._timed_out()):
            val = self._negamax(board, depth - 1, alpha, beta, ply=ply + 1)
        return val

    # -----------------------------
    # Move ordering
    # -----------------------------

    def _ordered_moves(self, board: Board, ply: int) -> List[int]:
        moves = board.legal_moves(kgp.SOUTH)
        if not moves:
            return []

        ordered: List[int] = []

        # 1) TT best move
        if self.cfg.use_transposition:
            entry = self._tt.get(self._tt_key(board))
            if entry and (entry.best_move in moves):
                ordered.append(entry.best_move)

        # 2) Killer moves
        for km in self._killers.get(ply, []):
            if km in moves and km not in ordered:
                ordered.append(km)

        # 3) Score the rest
        rest = [m for m in moves if m not in ordered]
        scored: List[Tuple[Tuple[int, int, int, int, float], int]] = []

        n = len(board.south_pits)
        for m in rest:
            child, repeat = board.sow(kgp.SOUTH, m, pure=True)

            store_delta = int(child.south - board.south)
            stones = int(board.south_pits[m])

            # Store passes from sowing laps; capture gain = store increase beyond those passes.
            passes = _store_passes(stones=stones, pit_index=m, n=n)
            capture_gain = max(0, store_delta - passes)

            static_eval = self._evaluate(child)
            hist = self._history[m] if m < len(self._history) else 0

            score_key = (
                1 if repeat else 0,
                capture_gain,
                store_delta,
                hist,
                static_eval,
            )
            scored.append((score_key, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        ordered.extend([m for _, m in scored])
        return ordered

    def _note_cutoff(self, move: int, depth: int, ply: int) -> None:
        killers = self._killers.get(ply, [])
        if move not in killers:
            killers = [move] + killers
            self._killers[ply] = killers[:2]
        if move < len(self._history):
            self._history[move] += depth * depth

    # -----------------------------
    # Evaluation
    # -----------------------------

    def _evaluate(self, board: Board) -> float:
        """Static evaluation from SOUTH's perspective (positive is good for SOUTH)."""
        n = len(board.south_pits)

        store_diff = float(board.south - board.north)
        pit_diff = float(sum(board.south_pits) - sum(board.north_pits))

        # Potential extra turns: if a move ends in SOUTH store.
        mod = 2 * n + 1
        extra_self = sum(
            1 for idx, stones in enumerate(board.south_pits)
            if stones > 0 and (stones % mod) == (n - idx)
        )
        extra_opp = sum(
            1 for idx, stones in enumerate(board.north_pits)
            if stones > 0 and (stones % mod) == (n - idx)
        )
        extra_term = float(extra_self - extra_opp)

        capture_term = float(_capture_potential_fast(board))

        return (
            self.cfg.w_store * store_diff
            + self.cfg.w_pits * pit_diff
            + self.cfg.w_extra_turn * extra_term
            + self.cfg.w_capture * capture_term
        )

    # -----------------------------
    # Budget / misc helpers
    # -----------------------------

    def _timed_out(self) -> bool:
        if self.stop_evt is not None and self.stop_evt.is_set():
            return True
        return (time.time() - self._start_time) >= self.cfg.time_limit_s

    def _depth_cap(self, board: Board) -> int:
        stones_left = sum(board.south_pits) + sum(board.north_pits)
        if stones_left <= self.cfg.endgame_stones_threshold:
            return max(self.cfg.max_depth, self.cfg.endgame_max_depth)
        return self.cfg.max_depth

    def _ensure_history_len(self, n_pits: int) -> None:
        needed = n_pits + 1
        if needed > len(self._history):
            self._history.extend([0] * (needed - len(self._history)))

    @staticmethod
    def _swap_perspective(board: Board) -> Board:
        """Swap NORTH/SOUTH to maintain the 'SOUTH to move' invariant in negamax."""
        return Board(
            south=board.north,
            north=board.south,
            south_pits=list(board.north_pits),
            north_pits=list(board.south_pits),
        )

    @staticmethod
    def _tt_key(board: Board) -> Tuple[int, ...]:
        """Compact immutable key for transposition table."""
        return (
            *board.south_pits,
            -1,  # separator
            *board.north_pits,
            -2,  # separator
            int(board.south),
            int(board.north),
        )

# -----------------------------
# Utility heuristics
# -----------------------------

def _store_passes(stones: int, pit_index: int, n: int) -> int:
    """
    Exact number of times a sow from SOUTH pit 'pit_index' drops a stone into SOUTH store,
    assuming the standard Kalah track length (2*n + 1), including SOUTH store.
    """
    dist_to_store = n - pit_index
    if stones < dist_to_store:
        return 0
    cycle = 2 * n + 1
    return 1 + (stones - dist_to_store) // cycle


def _capture_potential_fast(board: Board) -> int:
    """
    Fast capture-bias heuristic (not rules-exact).

    Sums opponent stones opposite our empty pits that are plausibly reachable in one move.
    """
    n = len(board.south_pits)
    south = board.south_pits
    north = board.north_pits

    moves = [i for i, s in enumerate(south) if s > 0]
    if not moves:
        return 0

    value = 0
    for i in range(n):
        if south[i] != 0:
            continue

        opp = (n - 1) - i
        if north[opp] <= 0:
            continue

        # Simple reachability: last stone lands on i from some j without wrap-around.
        if any(south[j] == (i - j) for j in moves if i - j > 0):
            value += north[opp]

    return value


def _is_start_position(board: Board) -> bool:
    """Detect a new game start position for generic (m, n)."""
    if int(board.south) != 0 or int(board.north) != 0:
        return False
    if not board.south_pits or not board.north_pits:
        return False
    if tuple(board.south_pits) != tuple(board.north_pits):
        return False
    first = board.south_pits[0]
    return all(x == first for x in board.south_pits)


# -----------------------------
# Global engine (persist across states)
# -----------------------------

_ENGINE = AlphaBetaBoardEngine(SearchConfig())


# -----------------------------
# KGP agent entrypoint
# -----------------------------

def agent(board: Board, stop_evt):
    """
    Generator required by patched kgp.py: agent(board, stop_evt).
    Yields 0-based moves. kgp.py transmits move+1 on the wire.
    """
    legal = board.legal_moves(kgp.SOUTH)
    if not legal:
        return

    # Provide a fast first move before not returning anything
    yield legal[0]

    if stop_evt.is_set():
        return

    # Search for improvements.
    _ENGINE.begin_search(board, stop_evt)
    last = legal[0]

    for best in _ENGINE.iterative_best_moves(board):
        if stop_evt.is_set():
            return
        if best in legal and best != last:
            yield best
            last = best


def main():
    kgp.connect(
        agent,
        host="kalah.kwarc.info",
        name="Bigconda",
        authors=["Dhracyr"],
        # token="PUT_YOUR_REAL_TOKEN_HERE",
        debug=False,
    )


if __name__ == "__main__":
    main()
