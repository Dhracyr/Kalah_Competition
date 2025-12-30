
from __future__ import annotations

import math
import time

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from kgp import Board


@dataclass(frozen=True)
class KalahState:
    """
    Canonical Kalah board state for search.

    Layout convention:
      - pits[0] belongs to the side-to-move agent (SOUTH in KGP).
      - pits[1] belongs to the opponent (NORTH in KGP).
      - stores[0] is our store (SOUTH), stores[1] is opponent store (NORTH).
      - current_player: 0 or 1.
    """
    pits: Tuple[Tuple[int, ...], Tuple[int, ...]]
    stores: Tuple[int, int]
    current_player: int  # 0 or 1

    @property
    def n_pits(self) -> int:
        return len(self.pits[0])

    def is_terminal(self) -> bool:
        # Terminal when one side has no stones in pits
        return sum(self.pits[0]) == 0 or sum(self.pits[1]) == 0

    def finalize_if_terminal(self) -> "KalahState":
        """
        If terminal, sweep remaining stones into the respective stores.
        """
        if not self.is_terminal():
            return self

        p0_pits = list(self.pits[0])
        p1_pits = list(self.pits[1])
        s0, s1 = self.stores

        if sum(p0_pits) == 0:
            s1 += sum(p1_pits)
            p1_pits = [0] * self.n_pits
        elif sum(p1_pits) == 0:
            s0 += sum(p0_pits)
            p0_pits = [0] * self.n_pits

        return KalahState(
            pits=(tuple(p0_pits), tuple(p1_pits)),
            stores=(s0, s1),
            current_player=self.current_player,
        )

    def legal_moves(self) -> List[int]:
        """Returns pit indices (0..N-1) playable by current player."""
        cp = self.current_player
        return [i for i, stones in enumerate(self.pits[cp]) if stones > 0]

    def apply_move(self, pit_index: int) -> "KalahState":
        """
        Apply a move: sow stones from current_player's pit pit_index.
        Implements standard Kalah rules:
          - Sow counterclockwise including own store, excluding opponent store
          - Capture: if last stone lands in an empty own pit opposite non-empty opponent pit
          - Extra turn: if last stone lands in own store
          - Terminal sweep
        Returns the next state.
        """
        n = self.n_pits
        cp = self.current_player
        op = 1 - cp

        p0 = list(self.pits[0])
        p1 = list(self.pits[1])
        pits = [p0, p1]
        stores = [self.stores[0], self.stores[1]]

        stones = pits[cp][pit_index]
        if stones <= 0:
            raise ValueError(f"Illegal move: pit {pit_index} is empty.")
        pits[cp][pit_index] = 0

        pos_cp_side = True
        idx = pit_index  # first drop is into next position

        last_landed = ("pit", cp, pit_index)
        extra_turn = False


        # Gameloop
        while stones > 0:
            # advance to next position
            if pos_cp_side:
                # moving on current player's side pits then store
                if idx < n - 1:
                    idx += 1
                    pits[cp][idx] += 1
                    last_landed = ("pit", cp, idx)
                else:
                    # drop in current player's store
                    stores[cp] += 1
                    last_landed = ("store", cp, -1)
                    # after store, switch to opponent pits
                    pos_cp_side = False
                    idx = -1
                stones -= 1
            else:
                # moving on opponent pits only (skip opponent store)
                if idx < n - 1:
                    idx += 1
                    pits[op][idx] += 1
                    last_landed = ("pit", op, idx)
                    stones -= 1
                else:
                    # after opponent last pit, go back to current player's pits
                    pos_cp_side = True
                    idx = -1

        # Extra turn if last landed in current player's store
        if last_landed[0] == "store" and last_landed[1] == cp:
            extra_turn = True

        # Capture
        if last_landed[0] == "pit":
            landed_player, landed_idx = last_landed[1], last_landed[2]
            if landed_player == cp and pits[cp][landed_idx] == 1:
                opposite_idx = (n - 1) - landed_idx
                captured = pits[op][opposite_idx]
                if captured > 0:
                    pits[op][opposite_idx] = 0
                    pits[cp][landed_idx] = 0
                    stores[cp] += captured + 1

        next_player = cp if extra_turn else op

        next_state = KalahState(
            pits=(tuple(pits[0]), tuple(pits[1])),
            stores=(stores[0], stores[1]),
            current_player=next_player,
        ).finalize_if_terminal()

        return next_state

    # --- evaluation ---
    def stores_eval(self, perspective_player: int) -> int:
        """
        "Stores evaluation": store difference from perspective_player.
        Positive = good for perspective_player.
        """
        return self.stores[perspective_player] - self.stores[1 - perspective_player]

    def material_eval(self, perspective_player: int, pit_weight: float = 0.1) -> float:
        """Stores diff + small pit diff (pit_weight should be small)."""
        s = float(self.stores_eval(perspective_player))
        pits_self = sum(self.pits[perspective_player])
        pits_opp = sum(self.pits[1 - perspective_player])
        return s + pit_weight * float(pits_self - pits_opp)


# ----------------------------
# Alpha-beta search (negamax)
# ----------------------------

@dataclass
class SearchConfig:
    max_depth: int = 10
    use_iterative_deepening: bool = True
    time_limit_s: Optional[float] = None  # no limit
    use_transposition: bool = True
    pit_weight: float = 0.0  # 0.0 => pure Stores eval; >0 adds small pit term


@dataclass
class SearchResult:
    move: int
    score: float
    depth_reached: int


class AlphaBetaAgent:
    """Alpha-beta pruning with negamax formulation and move ordering."""

    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self._tt: Dict[Tuple, Tuple[int, float]] = {}  # key -> (depth, score)
        self._start_time: float = 0.0

    def choose_move(self, state: KalahState) -> SearchResult:
        self._start_time = time.time()
        self._tt.clear()

        legal = state.legal_moves()
        if not legal:
            # Defensive fallback (should not occur in valid Kalah positions)
            return SearchResult(move=0, score=-math.inf, depth_reached=0)

        best_move = legal[0]
        best_score = -math.inf
        depth_reached = 0

        depths = range(1, self.cfg.max_depth + 1) if self.cfg.use_iterative_deepening else [self.cfg.max_depth]

        for depth in depths:
            if self._timed_out():
                break

            move, score = self._search_root(state, depth)
            if self._timed_out():
                break

            best_move, best_score = move, score
            depth_reached = depth

        return SearchResult(move=best_move, score=best_score, depth_reached=depth_reached)

    def _search_root(self, state: KalahState, depth: int) -> Tuple[int, float]:
        cp = state.current_player
        best_move = state.legal_moves()[0]
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        for move in self._ordered_moves(state, player=cp):
            if self._timed_out():
                break

            child = state.apply_move(move)
            score = -self._negamax(child, depth - 1, -beta, -alpha, perspective=cp)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # prune

        return best_move, best_score

    def _negamax(self, state: KalahState, depth: int, alpha: float, beta: float, perspective: int) -> float:
        """
        Negamax with alpha-beta pruning.
        perspective = player for whom the returned value is good when positive.
        """
        if self._timed_out():
            return self._evaluate(state, perspective)

        if depth <= 0 or state.is_terminal():
            return self._evaluate(state, perspective)

        # Transposition table lookup
        if self.cfg.use_transposition:
            key = self._tt_key(state, perspective)
            hit = self._tt.get(key)
            if hit is not None:
                hit_depth, hit_score = hit
                if hit_depth >= depth:
                    return hit_score

        cp = state.current_player
        value = -math.inf

        for move in self._ordered_moves(state, player=cp):
            if self._timed_out():
                break

            child = state.apply_move(move)
            score = -self._negamax(child, depth - 1, -beta, -alpha, perspective=perspective)

            value = max(value, score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # prune

        if self.cfg.use_transposition:
            self._tt[self._tt_key(state, perspective)] = (depth, value)

        return value

    def _evaluate(self, state: KalahState, perspective: int) -> float:
        if self.cfg.pit_weight == 0.0:
            return float(state.stores_eval(perspective))
        return float(state.material_eval(perspective, pit_weight=self.cfg.pit_weight))

    def _ordered_moves(self, state: KalahState, player: int) -> List[int]:
        """
        Move ordering:
          1) Prefer extra-turn moves
          2) Prefer capture moves (approx by store delta)
          3) Prefer higher static evaluation after the move
        """
        moves = state.legal_moves()
        scored: List[Tuple[Tuple[int, int, float], int]] = []

        for m in moves:
            child = state.apply_move(m)
            extra_turn = 1 if child.current_player == player else 0
            capture = child.stores[player] - state.stores[player]
            static = self._evaluate(child, player)
            scored.append(((extra_turn, capture, static), m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _timed_out(self) -> bool:
        if self.cfg.time_limit_s is None:
            return False
        return (time.time() - self._start_time) >= self.cfg.time_limit_s

    @staticmethod
    def _tt_key(state: KalahState, perspective: int) -> Tuple:
        return (state.pits, state.stores, state.current_player, perspective)


# ----------------------------
# KGP integration
# ----------------------------

def board_to_state(board: Board) -> KalahState:
    """
    Convert kgp.Board -> internal KalahState.

    KGP provides `south_pits`, `north_pits`, `south`, `north`.
    The server ensures we act as SOUTH (our pits/stores are "south_*").
    """
    pits_self = tuple(int(x) for x in board.south_pits)
    pits_opp = tuple(int(x) for x in board.north_pits)
    store_self = int(board.south)
    store_opp = int(board.north)

    return KalahState(
        pits=(pits_self, pits_opp),
        stores=(store_self, store_opp),
        current_player=0,  # our turn when agent() is called
    )


# Build a single reusable searcher instance to reduce per-move overhead.
_CFG = SearchConfig(
    max_depth=10,
    use_iterative_deepening=True,
    time_limit_s=0.8,
    use_transposition=True,
    pit_weight=0.0,
)
_SEARCHER = AlphaBetaAgent(_CFG)


def agent(board: Board):
    """
    KGP agent entry point.

    Receives a kgp.Board and yields exactly one 0-based pit index.
    """
    state = board_to_state(board)
    result = _SEARCHER.choose_move(state)
    yield result.move
