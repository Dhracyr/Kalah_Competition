from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict

import math
import time


# ----------------------------
# Core Kalah representation
# ----------------------------

@dataclass(frozen=True)
class KalahState:
    """
    Canonical Kalah board state for search.

    Layout convention:
      - Each player has N pits (typically N=6) and one store.
      - pits[0] belongs to player 0, pits[1] belongs to player 1
      - store[0], store[1] are the stores.
      - current_player is 0 or 1.

    pits[player][i] is pit i (0..N-1) for that player.
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
        """
        Returns pit indices (0..N-1) playable by current player.
        """
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

        # We walk through positions in a fixed cycle:
        # current player's pits pit_index+1..N-1, then current player's store,
        # then opponent pits 0..N-1, then back to current player's pits 0..N-1, etc.
        # Opponent store is skipped.

        pos_cp_side = True
        idx = pit_index  # start from this pit, first drop is into next position

        last_landed = ("pit", cp, pit_index)  # will be overwritten
        extra_turn = False

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

        # Capture rule: last stone landed in own empty pit and opposite has stones
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

    # --- evaluation (Stores evaluation) ---
    def stores_eval(self, perspective_player: int) -> int:
        """
        "Stores evaluation": store difference from perspective_player.
        Positive = good for perspective_player.
        """
        return self.stores[perspective_player] - self.stores[1 - perspective_player]

    def material_eval(self, perspective_player: int, pit_weight: float = 0.1) -> float:
        """
        Slightly richer eval: stores diff + small pit diff.
        Keep pit_weight small; stores dominate.
        """
        s = float(self.stores_eval(perspective_player))
        pits_self = sum(self.pits[perspective_player])
        pits_opp = sum(self.pits[1 - perspective_player])
        return s + pit_weight * float(pits_self - pits_opp)


# ----------------------------
# Alpha-beta search (negamax)
# ----------------------------

@dataclass
class SearchConfig:
    max_depth: int = 8
    use_iterative_deepening: bool = True
    time_limit_s: Optional[float] = None  # None = no limit
    use_transposition: bool = True
    pit_weight: float = 0.0  # 0.0 => pure Stores eval; >0 adds small pit term


@dataclass
class SearchResult:
    move: int
    score: float
    depth_reached: int


class AlphaBetaAgent:
    """
    Alpha-beta pruning with negamax formulation and move ordering.
    """

    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self._tt: Dict[Tuple, Tuple[int, float]] = {}  # key -> (depth, score)
        self._start_time: float = 0.0

    def choose_move(self, state: KalahState) -> SearchResult:
        """
        Returns best move for state.current_player.
        """
        self._start_time = time.time()
        self._tt.clear()

        legal = state.legal_moves()
        if not legal:
            # Should not happen in valid Kalah, but be safe.
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
        """
        Root search: returns (best_move, best_score).
        """
        cp = state.current_player
        best_move = state.legal_moves()[0]
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        for move in self._ordered_moves(state, cp):
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
            # Conservative fallback: static eval at cutoff
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

        for move in self._ordered_moves(state, cp):
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
        """
        Stores evaluation, optionally with a small pit term.
        """
        if self.cfg.pit_weight == 0.0:
            return float(state.stores_eval(perspective))
        return float(state.material_eval(perspective, pit_weight=self.cfg.pit_weight))

    def _ordered_moves(self, state: KalahState, player: int) -> List[int]:
        """
        Move ordering:
          1) Prefer extra-turn moves
          2) Prefer capture moves
          3) Then prefer moves with better static eval after move

        This significantly improves alpha-beta pruning in Kalah.
        """
        moves = state.legal_moves()

        scored: List[Tuple[Tuple[int, int, float], int]] = []
        for m in moves:
            child = state.apply_move(m)

            extra_turn = 1 if child.current_player == player else 0
            capture = self._capture_delta(state, child, player)
            static = self._evaluate(child, player)

            # Sort descending by (extra_turn, capture, static)
            scored.append(((extra_turn, capture, static), m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    @staticmethod
    def _capture_delta(parent: KalahState, child: KalahState, player: int) -> int:
        """
        Approximate "capture amount" by store increase (beyond normal sowing).
        Not perfect, but strong enough for ordering.
        """
        return child.stores[player] - parent.stores[player]

    def _timed_out(self) -> bool:
        if self.cfg.time_limit_s is None:
            return False
        return (time.time() - self._start_time) >= self.cfg.time_limit_s

    @staticmethod
    def _tt_key(state: KalahState, perspective: int) -> Tuple:
        # perspective is included so the score meaning stays consistent.
        return (state.pits, state.stores, state.current_player, perspective)


# ----------------------------
# pykgp integration (adapter)
# ----------------------------

class PyKGPStateAdapter:
    """
    Adapter stub: convert pykgp's state object into our KalahState.

    You MUST adjust these two methods according to pykgp's actual state representation.
    The alpha-beta implementation above stays unchanged.
    """

    @staticmethod
    def from_pykgp(py_state) -> KalahState:
        """
        Expected to extract:
          - pits for both players as tuples of ints length N
          - stores for both players
          - current player index (0/1)
        """
        # ---- EXAMPLE ASSUMPTIONS (edit this) ----
        # Many Kalah libraries store:
        #   py_state.pits[0], py_state.pits[1]  (lists length N)
        #   py_state.stores[0], py_state.stores[1]
        #   py_state.player_to_move  (0/1)
        pits0 = tuple(py_state.pits[0])
        pits1 = tuple(py_state.pits[1])
        stores = (int(py_state.stores[0]), int(py_state.stores[1]))
        current = int(py_state.player_to_move)
        # ----------------------------------------
        return KalahState(pits=(pits0, pits1), stores=stores, current_player=current)

    @staticmethod
    def to_pykgp_move(move: int):
        """
        Convert our move (0..N-1 pit index) to whatever pykgp expects.
        Often it's just an int pit index; sometimes 1..N indexing.
        """
        return move  # edit if pykgp uses 1-based pits, etc.

from kgp import NORTH, SOUTH, Board

def board_to_state(board: Board):
    """
    Convert kgp.Board → internal KalahState
    Always from the perspective of the side to move.
    """
    # In KGP, the agent is always asked to move for "its" side.
    # By convention, that side is SOUTH.
    # (The server flips boards when needed.)
    current_player = 0  # we always treat ourselves as player 0

    pits_self = tuple(board.south_pits)
    pits_opp  = tuple(board.north_pits)

    store_self = board.south
    store_opp  = board.north

    return KalahState(
        pits=(pits_self, pits_opp),
        stores=(store_self, store_opp),
        current_player=current_player,
    )

# ----------------------------
# Example agent function
# ----------------------------

def agent(board: Board):
    """
    KGP agent generator.
    Receives a Board, yields 0-based pit indices.
    """
    state = board_to_state(board)

    cfg = SearchConfig(
        max_depth=10,
        use_iterative_deepening=True,
        time_limit_s=0.8,   # safe for practice server
        use_transposition=True,
        pit_weight=0.0,     # pure Stores evaluation (as requested)
    )

    searcher = AlphaBetaAgent(cfg)
    result = searcher.choose_move(state)

    # Yield exactly one move
    yield result.move

