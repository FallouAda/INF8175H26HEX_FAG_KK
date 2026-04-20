# =============================================================================
# INF8175 - Projet Hex
# Agent: Minimax avec Alpha-Beta Pruning et Iterative Deepening
# ELHADJI FALLOU ADAMA GUEYE - 2126169
# KYLIAN KOUASSI - 2194817
# Stratégie:
#   - Minimax avec élagage alpha-beta pour explorer l'arbre de jeu
#   - Iterative deepening: on cherche à profondeur 1, puis 2, puis 3...
#     jusqu'à ce que le temps soit presque écoulé
#   - Heuristique: différence de longueur de chemin le plus court
#     (Dijkstra) entre l'adversaire et nous-mêmes
# =============================================================================

import time
import heapq

from player_hex import PlayerHex
from game_state_hex import GameStateHex
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction


class MyPlayer(PlayerHex):
    """
    Agent Hex utilisant Minimax + Alpha-Beta + Iterative Deepening.

    Attributes:
        piece_type (str): "R" (rouge, relie haut-bas) ou "B" (bleu, relie gauche-droite)
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self._time_limit = None
        self._start_time = None

    def compute_action(self, current_state: GameStateHex,
                       remaining_time: float = 15 * 60, **kwargs) -> Action:
        TIME_PER_MOVE = 10.0
        SAFETY_BUFFER = 2.0

        available = min(TIME_PER_MOVE, remaining_time - SAFETY_BUFFER)
        available = max(available, 0.5)

        self._start_time = time.time()
        self._time_limit = self._start_time + available

        best_action = self._iterative_deepening(current_state)
        return best_action

    def _iterative_deepening(self, state: GameStateHex) -> Action:
        possible_actions = list(state.get_possible_stateless_actions())

        if len(possible_actions) == 1:
            return possible_actions[0]

        dim = state.get_rep().get_dimensions()[0]
        center = dim / 2.0
        possible_actions.sort(
            key=lambda a: abs(a.data["position"][0] - center)
                        + abs(a.data["position"][1] - center)
        )

        best_action = possible_actions[0]
        depth = 1

        while True:
            if time.time() >= self._time_limit:
                break

            try:
                action, _ = self._minimax_root(state, possible_actions, depth)
                best_action = action
                depth += 1

                if depth > 6:
                    break

            except _TimeOut:
                break

        return best_action

    def _minimax_root(self, state: GameStateHex, actions, depth: int):
        best_score = float('-inf')
        best_action = actions[0]
        alpha = float('-inf')
        beta = float('inf')

        for action in actions:
            self._check_time()
            next_state = state.apply_action(action)
            score = self._minimax(next_state, depth - 1,
                                  alpha, beta, is_maximizing=False)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)

        return best_action, best_score

    def _minimax(self, state: GameStateHex, depth: int,
                 alpha: float, beta: float, is_maximizing: bool) -> float:
        self._check_time()

        if state.is_done():
            scores = state.get_scores()
            my_id = self._get_my_player_id(state)
            return 10000.0 if scores.get(my_id, 0) == 1.0 else -10000.0

        if depth == 0:
            return self._heuristic(state)

        actions = list(state.get_possible_stateless_actions())
        dim = state.get_rep().get_dimensions()[0]
        center = dim / 2.0
        actions.sort(
            key=lambda a: abs(a.data["position"][0] - center)
                        + abs(a.data["position"][1] - center)
        )

        if is_maximizing:
            max_score = float('-inf')
            for action in actions:
                self._check_time()
                next_state = state.apply_action(action)
                score = self._minimax(next_state, depth - 1,
                                      alpha, beta, is_maximizing=False)
                max_score = max(max_score, score)
                alpha = max(alpha, max_score)
                if beta <= alpha:
                    break
            return max_score

        else:
            min_score = float('inf')
            for action in actions:
                self._check_time()
                next_state = state.apply_action(action)
                score = self._minimax(next_state, depth - 1,
                                      alpha, beta, is_maximizing=True)
                min_score = min(min_score, score)
                beta = min(beta, min_score)
                if beta <= alpha:
                    break
            return min_score

    def _heuristic(self, state: GameStateHex) -> float:
        my_dist = self._shortest_path(state, self.piece_type)
        opp_type = "B" if self.piece_type == "R" else "R"
        opp_dist = self._shortest_path(state, opp_type)

        if opp_dist == 0:
            return -10000.0
        if my_dist == 0:
            return 10000.0

        return float(opp_dist) - float(my_dist)

    def _shortest_path(self, state: GameStateHex, piece_type: str) -> float:
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]

        dist = [[float('inf')] * dim for _ in range(dim)]
        pq = []

        if piece_type == "R":
            for j in range(dim):
                cell = env.get((0, j))
                if cell is None:
                    cost = 1
                elif cell.piece_type == "R":
                    cost = 0
                else:
                    continue
                if cost < dist[0][j]:
                    dist[0][j] = cost
                    heapq.heappush(pq, (cost, (0, j)))
        else:
            for i in range(dim):
                cell = env.get((i, 0))
                if cell is None:
                    cost = 1
                elif cell.piece_type == "B":
                    cost = 0
                else:
                    continue
                if cost < dist[i][0]:
                    dist[i][0] = cost
                    heapq.heappush(pq, (cost, (i, 0)))

        while pq:
            d, (i, j) = heapq.heappop(pq)

            if d > dist[i][j]:
                continue

            if piece_type == "R" and i == dim - 1:
                return d
            if piece_type == "B" and j == dim - 1:
                return d

            for _, (ni, nj) in state.get_rep().get_neighbours(i, j).values():
                if ni < 0 or ni >= dim or nj < 0 or nj >= dim:
                    continue
                neighbor_cell = env.get((ni, nj))
                if neighbor_cell is not None and neighbor_cell.piece_type != piece_type:
                    continue

                move_cost = 0 if (neighbor_cell is not None) else 1
                new_dist = d + move_cost

                if new_dist < dist[ni][nj]:
                    dist[ni][nj] = new_dist
                    heapq.heappush(pq, (new_dist, (ni, nj)))

        return float('inf')

    def _get_my_player_id(self, state: GameStateHex) -> int:
        for player in state.players:
            if player.get_piece_type() == self.piece_type:
                return player.get_id()
        return -1

    def _check_time(self):
        if time.time() >= self._time_limit:
            raise _TimeOut()

class _TimeOut(Exception):
    pass