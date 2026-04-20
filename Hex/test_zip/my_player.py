# =============================================================================
# INF8175 - Projet Hex
# Agent: Minimax avec Alpha-Beta Pruning et Iterative Deepening
#
# Stratégie:
#   - Minimax avec élagage alpha-beta pour explorer l'arbre de jeu
#   - Iterative deepening: on cherche à profondeur 1, puis 2, puis 3...
#     jusqu'à ce que le temps soit presque écoulé
#   - Heuristique: différence de longueur de chemin le plus court
#     (Dijkstra) entre l'adversaire et nous-mêmes
# =============================================================================

import time
import heapq
import numpy as np

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
        # Ces attributs commencent par _ pour ne pas apparaître dans les JSON de parties
        self._time_limit = None   # sera fixé dynamiquement à chaque appel
        self._start_time = None   # heure de début du tour

    # =========================================================================
    # POINT D'ENTRÉE PRINCIPAL
    # =========================================================================

    def compute_action(self, current_state: GameStateHex,
                       remaining_time: float = 15 * 60, **kwargs) -> Action:
        """
        Choisit la meilleure action via Minimax + Alpha-Beta + Iterative Deepening.

        On réserve un buffer de sécurité pour ne jamais dépasser le temps total.
        On consacre au plus TIME_PER_MOVE secondes par coup.
        """
        TIME_PER_MOVE = 10.0   # secondes max par coup
        SAFETY_BUFFER = 2.0    # on ne touche pas aux dernières secondes

        available = min(TIME_PER_MOVE, remaining_time - SAFETY_BUFFER)
        available = max(available, 0.5)   # toujours au moins 0.5s

        self._start_time = time.time()
        self._time_limit = self._start_time + available

        best_action = self._iterative_deepening(current_state)
        return best_action

    # =========================================================================
    # ITERATIVE DEEPENING
    # =========================================================================

    def _iterative_deepening(self, state: GameStateHex) -> Action:
        """
        Lance minimax à profondeur 1, 2, 3, ... jusqu'à manque de temps.
        Retourne toujours le meilleur résultat de la dernière profondeur complète.
        """
        # Récupère toutes les actions possibles une seule fois
        possible_actions = list(state.get_possible_stateless_actions())

        # Cas dégénéré : un seul coup possible
        if len(possible_actions) == 1:
            return possible_actions[0]

        # On ordonne les actions pour accélérer l'alpha-beta (heuristique simple)
        dim = state.get_rep().get_dimensions()[0]
        center = dim / 2.0
        possible_actions.sort(
            key=lambda a: abs(a.data["position"][0] - center)
                        + abs(a.data["position"][1] - center)
        )

        best_action = possible_actions[0]   # fallback : coup central
        depth = 1

        while True:
            # Vérifie s'il reste du temps pour une nouvelle profondeur
            if time.time() >= self._time_limit:
                break

            try:
                action, _ = self._minimax_root(state, possible_actions, depth)
                best_action = action   # sauvegarde le résultat complet
                depth += 1

                # Profondeur max raisonnable pour un 14x14
                if depth > 6:
                    break

            except _TimeOut:
                # La recherche a été interrompue en cours de route :
                # on garde le meilleur résultat de la profondeur précédente
                break

        return best_action

    def _minimax_root(self, state: GameStateHex, actions, depth: int):
        """
        Niveau racine du minimax : on est toujours MAX ici.
        Retourne (meilleure_action, meilleur_score).
        """
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

    # =========================================================================
    # MINIMAX AVEC ALPHA-BETA
    # =========================================================================

    def _minimax(self, state: GameStateHex, depth: int,
                 alpha: float, beta: float, is_maximizing: bool) -> float:
        """
        Minimax récursif avec élagage alpha-beta.

        alpha = meilleur score que MAX est sûr d'obtenir (initiallement -inf)
        beta  = meilleur score que MIN est sûr d'obtenir (initialement +inf)

        Si beta <= alpha : on "élagué" (prune) — on arrête d'explorer cette branche.
        """
        self._check_time()

        # --- Cas terminal : partie finie ou profondeur atteinte ---
        if state.is_done():
            # Quelqu'un a gagné : score extrême selon le gagnant
            scores = state.get_scores()
            my_id = self._get_my_player_id(state)
            return 10000.0 if scores.get(my_id, 0) == 1.0 else -10000.0

        if depth == 0:
            return self._heuristic(state)

        # --- Génère les actions et les ordonne (centre d'abord) ---
        actions = list(state.get_possible_stateless_actions())
        dim = state.get_rep().get_dimensions()[0]
        center = dim / 2.0
        actions.sort(
            key=lambda a: abs(a.data["position"][0] - center)
                        + abs(a.data["position"][1] - center)
        )

        if is_maximizing:
            # C'est notre tour : on cherche le score maximum
            max_score = float('-inf')
            for action in actions:
                self._check_time()
                next_state = state.apply_action(action)
                score = self._minimax(next_state, depth - 1,
                                      alpha, beta, is_maximizing=False)
                max_score = max(max_score, score)
                alpha = max(alpha, max_score)
                if beta <= alpha:
                    break   # Beta cut-off : l'adversaire ne choisira jamais ce sous-arbre
            return max_score

        else:
            # C'est le tour de l'adversaire : il cherche le score minimum (pour nous)
            min_score = float('inf')
            for action in actions:
                self._check_time()
                next_state = state.apply_action(action)
                score = self._minimax(next_state, depth - 1,
                                      alpha, beta, is_maximizing=True)
                min_score = min(min_score, score)
                beta = min(beta, min_score)
                if beta <= alpha:
                    break   # Alpha cut-off : on ne choisira jamais ce sous-arbre
            return min_score

    # =========================================================================
    # HEURISTIQUE : DIFFÉRENCE DE CHEMINS LES PLUS COURTS (DIJKSTRA)
    # =========================================================================

    def _heuristic(self, state: GameStateHex) -> float:
        """
        Score = (longueur chemin adversaire) - (longueur chemin joueur courant)

        Plus notre chemin est court et celui de l'adversaire est long, mieux c'est.
        On utilise Dijkstra : les cases vides coûtent 1, nos pièces coûtent 0.
        Les pièces adverses sont des murs (on les ignore / leur attribue un coût infini).
        """
        my_dist = self._shortest_path(state, self.piece_type)
        opp_type = "B" if self.piece_type == "R" else "R"
        opp_dist = self._shortest_path(state, opp_type)

        # Si l'adversaire a gagné : score catastrophique
        if opp_dist == 0:
            return -10000.0
        # Si on a gagné : score maximal
        if my_dist == 0:
            return 10000.0

        return float(opp_dist) - float(my_dist)

    def _shortest_path(self, state: GameStateHex, piece_type: str) -> float:
        """
        Dijkstra du côté de départ vers le côté d'arrivée pour un joueur donné.

        Rouge (R) : relie ligne 0 → ligne dim-1 (haut → bas)
        Bleu  (B) : relie colonne 0 → colonne dim-1 (gauche → droite)

        Coût d'une case :
            - 0  si elle contient déjà une pièce du bon joueur
            - 1  si elle est vide
            - inf si elle contient une pièce adverse (mur infranchissable)
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]

        # dist[i][j] = distance minimale connue vers (i,j)
        dist = np.full((dim, dim), np.inf)
        pq = []   # min-heap : (coût, (i, j))

        if piece_type == "R":
            # Départ : toutes les cases de la ligne 0
            for j in range(dim):
                cell = env.get((0, j))
                if cell is None:
                    cost = 1
                elif cell.piece_type == "R":
                    cost = 0
                else:
                    continue   # pièce bleue sur la ligne de départ → mur
                if cost < dist[0, j]:
                    dist[0, j] = cost
                    heapq.heappush(pq, (cost, (0, j)))
        else:
            # Départ : toutes les cases de la colonne 0
            for i in range(dim):
                cell = env.get((i, 0))
                if cell is None:
                    cost = 1
                elif cell.piece_type == "B":
                    cost = 0
                else:
                    continue
                if cost < dist[i, 0]:
                    dist[i, 0] = cost
                    heapq.heappush(pq, (cost, (i, 0)))

        # Dijkstra
        while pq:
            d, (i, j) = heapq.heappop(pq)

            if d > dist[i, j]:
                continue   # entrée obsolète dans le heap

            # Vérifie si on a atteint le côté d'arrivée
            if piece_type == "R" and i == dim - 1:
                return d
            if piece_type == "B" and j == dim - 1:
                return d

            # Explore les voisins (6 directions hexagonales)
            for _, (ni, nj) in state.get_rep().get_neighbours(i, j).values():
                if ni < 0 or ni >= dim or nj < 0 or nj >= dim:
                    continue
                neighbor_cell = env.get((ni, nj))
                if neighbor_cell is not None and neighbor_cell.piece_type != piece_type:
                    continue   # pièce adverse : mur

                move_cost = 0 if (neighbor_cell is not None) else 1
                new_dist = d + move_cost

                if new_dist < dist[ni, nj]:
                    dist[ni, nj] = new_dist
                    heapq.heappush(pq, (new_dist, (ni, nj)))

        return np.inf   # aucun chemin trouvé (ne devrait pas arriver avant la fin)

    # =========================================================================
    # UTILITAIRES
    # =========================================================================

    def _get_my_player_id(self, state: GameStateHex) -> int:
        """Retourne l'identifiant du joueur associé à notre piece_type."""
        for player in state.players:
            if player.get_piece_type() == self.piece_type:
                return player.get_id()
        return -1

    def _check_time(self):
        """Lève une exception si le temps alloué est dépassé."""
        if time.time() >= self._time_limit:
            raise _TimeOut()


# =============================================================================
# Exception interne pour interrompre la recherche proprement
# =============================================================================

class _TimeOut(Exception):
    """Levée quand le budget temps du tour est épuisé."""
    pass