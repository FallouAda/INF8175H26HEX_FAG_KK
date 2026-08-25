Agent IA pour le jeu de Hex — Mégalodon de l'Atlantide

Agent intelligent capable de jouer au jeu de Hex sur un plateau 14×14, développé dans le cadre du cours INF8175 — Intelligence Artificielle à Polytechnique Montréal.

Projet réalisé en équipe de 2 avec Kylian Kouassi.

🎯 Le jeu de Hex

Le Hex est un jeu de stratégie à somme nulle sans possibilité de match nul : chaque joueur essaie de connecter ses deux bords opposés du plateau avec une chaîne continue de ses pièces, avant que l'adversaire ne connecte les siens.

🧠 Approche

L'agent repose sur trois composantes principales :

1. Minimax

Construction d'un arbre de coups possibles où l'agent maximise son score en supposant que l'adversaire joue toujours le coup le pire pour lui (minimisation).

2. Élagage Alpha-Beta

Permet d'ignorer les branches de l'arbre qui ne peuvent pas influencer la décision finale, ce qui double environ la profondeur de recherche atteignable dans le même temps de calcul — un gain critique sur un plateau de 196 cellules.

3. Iterative Deepening (approfondissement itératif)

L'agent cherche d'abord à profondeur 1, puis 2, puis 3, etc., en conservant toujours le résultat de la dernière recherche complète. Si le temps alloué au tour est écoulé en cours de recherche, il retombe sur le résultat de la profondeur précédente — garantissant qu'il ne dépasse jamais le temps imparti.

4. Heuristique — plus court chemin (Dijkstra)

À profondeur maximale, la position est évaluée avec un algorithme de Dijkstra multi-source :

Distance la plus courte pour connecter ses propres bords (pièces alliées = coût 0, cellules vides = coût 1, pièces adverses = mur infranchissable)
Même calcul pour l'adversaire
score = distance_adversaire − distance_propre

Un score élevé signifie que l'agent est proche de la victoire pendant que l'adversaire en est loin.

5. Move ordering

Les coups sont triés par proximité au centre du plateau avant exploration, les coups centraux étant généralement plus forts au Hex — ce qui augmente l'efficacité de l'élagage alpha-beta.

📁 Structure du projet
.
├── my_player.py         # Notre agent (Minimax + Alpha-Beta + Iterative Deepening)
├── player_hex.py         # Classe de base fournie par le cours
├── game_state_hex.py     # Représentation de l'état du jeu
└── seahorse/              # Framework de jeu fourni par le cours (INF8175)
⚙️ Fonctionnement
compute_action()
  └── iterative deepening (profondeur 1, 2, 3... jusqu'à épuisement du temps)
        └── minimax(state, profondeur, alpha, beta, is_maximizing)
              ├── si profondeur == 0 ou fin de partie → heuristic_score(state)
              ├── si maximisation → essayer tous les coups, garder le max
              └── si minimisation → essayer tous les coups, garder le min

heuristic_score(state)
  └── Dijkstra (plus court chemin) pour soi-même
  └── Dijkstra (plus court chemin) pour l'adversaire
  └── retourne distance_adversaire − distance_propre
🏆 Résultats

Lors du tournoi du cours, l'agent a terminé avec un bilan de 4 victoires / 5 défaites (44 %) :

100 % de victoires en jouant Rouge
17 % de victoires en jouant Bleu
🛠️ Technologies
Python 3
NumPy — calculs matriciels et Dijkstra
Seahorse — framework de jeu fourni pour le cours INF8175
📚 Contexte académique

Projet réalisé dans le cadre du cours INF8175 (Intelligence Artificielle) à Polytechnique Montréal, 2025.
