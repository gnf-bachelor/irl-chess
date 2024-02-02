from project.chess_utils.sunfish import Position, Move, Searcher, render
from time import time


def sunfish_move(searcher: Searcher, hist: list[Position], time_limit:float=1., ) -> tuple[Move, dict]:
    """
    Given a sunfish searcher, the game history so far
    and a time_limit in seconds, return the move that
    in the given time was found to have the best score.
    Also returns a dictionary with the info from the
    search. If there is no best move found it (this
    should in theory be impossible) it crashes.
    """
    start = time()
    best_move = None
    for depth, gamma, score, move in searcher.search(hist):
        if score >= gamma:
            best_move = move
        if best_move and time() - start > time_limit:
            break

    info = {'depth': depth,
            'gamma': gamma,
            'score': score,
            'nodes': searcher.nodes}
    return best_move, info


def sunfish_move_to_str(move: Move, is_black:bool=False):
    i, j = move.i, move.j
    if is_black:
        i, j = 119 - i, 119 - j
    move_str = render(i) + render(j) + move.prom.lower()
    return move_str
