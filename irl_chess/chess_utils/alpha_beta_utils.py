from copy import deepcopy
from collections import deque
from itertools import chain, count
from collections.abc import Sized, Iterable, Iterator
from joblib import Parallel, delayed
import chess
import numpy as np
from tqdm import tqdm
from queue import PriorityQueue
from irl_chess.chess_utils.sunfish import piece, pst, pst_only
from irl_chess.chess_utils.sunfish_utils import eval_pos

class PriorityQueueWithFIFO(PriorityQueue):
    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize)
        self.counter = count()
        self.size = 0

    def put(self, priority_item):
        self.size += 1
        priority, data = priority_item
        tie_breaker = - next(self.counter) # Negative for LIFO behavior. 
        super().put((priority, tie_breaker, data))

    def get(self):
        self.size -= 1
        priority, _, data = super().get()
        return priority, data
    
    def update_best_moves(self, eval, data, maximize):
        self.put(((1 if maximize else -1)*eval, data))
        if self.full():
            self.get()  # Remove the least best move if queue is full. Keep the highest eval. 

    def process_move(self, eval, data, maximize):
        self.update_best_moves(eval, data, maximize)
        # Return evaluation of worst move in priority queue. Highest for white and lowest for black. 
        return (1 if maximize else -1)*self.queue[0][0]   
    
    def to_ordered_list(self, maximize, k = None, verbose = False) -> list[tuple[float, type]]:
        # Extract the k best moves from the priority queue
        if k is None: k = self.maxsize-1
        k_best_moves = []
        while not self.empty():
            eval, data = self.get()
            k_best_moves.append(((1 if maximize else -1)*eval, data))
        k_best_moves.reverse() # Order from best to worst
        if verbose and len(k_best_moves) != k: print(f"Warning: only {len(k_best_moves)} available and not {k} from this position")
        return k_best_moves

# Evaluates the piece positions according to sunfish
def eval_pst_only(board):
    s = str(board()).replace(' ', '').replace('\n', '')
    eval = 0
    for i, char in enumerate(s):
        if char == '.':
            continue
        else:
            eval += pst_only[char][i]
    return eval

def evaluate_board(board, R, pst = False, white=True):
    """
    positive if (WhitePieces + white perspective), 
    negative if (Not WhitePieces + white perspective), 

    :param board:
    :param R:
    :param pst: Whether to include piece square tables or not. Currently not implemented.
    :param white: True if viewing from White's perspective. Should be always be left as true since black is trying to minimize.  
    :return:
    """
    if pst: # I am unsure which function is faster. This one of the eval_pos when using pst and without. We could test it.  
        #print("I am quite certain there is a bug in how the following function flips the board to evaluate the position")
        return eval_pos(board, R=R)
    eval = 0
    for WhitePieces in (True, False):
        keys = {val if WhitePieces else val.lower(): 0 for val in piece.keys()}
        for char in board.fen().split(" ", 1)[0]: # Do not include turn and castling information. The "b" for black turn is counted as a black rook. Strip the end of the string.
            if char in keys:
                keys[char] += 1
        pos = np.array([val for val in keys.values()])
        eval += (pos @ R) * (1 if WhitePieces else -1) # Add if 
    return eval * (1 if white else -1) + (eval_pst_only(board) if pst else 0)

def move_generator(board: chess.Board, depth: int) -> Iterator[chess.Move]:
    # Skip captures on the second pass since we already considered them.
    if depth > 0: it = chain(board.generate_legal_captures(), filter(lambda m : not board.is_capture(m), board.generate_legal_moves()))
    else: it = board.generate_legal_captures()
    next_move = next(it, -1)
    no_moves = next_move == -1
    if not no_moves: it = chain([next_move], it)
    return it, no_moves

def no_moves_eval(board: chess.Board, R: np.array, pst: bool, evaluation_function=evaluate_board, _maximize: bool = True) -> tuple[float, chess.Board, deque]:
    if board.outcome() is None: # Game is still going on
        assert not board.is_game_over()
        final_score = evaluation_function(board, R, pst)
    elif board.outcome().winner is None: # The game is tied
        final_score = 0
    else:
        final_score = (np.sum(R)*100 if board.outcome().winner else -np.sum(R)*100)
    return final_score * (1 if _maximize else -1), deepcopy(board), deque() # This evaluation function should be static. Positive is good for white and negative is good for black.

def alpha_beta_search_k(board: chess.Board,
                      depth,
                      k=1, # Number of best moves to return
                      alpha=-np.inf,
                      beta=np.inf,
                      maximize=True,
                      R: np.array = np.zeros(1),
                      pst: bool = False,
                      evaluation_function=evaluate_board,
                      quiesce: bool = False) -> list[tuple[float, chess.Board, deque]]: 
    """
    When maximize is True the board must be evaluated from the White
    player's perspective.

    :param board:
    :param depth:
    :param k: Number of best moves to return
    :param alpha:
    :param beta:
    :param maximize: The current turn. White is True and black is False.
    :param R:
    :param evaluation_function:
    :return:
    """
    assert board.turn == maximize
    #if depth < 0: print(f"Deep diving with depth: {depth}")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Priority queue to store k best moves
    best_moves = PriorityQueueWithFIFO(maxsize=k+1) # Whenever full, eject worst (k+1)-1=k

    if maximize:
        it, no_move = move_generator(board, depth)
        
        if no_move or depth == 0: 
            if no_move or (not quiesce): # Static evaluation.
                static_eval, board_best, move_queue_best = no_moves_eval(board, R, pst, evaluation_function)
                return [(static_eval, board_best, move_queue_best)]
            else:
                return quiescence_search(board, depth, k=1, alpha=alpha, beta=beta, maximize=True, R=R, pst=pst, evaluation_function=evaluation_function)
        

        if ((not no_move) and (depth > 0)): # Consider next moves if we have the depth or we are quiescing. 
            for move in it:
                board.push(move)
                eval, board_last, move_queue = alpha_beta_search_k(board, depth - 1, k=1, alpha=alpha, beta=beta, maximize=False, R=R, pst=pst, evaluation_function=evaluation_function, quiesce=quiesce)[0]
                board.pop()
                move_queue.appendleft(move)
                atLeastEval = best_moves.process_move(eval, (board_last, move_queue), maximize) # Returns eval of the worst move in our best_moves list
                if best_moves.size == k: alpha = atLeastEval
                if beta <= alpha:
                    break  # Beta cut-off
    
    else:
        it, no_move = move_generator(board, depth)
        
        if no_move or depth == 0: 
            if no_move or (not quiesce): # Static evaluation.
                static_eval, board_best, move_queue_best = no_moves_eval(board, R, pst, evaluation_function)
                return [(static_eval, board_best, move_queue_best)]
            else:
                return quiescence_search(board, depth, k=1, alpha=alpha, beta=beta, maximize=False, R=R, pst=pst, evaluation_function=evaluation_function)

        if ((not no_move) and (depth > 0)): # Consider next moves if we have the depth or we are quiescing.
            for move in it:
                board.push(move)
                eval, board_last, move_queue = alpha_beta_search_k(board, depth=depth - 1, k=1, alpha=alpha, beta=beta, maximize=True, R=R, pst=pst, evaluation_function=evaluation_function, quiesce=quiesce)[0]
                board.pop()
                move_queue.appendleft(move)
                atLeastEval = best_moves.process_move(eval, (board_last, move_queue), maximize) # Returns eval of the worst move in our best_moves list
                if best_moves.size == k: beta = atLeastEval
                if beta <= alpha:
                    break  # Alpha cut-off

    return [(eval, board, move_queue) for eval, (board, move_queue) in best_moves.to_ordered_list(maximize, k)]

def quiescence_search(board: chess.Board,
                      depth,
                      k=1, # Number of best moves to return
                      alpha=-np.inf,
                      beta=np.inf,
                      maximize=True,
                      R: np.array = np.zeros(1),
                      pst: bool = False,
                      evaluation_function=evaluate_board) -> list[tuple[float, chess.Board, deque]]:
    """
    When maximize is True the board must be evaluated from the White
    player's perspective.

    :param board:
    :param depth:
    :param k: Number of best moves to return
    :param alpha:
    :param beta:
    :param maximize: The current turn. White is True and black is False.
    :param R:
    :param evaluation_function:
    :return:
    """
    assert board.turn == maximize
    #if depth < 0: print(f"Deep diving with depth: {depth}")

    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Priority queue to store k best moves
    best_moves = PriorityQueueWithFIFO(maxsize=k+1) # Whenever full, eject worst (k+1)-1=k 

    if maximize:
        it, no_move = move_generator(board, depth) # Only generates captures because depth <= 0
        
        # Static evaluation. Alternative to taking a piece for Quiesce. 
        static_eval, board_best, move_queue_best = no_moves_eval(board, R, pst, evaluation_function)
        if static_eval >= beta: return [(beta, board_best, move_queue_best)] # Position is too good for white for black to allow. 
                # Or at best equal to something already explored. This means it should never be included in our final returned moves.
        atLeastEval = best_moves.process_move(static_eval, (board_last, move_queue), maximize) 
        if best_moves.size == k: alpha = atLeastEval

        
        if (not no_move):
            for move in it:
                board.push(move)
                # Be really careful about the quiesce line! Have to check it thoroughly. 
                eval, board_last, move_queue = quiescence_search(board, depth - 1, k=1, alpha=alpha, beta=beta, maximize=False, R=R, pst=pst, evaluation_function=evaluation_function)[0]
                board.pop()
                move_queue.appendleft(move)
                atLeastEval = best_moves.process_move(eval, (board_last, move_queue), maximize) # Returns eval of the worst move in our best_moves list
                if best_moves.size == k: alpha = atLeastEval
                if beta <= alpha:
                    break  # Beta cut-off
    
    else:
        it, no_move = move_generator(board, depth)
        
        # Static evaluation. Alternative to taking a piece for Quiesce. 
        static_eval, board_best, move_queue_best = no_moves_eval(board, R, pst, evaluation_function)
        if static_eval <= alpha: return [(alpha, board_best, move_queue_best)] # Position is too good for black for white to allow. 
                # Or at best equal to something already explored. This means it should never be included in our final returned moves.
        atLeastEval = best_moves.process_move(static_eval, (board_last, move_queue), maximize) 
        if best_moves.size == k: beta = atLeastEval

        if (not no_move):
            for move in it:
                board.push(move)
                eval, board_last, move_queue = quiescence_search(board, depth=depth - 1, k=1, alpha=alpha, beta=beta, maximize=True, R=R, pst=pst, evaluation_function=evaluation_function)[0]
                board.pop()
                move_queue.appendleft(move)
                atLeastEval = best_moves.process_move(eval, (board_last, move_queue), maximize) # Returns eval of the worst move in our best_moves list
                if best_moves.size == k: beta = atLeastEval
                if beta <= alpha:
                    break  # Beta cut-off

    return [(eval, board, move_queue) for eval, (board, move_queue) in best_moves.to_ordered_list(maximize, k)]

def alpha_beta_search(board: chess.Board,
                      depth,
                      alpha=-np.inf,
                      beta=np.inf,
                      maximize=True,
                      R: np.array = np.zeros(1),
                      pst: bool = False,
                      evaluation_function=evaluate_board,
                      quiesce: bool = False) -> list[tuple[float, chess.Board, deque]]: 
    
    return alpha_beta_search_k(board, depth, k = 1, alpha=alpha, beta=beta, maximize=maximize, R= R,
                      pst=pst, evaluation_function=evaluation_function, quiesce=quiesce)[0]

def list_first_moves(k_best_moves: list[tuple[float, chess.Board, deque]], with_eval = False):
    assert all([len(moves) > 0 for eval, board_last, moves in k_best_moves]), "Some of the move tractories contain no moves. The preceeding function was probably run with depth 0."
    if not with_eval:
        return [deepcopy(moves).popleft() for eval, board_last, moves in k_best_moves]
    else: 
        return [(eval, deepcopy(moves).popleft()) for eval, board_last, moves in k_best_moves]
