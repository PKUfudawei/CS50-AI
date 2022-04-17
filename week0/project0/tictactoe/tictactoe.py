"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_X = sum(row.count(X) for row in board)
    num_O = sum(row.count(O) for row in board)
    if num_X <= num_O:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    _actions=set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j]==EMPTY:
                _actions.add((i,j))
    return _actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i,j=action
    _board=[[element for element in row] for row in board]
    _board[i][j]=player(board)
    return _board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """ 
    rows=[board[i][:] for i in range(len(board))]
    cols=[[board[i][j] for i in range(len(board))] for j in range(len(board[0]))]
    diags=[[board[i][i] for i in range(len(board))], [board[i][~i] for i in range(len(board))]]
    for row in rows+cols+diags:
        if row.count(X)==3:
            return X
        elif row.count(O)==3:
            return O
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) or actions(board)==set():
        return True
    else:
        return False

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board)==X:
        return 1
    elif winner(board)==O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
      
    def max_value(board):
        if terminal(board):
            return utility(board)
        v=-math.inf
        for action in actions(board):
            v=max(v, min_value(result(board,action)))
        return v
    
    def min_value(board):
        if terminal(board):
            return utility(board)
        v=math.inf
        for action in actions(board):
            v=min(v, max_value(result(board,action)))
        return v
    
    if player(board)==X:
        optimal_action=max(actions(board), key=lambda action: min_value(result(board,action)))
    else:
        optimal_action=min(actions(board), key=lambda action: max_value(result(board,action)))
        
    return optimal_action
        
            
