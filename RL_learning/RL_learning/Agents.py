import numpy as np
import math
import random
# Agents to implement
    # Minimax agent
    # Kaggle agents

# An agent takes args: observation, configuration



# Helper functions for minMaxAgent

# Helper function for get_heuristic: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)

# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows

# Helper function for score_move: gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid

def get_heuristic(grid, mark, config):
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_threes_opp = count_windows(grid, 3, mark%2+1, config)
    num_fours_opp = count_windows(grid, 4, mark%2+1, config)
    score = num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours
    return score

def score_move2(grid, col, mark, config, nsteps):
    # Gets the next grid for the move
    next_grid = drop_piece(grid, col, mark, config)
    # Check if there are any winning moves
    score = negamax(next_grid, nsteps-1, mark, 1, config, -math.inf, math.inf)

    return score

# Helper function for minimax: checks if agent or opponent has four in a row in the window
def is_terminal_window(window, config):
    return window.count(1) == config.inarow or window.count(2) == config.inarow

# Helper function for minimax: checks if game has ended
def is_terminal_node(grid, config):
    # Check for draw 
    if list(grid[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal
    # horizontal 
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if is_terminal_window(window, config):
                return True
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if is_terminal_window(window, config):
                return True
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if is_terminal_window(window, config):
                return True
    return False
tables = dict()
def negamax(node, depth, mark, color, config, alpha, beta):
            # Check if node is in the table
        
        if node.tobytes() in tables:
            table = tables[node.tobytes()]
            #check if node is in correct depth
            if table['depth'] >= depth:
                if(table['flag'] == 0):
                    return table['value']
                elif(table['flag'] == -1):
                    alpha = max(alpha, table['value'])
                elif(table['flag'] == 1):
                    beta = min(beta, table['value'])
                if alpha >= beta:
                    return table['value']
        # Get all the valid moves
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        
        # If it is the last node
        if (depth == 0 or is_terminal_node(node, config)):
            #Color changes the sign of value
                return get_heuristic(node, mark, config)*color
        value = -10000;

        for move in valid_moves:
            # Returns the state after executing a move
            
            child = drop_piece(node, move, mark, config)

            value = max(value, -negamax(child, depth - 1, mark, -color, config, -beta, -alpha))
            
            a = max(alpha,value)

            if a >= beta:
                break
        
        d = {'value':value, 'flag':0, 'depth':depth}
        if value <= alpha:
            d['flag'] = 1;
        elif value <=beta:
            d['flag'] = -1
        tables[node.tobytes()] = d
        
        return value;
# This agent random chooses a non-empty column.
    
class Agent_negamax():
    def __init__(self, *args, **kwargs):
        self.transposition_table = dict()
        return super().__init__(*args, **kwargs)

    def negamaxAgent(self, obs, config):
        N_STEPS = 3
        # Get list of valid moves
        player = obs.mark;
        valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
       
        # Convert the board to a 2D grid
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        # If first player then place the piece in the middle
        if(max(obs.board)==0): return 3
        for col in valid_moves:
            if count_windows(drop_piece(grid, col, obs.mark, config), 4, obs.mark, config) > 0:
                return col
            if(count_windows(drop_piece(grid, col, obs.mark%2+1, config),4,obs.mark%2+1,config) > 0):
                return col
        # Use the heuristic to assign a score to each possible board in the next step
        scores = dict(zip(valid_moves, [score_move2(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
        # Get a list of columns (moves) that maximize the heuristic
        max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
        # Select at random from the maximizing columns
        return random.choice(max_cols)
    