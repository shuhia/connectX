import random
import numpy as np

# Checks if we have 4 in a row 
# Checks if opponent has 4 in a row
# Opponent position = position XOR mask
# connected_four(state) = true or false
# state = position
def my_agent(obs, config):

    #############
    # helper functions
        #source: https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0
    def get_position_mask_bitmap(board, player):
        position, mask = '', ''
        # Start with right-most column
        for j in range(6, -1, -1):
            # Add 0-bits to sentinel 
            mask += '0'
            position += '0'
            # Start with bottom row
            for i in range(0, 6):
                mask += ['0', '1'][board[i, j] != 0]
                position += ['0', '1'][board[i, j] == player]
        return int(position, 2), int(mask, 2)


    def connected_four(position):
        # Horizontal check
        m = position & (position >> 7)
        if m & (m >> 14):
            return True
        # Diagonal \
        m = position & (position >> 6)
        if m & (m >> 12):
            return True
        # Diagonal /
        m = position & (position >> 8)
        if m & (m >> 16):
            return True
        # Vertical
        m = position & (position >> 1)
        if m & (m >> 2):
            return True
        # Nothing found
        return False
    # connected n


    def make_move(position, mask, col):
        new_position = position ^ mask
        new_mask = mask | (mask + (1 << (col*7)))
        return new_position^new_mask, new_mask

   
    # negamax
    # Plan moves n steps ahead
    ######## helper functions
    # score function
    def score(pos, mask):
        # Player
        sum = 0;
        if(connected_four(pos)):
            sum += 22
        # Opponent
        elif(connected_four(pos^mask)):
            sum += -22
  
        return sum

    # negamax alpha-beta
    def negamax(node,config, depth, a, b, c):
        pos, mask = node
        # Check if terminal
        if depth == 0 or connected_four(pos) or connected_four(pos^mask):
            return c*score(pos,mask);
        moves = []
        # generate valid moves, every seventh bit and pos
        for n in range(7):
            # shift 6 bits every cycle and check if the spot is empty
            _, temp = make_move(pos, mask,5*(n+1))
            if(temp^mask != 0):
                moves.append(n)
        
        nodes = [make_move(pos,mask,col) for col in moves]

        value = -10**9

        for node in nodes:
            value = max(value, -negamax(node, config, depth-1, -b,-a,-c))
            a = max(a, value)
            if( a >= b):
                break

        return value
    
    def get_move(obs, config):
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        pos, mask = get_position_mask_bitmap(grid, obs.mark);
        v_moves = [c for c in range(config.columns) if obs.board[c] == 0]
        ############


        # Heuristic
        # Checks if game can be won
        # Stop opponent from ending the game.
   
        for col in v_moves:
            pos2, mask2 = make_move(pos, mask, col)
            # Win
            if(connected_four(pos2)): return col
            # Prevent lose
            elif(connected_four(pos2^mask2)): return col;


        # Get position and values for player
        root = pos,mask
        depth = 5
        map = {move:negamax(root,config,depth,-10**9, 10**9,1) for move in v_moves}

        # Choose the best positions
        print(map)
        max_key = max(map, key = map.get)
        return max_key


    return get_move(obs, config)

   





    # depth
    # alpha beta
    # table 
# Table lookup

# Neural network

