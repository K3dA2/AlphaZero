import torch
import numpy as np

class TicTacToe:
    def __init__(self):
        self.action_size = 9

    def reset(self):
        """
        Returns a new empty board as a 3x3 grid initialized to 0.
        """
        return np.array([[0 for _ in range(3)] for _ in range(3)])

    def make_move(self, board, action, current_player):
        """
        Makes a move on the given board for the current player.
        """
        row, col = self.index_to_action(action)
        if board[row][col] != 0:
            raise ValueError("Invalid move! Cell is already occupied.")
        new_board = [row[:] for row in board]  # Create a copy of the board
        new_board[row][col] = current_player
        return new_board

    def is_valid_move(self, board, action):
        """
        Checks if the move is valid for the given board.
        """
        row, col = self.index_to_action(action)
        return board[row][col] == 0

    def check_winner(self, board):
        """
        Checks if there's a winner on the current board.
        """
        
        for i in range(3):
            if abs(sum(board[i])) == 3:  # Check rows
                return board[i][0]
            if abs(sum([board[j][i] for j in range(3)])) == 3:  # Check columns
                return board[0][i]

        # Check diagonals
        if abs(board[0][0] + board[1][1] + board[2][2]) == 3:
            return board[0][0]
        if abs(board[0][2] + board[1][1] + board[2][0]) == 3:
            return board[0][2]

        # Check for draw (no empty cells)
        if all(cell != 0 for row in board for cell in row):
            return 0  # Draw
        return None  # Game is still ongoing

    def get_valid_moves(self, board):
        """
        Returns a numpy array with 1 where the valid moves (0s) are, and 0 otherwise.
        """
        board_np = np.array(board)
        valid_moves = (board_np == 0).astype(np.int32)
        return valid_moves

    def print_board(self, board):
        """
        Prints the current board with 'X', 'O', and empty spaces.
        """
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in board:
            print("|".join(symbols[cell] for cell in row))
            print("-" * 5)

    def get_board_tensor(self, board):
        """
        Returns the given board state as a PyTorch tensor.
        """
        board_tensor = torch.tensor(board, dtype=torch.float32)
        return torch.unsqueeze(torch.unsqueeze(board_tensor, dim=0), dim=0)

    def get_reward(self, board):
        """
        Returns the reward based on the game state.
        - 1 for a win for the current player.
        - -1 for a loss for the current player.
        - 0 for a draw or ongoing game.
        """
        winner = self.check_winner(board)
        if winner == 1:  # 'X' wins
            return 1 
        elif winner == -1:  # 'O' wins
            return -1 
        return 0  # Game is still ongoing

    def is_terminal(self, board):
        """
        Checks if the game has reached a terminal state.
        """
        return self.check_winner(board) is not None

    def change_perspective(self, state, player):
        state = np.array(state)
        return state * player
    
    def index_to_action(self, index):
        """
        Converts an action index (0-8) back to a board position (row, col).
        """
        return divmod(index, 3)


if __name__ == "__main__":
    # Initialize the game logic
    game = TicTacToe()

    # Initialize a board and the current player
    board = game.reset()
    current_player = 1  # Player 'X'

    # Main game loop
    while not game.is_terminal(board):
        game.print_board(board)
        print(f"Player {current_player}'s turn.")

        valid_moves = game.get_valid_moves(board)
        print(f"Valid moves:\n{valid_moves}")

        # Get input from user (or AI)
        row, col = map(int, input("Enter row and col: ").split())
        action = game.action_to_index(row, col)

        if game.is_valid_move(board, action):
            board = game.make_move(board, action, current_player)
            current_player = -current_player  # Switch turns
        else:
            print("Invalid move, try again.")

    # Print result
    winner = game.check_winner(board)
    if winner == 0:
        print("It's a draw!")
    else:
        print(f"Player {'X' if winner == 1 else 'O'} wins!")