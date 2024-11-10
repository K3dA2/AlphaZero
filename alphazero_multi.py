import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm.notebook import tqdm, trange
from Games.tictactoe import TicTacToe
from model import Model
from mcts import MCTS
import queue

class AlphaZero:
    def __init__(self, model, optimizer, game):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.mcts = MCTS(game, model)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.reset()
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            
            action_probs = self.mcts.search(neutral_state)
            memory.append((neutral_state, action_probs, player))
            
            action = np.random.choice(self.game.action_size, p=action_probs)
            state = self.game.make_move(state, action, player)
            value = self.game.get_reward(state)
            is_terminal = self.game.is_terminal(state)
            
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    returnMemory.append((
                        hist_neutral_state,
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            
            player = player * -1
                
    def train(self, memory, device, batch_size=64):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), batch_size):
            sample = memory[batchIdx:min(len(memory), batchIdx + batch_size)]
            state, policy_targets, value_targets = zip(*sample)
            state = torch.tensor(np.array(state), dtype=torch.float32, device=device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=device)

            out_policy, out_value = self.model(state.unsqueeze(1))
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def self_play_worker(data_queue, model, game):
    """Worker function for self-play games, collecting training data."""
    alpha_zero = AlphaZero(model.to("cpu"), None, game)
    while True:
        memory = alpha_zero.selfPlay()
        data_queue.put(memory)

def gpu_training_process(data_queue, model, optimizer, device, batch_size=64, epochs=50, max_iterations=100):
    """Process to handle training on the GPU using data collected by self-play workers."""
    model.to(device)
    
    # Initialize tqdm progress bar for the total number of iterations
    with tqdm(total=max_iterations, desc="Total Training Iterations", leave=True) as iteration_bar:
        iteration_count = 0

        while iteration_count < max_iterations:
            # Collect a batch of games from the queue
            memory = []
            try:
                while len(memory) < batch_size:
                    memory += data_queue.get(timeout=1)
            except queue.Empty:
                continue  # If queue is empty, just wait for more data

            # Train the model with collected data
            alpha_zero = AlphaZero(model, optimizer, None)
            for epoch in trange(epochs, desc=f"Training Iteration {iteration_count+1}/{max_iterations}", leave=True):
                alpha_zero.train(memory, device, batch_size)
                #tqdm.write(f"Epoch {epoch+1}/{epochs} completed for iteration {iteration_count+1}.")

            if iteration_count % 5000 == 0:
                # Save the model periodically (optional)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f"weights/model_checkpoint_iter_{iteration_count+1}.pth")

            # Update the progress bar and increment the iteration count
            iteration_bar.update(1)
            iteration_count += 1

    print("Training process completed after reaching max_iterations.")

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Use 'spawn' instead of 'fork'

    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_built() and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    #Hyperparams
    max_iteration = 100001
    batch_size=64
    epochs=50

    # Game and model setup
    tictactoe = TicTacToe()
    model = Model()
    model.share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    '''
    checkpoint = torch.load("weights/model_checkpoint_iter_10001.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    '''
    
    # Data queue for inter-process communication
    data_queue = mp.Queue(maxsize=100)

    # Start GPU training process
    training_process = mp.Process(target=gpu_training_process, args=(data_queue, model, optimizer, device, batch_size, epochs, max_iteration))
    training_process.start()

    # Start multiple self-play workers
    num_workers = mp.cpu_count() - 1  # Keep one CPU free
    workers = [
        mp.Process(target=self_play_worker, args=(data_queue, model, tictactoe))
        for _ in range(num_workers)
    ]
    for worker in workers:
        worker.start()

    # Wait for all processes to finish (in practice, you'd likely want more control here)
    training_process.join()
    for worker in workers:
        worker.join()
