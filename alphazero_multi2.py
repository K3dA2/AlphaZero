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
import time

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
            
            self.model = self.model.to(device)
            
            out_policy, out_value = self.model(state.unsqueeze(1))
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.to(device)
                    
            self.optimizer.step()

def self_play_worker(data_queue, weight_queue, game):
    """Worker function for self-play games with weight updates."""
    device = torch.device("cpu")  # Workers always use CPU
    model = Model().to(device)
    model.eval()
    
    while True:
        try:
            new_weights = weight_queue.get_nowait()
            model.load_state_dict(new_weights)
            print(f"Worker updated weights")
        except queue.Empty:
            pass
        
        alpha_zero = AlphaZero(model, None, game)
        memory = alpha_zero.selfPlay()
        data_queue.put(memory)

def gpu_training_process(data_queue, weight_queues, model, optimizer, device, iteration_start=0, batch_size=64, epochs=50, max_iterations=100, update_frequency=100):
    """Training process that periodically updates worker weights."""
    model = model.to(device)
    
    # Move optimizer states to correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    
    model.train()
    
    with tqdm(total=max_iterations, desc="Total Training Iterations", leave=True) as iteration_bar:
        iteration_count = iteration_start

        while iteration_count < max_iterations:
            memory = []
            try:
                while len(memory) < batch_size:
                    memory += data_queue.get(timeout=1)
            except queue.Empty:
                continue

            alpha_zero = AlphaZero(model, optimizer, None)
            for epoch in trange(epochs, desc=f"Training Iteration {iteration_count+1}/{max_iterations}", leave=True):
                alpha_zero.train(memory, device, batch_size)

            if iteration_count % update_frequency == 0:
                # Move model to CPU before getting state dict
                model_cpu = model.cpu()
                cpu_weights = model_cpu.state_dict()
                
                # Distribute to all workers
                for weight_queue in weight_queues:
                    try:
                        while not weight_queue.empty():
                            weight_queue.get_nowait()
                        weight_queue.put(cpu_weights)
                    except queue.Full:
                        pass
                
                # Move model back to training device
                model = model_cpu.to(device)
                print(f"Updated worker weights at iteration {iteration_count}")

            if iteration_count % 5000 == 0:
                model_cpu = model.cpu()
                torch.save({
                    'model_state_dict': model_cpu.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f"weights/model_checkpoint_iter_{iteration_count+1}.pth")
                model = model_cpu.to(device)

            iteration_bar.update(1)
            iteration_count += 1

    print("Training process completed after reaching max_iterations.")

if __name__ == "__main__":
    mp.set_start_method('spawn')

    # Initialize device
    device = torch.device("mps" if torch.backends.mps.is_built() and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparams
    max_iteration = 100001
    batch_size = 64
    epochs = 50
    iteration_start = 20001

    # Game and model setup
    tictactoe = TicTacToe()
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load checkpoint - make sure model is on CPU when loading
    model = model.cpu()
    checkpoint = torch.load("weights/model_checkpoint_iter_20001.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Data queue for game results
    data_queue = mp.Queue(maxsize=100)
    
    # Create weight queues for each worker
    num_workers = mp.cpu_count() - 1
    weight_queues = [mp.Queue(maxsize=1) for _ in range(num_workers)]

    # Start training process
    training_process = mp.Process(
        target=gpu_training_process, 
        args=(data_queue, weight_queues, model, optimizer, device, iteration_start, batch_size, epochs, max_iteration)
    )
    training_process.start()

    # Start workers with their respective weight queues
    workers = [
        mp.Process(
            target=self_play_worker, 
            args=(data_queue, weight_queue, tictactoe)
        )
        for weight_queue in weight_queues
    ]
    for worker in workers:
        worker.start()

    # Wait for processes to finish
    training_process.join()
    for worker in workers:
        worker.join()