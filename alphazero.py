from mcts import MCTS
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange
from Games.tictactoe import TicTacToe
from model import Model

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
                
    def train(self, memory, device,batch_size = 64):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), batch_size):
            sample = memory[batchIdx:min(len(memory), batchIdx + batch_size)]
            
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=device)

            out_policy, out_value = self.model(state.unsqueeze(1))
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step() 
    
    def learn(self, num_iterations = 50, num_selfPlay = 30, epochs = 10, device = "cpu"):
        for iteration in range(num_iterations):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in trange(num_selfPlay):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in trange(epochs):
                self.train(memory,device)
            
        torch.save({
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()
        },f"model_{iteration}.pth")

if __name__ == "__main__":
    # Check for available device (CUDA or MPS, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    device = "cpu"

    tictactoe = TicTacToe()

    model = Model().to(device)  # Move model to the chosen device

    tictactoe = TicTacToe()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    alphaZero = AlphaZero(model, optimizer, tictactoe)
    alphaZero.learn(num_iterations=100000,num_selfPlay=100,epochs=50,device=device)