import torch
from node import Node
import numpy as np

# Set the device to MPS if available, else CUDA if available, else CPU
device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
device = "cpu"

class MCTS:
    def __init__(self, game, model,num_searches = 500):
        self.game = game
        self.num_searches = num_searches
        self.model = model
        
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, state)
        
        for search in range(self.num_searches):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
            
            if not self.game.is_terminal(node.state): 
                value = 0
                is_terminal = False
            else:
                value = self.game.get_reward(node.state)
                is_terminal = True
            
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(node.state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                
                policy *= valid_moves.reshape(-1)
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs