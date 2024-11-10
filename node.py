import numpy as np
import math

class Node:
    def __init__(self, game, state, parent=None, action_taken=None,C = 2,prior=0):
        self.game = game
        self.C = C
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
    
        self.children = []
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.C * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                
                child_state = self.game.make_move(child_state, action, 1)
                
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, child_state, self, action,self.C, prob)
                self.children.append(child)
            
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = value * -1
       
        if self.parent is not None:
            self.parent.backpropagate(value)  
