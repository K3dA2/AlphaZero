import asyncio
import websockets
import json
import numpy as np
from model import Model
import torch
from Games.tictactoe import TicTacToe

def convert_to_numpy(data):
    # Define a mapping for each symbol to a numerical value
    symbol_map = {'X': 1, 'O': -1, ' ': 0}
    
    # Map each item in the list using the dictionary, and convert it to a NumPy array
    numeric_data = np.array([symbol_map[item] for item in data])
    
    # Reshape the flat array into a 3x3 array
    return numeric_data.reshape(3, 3)

async def server(ws, path):
    model = Model()
    checkpoint = torch.load("weights/model_checkpoint_iter_35001.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    game = TicTacToe()
    
    async for message in ws:
        board = json.loads(message)  # Parse received JSON board state
        print(f"Received board state from client: {board}")
        
        state = convert_to_numpy(board)
        
        state *= -1
        #print(state)

        with torch.no_grad():
            policy, value = model(
                        torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    )
            policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        valid_moves = game.get_valid_moves(state)
        
        policy *= valid_moves.reshape(-1)
        policy /= np.sum(policy)

        #action = np.argmax(policy)
        action = np.random.choice(len(policy), p=policy)
        print(action)

        if action is not None:
            await ws.send(str(action))  # Send the cell index back to the client
        else:
            print("No possible actions (game may be over)")

start_server = websockets.serve(server, "localhost", 5000)
print("Server Started")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
