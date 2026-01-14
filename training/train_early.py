import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from core.game import CatanGame
from agents.early_agent import DQN, encode_vertex
from core.utils import PIPS


MODEL_PATH = "models/early.pt"
EPISODES = 5000
GAMMA = 0.95
LR = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_reward(state):
    production = sum(state[:5])          
    diversity = state[5] * 1.5            
    synergy = state[6] * 3 + state[7] * 3 
    edge_penalty = -2 if state[8] == 1 else 0

    return production + diversity + synergy + edge_penalty


def train():
    os.makedirs("models", exist_ok=True)
    
    model = DQN(input_size=12).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Wczytano model z {MODEL_PATH}")

    epsilon = EPSILON_START
    rewards_history = []

    print(f"Start treningu Early Agent")
    print(f"Epizody: {EPISODES}")
    print(f"Device: {DEVICE}")
    print()

    for episode in range(EPISODES):
        game = CatanGame()

        legal_vertices = game.get_legal_settlements(player_id=1, is_setup=True)

        states = [encode_vertex(game, v) for v in legal_vertices]

        if random.random() < epsilon:
            idx = random.randint(0, len(states) - 1)
        else:
            with torch.no_grad():
                states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)
                q_values = model(states_tensor).squeeze()
                idx = torch.argmax(q_values).item()

        chosen_vertex = legal_vertices[idx]
        chosen_state = states[idx]
        
        reward = compute_reward(chosen_state)
        rewards_history.append(reward)

        state_tensor = torch.tensor(chosen_state, dtype=torch.float32).to(DEVICE)
        q_pred = model(state_tensor)
        q_target = torch.tensor([reward], dtype=torch.float32, device=DEVICE)

        loss = loss_fn(q_pred, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % 500 == 0:
            avg_reward = np.mean(rewards_history[-500:]) if rewards_history else 0
            print(f"Epizod {episode:4d} | "
                  f"Avg Reward: {avg_reward:5.2f} | "
                  f"ε={epsilon:.3f} | "
                  f"Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel zapisany w {MODEL_PATH}")
    
    print(f"\nStatystyki:")
    print(f"Średnia nagroda (ostatnie 1000): {np.mean(rewards_history[-1000:]):.2f}")
    print(f"Maksymalna nagroda: {max(rewards_history):.2f}")
    print(f"Minimalna nagroda: {min(rewards_history):.2f}")


if __name__ == "__main__":
    train()