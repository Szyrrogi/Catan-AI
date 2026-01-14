import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from core.game import CatanGame
from core.utils import roll_dice
from agents.early_agent import EarlyAgent
from agents.mid_agent import DQN, encode_state

MODEL_PATH = "models/mid.pt"
EPISODES = 3000
TURNS_PER_GAME = 50
GAMMA = 0.95
LR = 0.0005
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    os.makedirs("models", exist_ok=True)
    
    model = DQN(input_size=19, output_size=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✓ Wczytano model z {MODEL_PATH}")

    early_agent = EarlyAgent(player_id=1)

    epsilon = EPSILON_START
    rewards_history = []

    print(f"Start treningu Mid Agent")
    print(f"Epizody: {EPISODES}")
    print(f"Tury/gra: {TURNS_PER_GAME}")
    print(f"Device: {DEVICE}")
    print()

    for episode in range(EPISODES):
        game = CatanGame()

        for pid in [1, 2]:
            early_agent.set_player_id(pid)
            
            for _ in range(2):  
                v = early_agent.choose_vertex(game)
                if v is None:
                    continue
                    
                game.build_settlement(pid, v, is_setup=True)
                
                for n in game.all_vertices[v].neighbors:
                    if game.build_road(pid, v, n, is_setup=True):
                        break

        total_reward = 0
        player_id = 1  

        for turn in range(TURNS_PER_GAME):
            dice_roll = roll_dice()
            
            before = sum(game.players[player_id].resources.values())
            game.distribute_resources(dice_roll)
            after = sum(game.players[player_id].resources.values())
            
            resource_reward = (after - before) * 1

            state = encode_state(game, player_id)

            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float32).to(DEVICE))
                    action = torch.argmax(q_values).item()

            action_reward = 0
            action_success = False

            if action == 0:
                action_reward = -2

            elif action == 1:
                for edge_key in game.get_legal_roads(player_id):
                    if game.build_road(player_id, edge_key[0], edge_key[1]):
                        action_reward = 10
                        action_success = True
                        break
                if not action_success:
                    action_reward = -9

            elif action == 2:
                for v_id in game.get_legal_settlements(player_id):
                    if game.build_settlement(player_id, v_id):
                        action_reward = 200
                        action_success = True
                        break
                if not action_success:
                    action_reward = -20

            elif action == 3:
                for v_id in game.get_legal_cities(player_id):
                    if game.build_city(player_id, v_id):
                        action_reward = 200
                        action_success = True
                        break
                if not action_success:
                    action_reward = -15

            elif action == 4:
                if game.trade_bank(player_id):
                    action_reward = 5
                    action_success = True
                else:
                    action_reward = -3

            if game.players[player_id].points >= 10:
                action_reward += 99999
                total_reward += resource_reward + action_reward
                break

            next_state = encode_state(game, player_id)

            state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            q_pred = model(state_tensor)[action]

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                q_next = model(next_state_tensor).max()

            reward = resource_reward + action_reward
            q_target = reward + GAMMA * q_next

            loss = loss_fn(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        rewards_history.append(total_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"Epizod {episode:4d} | "
                  f"Avg Reward: {avg_reward:7.0f} | "
                  f"ε={epsilon:.3f} | "
                  f"Punkty: {game.players[1].points}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel zapisany w {MODEL_PATH}")
    
    print(f"\nStatystyki:")
    print(f"Średnia nagroda (ostatnie 500): {np.mean(rewards_history[-500:]):.0f}")
    print(f"Maksymalna nagroda: {max(rewards_history):.0f}")
    print(f"Minimalna nagroda: {min(rewards_history):.0f}")


if __name__ == "__main__":
    train()