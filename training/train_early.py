import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv

from core.game import CatanGame
from agents.early_agent import DQN, encode_vertex

MODEL_PATH = "models/early.pt"
RESULTS_PATH = "wyniki/wyniki_early.csv"

EPISODES = 5000
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

def get_last_episode(results_path):
    if not os.path.exists(results_path):
        return 0

    with open(results_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return 0

    last_line = lines[-1]
    last_episode = int(last_line.split(",")[0])
    return last_episode

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("wyniki", exist_ok=True)

    stabilized_run = (
        os.path.exists(RESULTS_PATH)
        and os.path.getsize(RESULTS_PATH) > 0
    )
    episode_offset = get_last_episode(RESULTS_PATH)

    if stabilized_run:
        epsilon = EPSILON_MIN
    else:
        epsilon = EPSILON_START

    results_file = open(RESULTS_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(results_file)

    model = DQN(input_size=12).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Wczytano model z {MODEL_PATH}")

    rewards_block = []

    print(f"Start treningu | Epizody: {EPISODES} | Device: {DEVICE}\n")

    for episode in range(1, EPISODES + 1):
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

        chosen_state = states[idx]
        reward = compute_reward(chosen_state)
        rewards_block.append(reward)

        state_tensor = torch.tensor(chosen_state, dtype=torch.float32).to(DEVICE)
        q_pred = model(state_tensor)
        q_target = torch.tensor([reward], dtype=torch.float32, device=DEVICE)

        loss = loss_fn(q_pred, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not stabilized_run:
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % 500 == 0:
            avg_reward = np.mean(rewards_block)

            print(
                f"Epizod {episode:4d} | "
                f"Avg(500): {avg_reward:6.3f} | ε={epsilon:.3f}"
            )

            global_episode = episode_offset + episode
            writer.writerow([global_episode, round(avg_reward, 3)])

            results_file.flush()

            rewards_block.clear()

    torch.save(model.state_dict(), MODEL_PATH)
    results_file.close()

    print("\nTrening zakończony")
    print(f"Wyniki dopisane do {RESULTS_PATH}")


if __name__ == "__main__":
    train()
