import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv

from core.game import CatanGame
from core.utils import roll_dice
from agents.early_agent import EarlyAgent
from agents.mid_agent import DQN, encode_state

MODEL_PATH = "models/mid.pt"
RESULTS_PATH = "wyniki/wyniki_mid.csv"

EPISODES = 2000
TURNS_PER_GAME = 50
GAMMA = 0.95
LR = 0.0005

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_last_episode(results_path):
    if not os.path.exists(results_path):
        return 0

    with open(results_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return 0

    return int(lines[-1].split(",")[0])

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

    model = DQN(input_size=19, output_size=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Wczytano model z {MODEL_PATH}")

    early_agent = EarlyAgent(player_id=1)

    rewards_block = []

    print(f"\nStart treningu MID AGENT")
    print(f"Epizody: {EPISODES}")
    print(f"Tury/gra: {TURNS_PER_GAME}")
    print(f"Device: {DEVICE}\n")

    for episode in range(1, EPISODES + 1):
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

        for _ in range(TURNS_PER_GAME):
            dice = roll_dice()

            before = sum(game.players[player_id].resources.values())
            game.distribute_resources(dice)
            after = sum(game.players[player_id].resources.values())

            resource_reward = after - before
            state = encode_state(game, player_id)

            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    q_vals = model(torch.tensor(state, dtype=torch.float32).to(DEVICE))
                    action = torch.argmax(q_vals).item()

            action_reward = 0
            success = False

            if action == 0:
                action_reward = -2

            elif action == 1:
                for e in game.get_legal_roads(player_id):
                    if game.build_road(player_id, e[0], e[1]):
                        action_reward = 10
                        success = True
                        break
                if not success:
                    action_reward = -9

            elif action == 2:
                for v in game.get_legal_settlements(player_id):
                    if game.build_settlement(player_id, v):
                        action_reward = 200
                        success = True
                        break
                if not success:
                    action_reward = -20

            elif action == 3:
                for v in game.get_legal_cities(player_id):
                    if game.build_city(player_id, v):
                        action_reward = 200
                        success = True
                        break
                if not success:
                    action_reward = -15

            elif action == 4:
                if game.trade_bank(player_id):
                    action_reward = 5
                else:
                    action_reward = -3

            if game.players[player_id].points >= 10:
                action_reward += 99999
                total_reward += resource_reward + action_reward
                break

            next_state = encode_state(game, player_id)

            q_pred = model(torch.tensor(state, dtype=torch.float32).to(DEVICE))[action]
            with torch.no_grad():
                q_next = model(torch.tensor(next_state, dtype=torch.float32).to(DEVICE)).max()

            reward = resource_reward + action_reward
            q_target = reward + GAMMA * q_next

            loss = loss_fn(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward

        if not stabilized_run:
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        rewards_block.append(total_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_block)
            global_episode = episode_offset + episode

            print(
                f"Epizod {episode:4d} | "
                f"Avg(100): {avg_reward:7.0f} | "
                f"ε={epsilon:.3f} | "
                f"Punkty: {game.players[1].points}"
            )

            writer.writerow([global_episode, round(avg_reward, 2)])
            results_file.flush()
            rewards_block.clear()

    torch.save(model.state_dict(), MODEL_PATH)
    results_file.close()

    print("\nTrening MID zakończony")
    print(f"Wyniki dopisane do {RESULTS_PATH}")
    print(f"Model zapisany w {MODEL_PATH}")


if __name__ == "__main__":
    train()
