import torch
import torch.nn as nn
import numpy as np
from agents.base_agent import BaseAgent
from core.utils import PIPS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_NOTHING = 0
ACTION_ROAD = 1
ACTION_SETTLEMENT = 2
ACTION_CITY = 3
ACTION_TRADE = 4


class DQN(nn.Module):

    def __init__(self, input_size=19, output_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)  
        )

    def forward(self, x):
        return self.net(x)


def encode_state(game, player_id):
    player = game.players[player_id]

    resources = np.array([
        player.resources["drewno"],
        player.resources["cegla"],
        player.resources["owca"],
        player.resources["zboze"],
        player.resources["ruda"]
    ], dtype=np.float32) / 10.0

    settlements = [v for v in game.all_vertices 
                   if v.owner == player_id and v.type == "settlement"]
    cities = [v for v in game.all_vertices 
              if v.owner == player_id and v.type == "city"]
    roads = [e for e in game.edges.values() if e.owner == player_id]

    possible_settlements = len(game.get_legal_settlements(player_id))
    possible_roads = len(game.get_legal_roads(player_id))
    possible_cities = len(game.get_legal_cities(player_id))

    structures = np.array([
        len(settlements),
        len(cities),
        len(roads),
        possible_settlements,
        possible_roads,
        possible_cities
    ], dtype=np.float32) / 10.0

    production = {
        "drewno": 0,
        "cegla": 0,
        "owca": 0,
        "zboze": 0,
        "ruda": 0
    }

    for h in game.hexes:
        if h.resource == "pustynia":
            continue

        for v in h.vertices:
            if v.owner == player_id:
                multiplier = 2 if v.type == "city" else 1
                production[h.resource] += PIPS.get(h.number, 0) * multiplier

    prod_array = np.array([
        production["drewno"],
        production["cegla"],
        production["owca"],
        production["zboze"],
        production["ruda"]
    ], dtype=np.float32) / 10.0

    era = np.array([0, 1, 0], dtype=np.float32)  

    return np.concatenate([resources, structures, prod_array, era])


class MidAgent(BaseAgent):
    def __init__(self, model_path="models/mid.pt", player_id=1):
        super().__init__(player_id)
        self.model_path = model_path
        self.load_model(model_path)
    
    def load_model(self, model_path):
        self.model = DQN(input_size=19, output_size=5).to(DEVICE)
        
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"Mid Agent: model wczytany z {model_path}")
        
        except FileNotFoundError:
            print(f"Mid Agent: brak modelu {model_path}, używam losowych wag")
            self.model.eval()
    
    def choose_action(self, game, **kwargs):
        state = encode_state(game, self.player_id)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()
            action = torch.argmax(q_values).item()
        
        return action
    
    def execute_action(self, game, action):
        pid = self.player_id
        
        if action == ACTION_NOTHING:
            return True
        
        elif action == ACTION_ROAD:
            legal_roads = game.get_legal_roads(pid)
            for v1, v2 in legal_roads:
                if game.build_road(pid, v1, v2):
                    return True
            return False
        
        elif action == ACTION_SETTLEMENT:
            legal_settlements = game.get_legal_settlements(pid)
            for v_id in legal_settlements:
                if game.build_settlement(pid, v_id):
                    return True
            return False
        
        elif action == ACTION_CITY:
            legal_cities = game.get_legal_cities(pid)
            for v_id in legal_cities:
                if game.build_city(pid, v_id):
                    return True
            return False
        
        elif action == ACTION_TRADE:
            return game.trade_bank(pid)
        
        return False