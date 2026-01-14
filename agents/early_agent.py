import torch
import torch.nn as nn
import numpy as np
from agents.base_agent import BaseAgent
from core.utils import PIPS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_size=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  
        )

    def forward(self, x):
        return self.net(x)


def encode_vertex(game, vertex_id):
    vertex = game.all_vertices[vertex_id]

    resource_pips = {
        "drewno": 0,
        "cegla": 0,
        "owca": 0,
        "zboze": 0,
        "ruda": 0
    }

    neighbor_hexes = [h for h in game.hexes if vertex in h.vertices]
    resources = set()

    for h in neighbor_hexes:
        if h.resource != "pustynia":
            resource_pips[h.resource] += PIPS.get(h.number, 0)
            resources.add(h.resource)

    has_wood_brick = int("drewno" in resources and "cegla" in resources)
    has_wheat_ore = int("zboze" in resources and "ruda" in resources)

    is_on_edge = int(len(neighbor_hexes) < 3)

    state = [
        resource_pips["drewno"],
        resource_pips["cegla"],
        resource_pips["owca"],
        resource_pips["zboze"],
        resource_pips["ruda"],
        len(resources),     
        has_wood_brick,
        has_wheat_ore,
        is_on_edge,
        1, 0, 0              
    ]

    return np.array(state, dtype=np.float32)


class EarlyAgent(BaseAgent):
    
    def __init__(self, model_path="models/early.pt", player_id=1):
        super().__init__(player_id)
        self.model_path = model_path
        self.load_model(model_path)
    
    def load_model(self, model_path):
        self.model = DQN(input_size=12).to(DEVICE)
        
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
            print(f"Early Agent: model wczytany z {model_path}")
        
        except FileNotFoundError:
            print(f"Early Agent: brak modelu {model_path}, używam losowych wag")
            self.model.eval()
    
    def choose_action(self, game, **kwargs):
        return self.choose_vertex(game)
    
    def choose_vertex(self, game):
        legal_vertices = game.get_legal_settlements(self.player_id, is_setup=True)
        
        if not legal_vertices:
            return None
        
        states = np.array([encode_vertex(game, v) for v in legal_vertices])
        
        with torch.no_grad():
            states_tensor = torch.from_numpy(states).to(DEVICE)
            q_values = self.model(states_tensor).squeeze()
            
            if q_values.dim() == 0:
                best_idx = 0
            else:
                best_idx = torch.argmax(q_values).item()
        
        return legal_vertices[best_idx]