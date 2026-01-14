from abc import ABC, abstractmethod


class BaseAgent(ABC):

    
    def __init__(self, player_id=1):
        self.player_id = player_id
        self.model = None
    
    @abstractmethod
    def load_model(self, model_path):
        pass
    
    @abstractmethod
    def choose_action(self, game, **kwargs):
        pass
    
    def set_player_id(self, player_id):
        self.player_id = player_id