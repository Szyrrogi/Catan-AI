import time
from core.game import CatanGame
from core.visualizer import CatanVisualizer
from core.utils import roll_dice
from agents.early_agent import EarlyAgent
from agents.mid_agent import MidAgent

MAX_TURNS = 500
REFRESH_EVERY = 1   
SLEEP_TIME = 0.2     
WIN_POINTS = 10      


def play_game():
    game = CatanGame()
    vis = CatanVisualizer(game)
    
    early_p1 = EarlyAgent(player_id=1)
    early_p2 = EarlyAgent(player_id=2)
    
    mid_p1 = MidAgent(player_id=1)
    mid_p2 = MidAgent(player_id=2)

    for pid, early_agent in [(1, early_p1), (2, early_p2)]:
        for _ in range(2):
            v = early_agent.choose_vertex(game)
            
            if v is None:
                continue
            
            game.build_settlement(pid, v, is_setup=True)
            
            for neighbor_id in game.all_vertices[v].neighbors:
                if game.build_road(pid, v, neighbor_id, is_setup=True):
                    break
    
    vis.draw_board()
    time.sleep(1.0)
    
    for turn in range(1, MAX_TURNS):
        dice_roll = roll_dice()
        game.distribute_resources(dice_roll)
        
        for pid, mid_agent in [(1, mid_p1), (2, mid_p2)]:
            action = mid_agent.choose_action(game)
            mid_agent.execute_action(game, action)
        
        if turn % REFRESH_EVERY == 0:
            vis.draw_board()
            time.sleep(SLEEP_TIME)
        
        winner = game.get_winner()
        if winner is not None:
            print(f"\nWYGRAŁ GRACZ {winner} w turze {turn}!")
            print(f"Gracz 1: {game.players[1].points} pkt")
            print(f"Gracz 2: {game.players[2].points} pkt")
            
            vis.draw_board()
            vis.show_final() 
            return

    print("\nKoniec czasu - Remis")
    print(f"Gracz 1: {game.players[1].points} pkt")
    print(f"Gracz 2: {game.players[2].points} pkt")
    
    vis.draw_board()
    vis.show_final()


if __name__ == "__main__":
    play_game()