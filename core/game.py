import random
from core.utils import RESOURCES, NUMBERS, HEX_DEFINITIONS, EDGE_DEFINITIONS, COSTS


class Vertex:

    def __init__(self, v_id):
        self.id = v_id
        self.owner = 0 
        self.type = None 
        self.neighbors = set() 

    def __repr__(self):
        return f"V{self.id}"


class Edge:

    def __init__(self, v1_id, v2_id):
        self.v1_id = v1_id
        self.v2_id = v2_id
        self.owner = 0  
        
    def __repr__(self):
        return f"Edge({self.v1_id}-{self.v2_id})"
    
    def is_connected_to_player(self, player_id, all_edges, all_vertices):
        if (all_vertices[self.v1_id].owner == player_id or 
            all_vertices[self.v2_id].owner == player_id):
            return True
        
        for edge in all_edges.values():
            if edge == self:
                continue
            if edge.owner == player_id:
                if (edge.v1_id in [self.v1_id, self.v2_id] or 
                    edge.v2_id in [self.v1_id, self.v2_id]):
                    return True
        
        return False


class Hex:
    def __init__(self, h_id, resource, number, vertices_refs):
        self.id = h_id
        self.resource = resource  
        self.number = number 
        self.vertices = vertices_refs 


class Player:
    def __init__(self, p_id):
        self.id = p_id
        self.resources = {r: 0 for r in RESOURCES if r != 'pustynia'}
        self.points = 0
    
    def can_afford(self, cost):
        for r, amount in cost.items():
            if self.resources.get(r, 0) < amount:
                return False
        return True

    def pay(self, cost):
        for r, amount in cost.items():
            self.resources[r] -= amount

    def add_resources(self, resources_dict):
        for r, amount in resources_dict.items():
            if r in self.resources:
                self.resources[r] += amount


class CatanGame:
    def __init__(self):
        self.all_vertices = [Vertex(i) for i in range(54)]
        self.hexes = []
        self.edges = {}  
        self.players = {1: Player(1), 2: Player(2)}
        
        self.generate_map()
        self.calculate_neighbors_and_edges()

    def generate_map(self):
        tiles = (['drewno']*4 + ['cegla']*3 + ['owca']*4 + 
                 ['zboze']*4 + ['ruda']*3 + ['pustynia'])
        random.shuffle(tiles)
        
        nums = NUMBERS[:]
        random.shuffle(nums)
        num_iter = iter(nums)
        
        self.hexes = []
        for h_id in range(19):
            res = tiles[h_id]
            val = 7 if res == 'pustynia' else next(num_iter)
            v_ids = HEX_DEFINITIONS[h_id]
            v_refs = [self.all_vertices[vid] for vid in v_ids]
            self.hexes.append(Hex(h_id, res, val, v_refs))

    def calculate_neighbors_and_edges(self):
        all_edges = set()
        for h_id, edges_list in EDGE_DEFINITIONS.items():
            for edge_tuple in edges_list:

                normalized = tuple(sorted(edge_tuple))
                all_edges.add(normalized)
        
        for edge_tuple in all_edges:
            self.edges[edge_tuple] = Edge(edge_tuple[0], edge_tuple[1])
        
        for edge_tuple in all_edges:
            v1_id, v2_id = edge_tuple
            self.all_vertices[v1_id].neighbors.add(v2_id)
            self.all_vertices[v2_id].neighbors.add(v1_id)


    def distribute_resources(self, roll):
        if roll == 7:
            return {1: 0, 2: 0}
        
        active_hexes = [h for h in self.hexes if h.number == roll]
        total_loot = {1: 0, 2: 0}
        
        for h in active_hexes:
            if h.resource == 'pustynia':
                continue
                
            for v in h.vertices:
                if v.owner != 0:
                    amount = 2 if v.type == 'city' else 1
                    self.players[v.owner].resources[h.resource] += amount
                    total_loot[v.owner] += amount
        
        return total_loot

    def check_build_settlement(self, player_id, v_id, is_setup=False):

        vertex = self.all_vertices[v_id]
        
        if vertex.owner != 0:
            return False
        
        for n_id in vertex.neighbors:
            if self.all_vertices[n_id].owner != 0:
                return False
        
        if not is_setup:
            has_road = False
            for n_id in vertex.neighbors:
                edge_key = tuple(sorted((v_id, n_id)))
                if edge_key in self.edges and self.edges[edge_key].owner == player_id:
                    has_road = True
                    break
            if not has_road:
                return False
        
        return True

    def check_build_city(self, player_id, v_id):
        vertex = self.all_vertices[v_id]
        
        if vertex.owner != player_id:
            return False
        if vertex.type != 'settlement':
            return False
        
        return True

    def check_build_road(self, player_id, v1, v2):
        edge_key = tuple(sorted((v1, v2)))
        
        if edge_key not in self.edges:
            return False
        
        edge = self.edges[edge_key]
        
        if edge.owner != 0:
            return False
        
        return edge.is_connected_to_player(player_id, self.edges, self.all_vertices)


    def build_settlement(self, player_id, v_id, is_setup=False):
        if not self.check_build_settlement(player_id, v_id, is_setup):
            return False
        
        player = self.players[player_id]
        
        if not is_setup:
            if not player.can_afford(COSTS['settlement']):
                return False
            player.pay(COSTS['settlement'])
        
        self.all_vertices[v_id].owner = player_id
        self.all_vertices[v_id].type = 'settlement'
        player.points += 1
        return True

    def build_city(self, player_id, v_id):
        if not self.check_build_city(player_id, v_id):
            return False
        
        player = self.players[player_id]
        
        if not player.can_afford(COSTS['city']):
            return False
        
        player.pay(COSTS['city'])
        
        self.all_vertices[v_id].type = 'city'
        player.points += 1  
        return True

    def build_road(self, player_id, v1, v2, is_setup=False):
        edge_key = tuple(sorted((v1, v2)))
        
        if edge_key not in self.edges:
            return False
        
        edge = self.edges[edge_key]
        
        if edge.owner != 0:
            return False
        
        if not edge.is_connected_to_player(player_id, self.edges, self.all_vertices):
            return False
        
        player = self.players[player_id]
        
        if not is_setup:
            if not player.can_afford(COSTS['road']):
                return False
            player.pay(COSTS['road'])
        
        edge.owner = player_id
        return True



    def trade_bank(self, player_id, give_resource=None, get_resource=None):
        player = self.players[player_id]
        
        if give_resource is None:
            for res, count in player.resources.items():
                if count >= 4:
                    give_resource = res
                    break
        
        if give_resource is None or player.resources.get(give_resource, 0) < 4:
            return False
        
        if get_resource is None:
            available = [r for r in RESOURCES if r != 'pustynia' and r != give_resource]
            get_resource = random.choice(available)
        
        player.resources[give_resource] -= 4
        player.resources[get_resource] += 1
        return True

    def get_legal_settlements(self, player_id, is_setup=False):
        return [
            v.id for v in self.all_vertices
            if self.check_build_settlement(player_id, v.id, is_setup)
        ]

    def get_legal_roads(self, player_id):
        legal = []
        for edge_key, edge in self.edges.items():
            if edge.owner == 0:
                if edge.is_connected_to_player(player_id, self.edges, self.all_vertices):
                    legal.append(edge_key)
        return legal

    def get_legal_cities(self, player_id):
        return [
            v.id for v in self.all_vertices
            if self.check_build_city(player_id, v.id)
        ]

    def get_winner(self):
        for pid, player in self.players.items():
            if player.points >= 10:
                return pid
        return None