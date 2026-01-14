import numpy as np


RESOURCES = ['drewno', 'cegla', 'owca', 'zboze', 'ruda', 'pustynia']
NUMBERS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

PIPS = {
    2: 1, 12: 1,
    3: 2, 11: 2,
    4: 3, 10: 3,
    5: 4, 9: 4,
    6: 5, 8: 5,
    7: 0  
}

RES_COLORS = {
    'drewno': '#228B22',
    'cegla': '#B22222',
    'owca': '#9ACD32',
    'zboze': '#FFD700',
    'ruda': '#708090',
    'pustynia': '#F4A460'
}

PLAYER_COLORS = {
    0: 'black',
    1: 'blue',
    2: 'red'
}


HEX_DEFINITIONS = {
    0: [1, 2, 10, 9, 8, 0],
    1: [3, 4, 12, 11, 10, 2],
    2: [5, 6, 14, 13, 12, 4],
    
    3: [8, 9, 19, 18, 17, 7],
    4: [10, 11, 21, 20, 19, 9],
    5: [12, 13, 23, 22, 21, 11],
    6: [14, 15, 25, 24, 23, 13],
    
    7: [17, 18, 29, 28, 27, 16],
    8: [19, 20, 31, 30, 29, 18],
    9: [21, 22, 33, 32, 31, 20],
    10: [23, 24, 35, 34, 33, 22],
    11: [25, 26, 37, 36, 35, 24],
    
    12: [29, 30, 40, 39, 38, 28],
    13: [31, 32, 42, 41, 40, 30],
    14: [33, 34, 44, 43, 42, 32],
    15: [35, 36, 46, 45, 44, 34],
    
    16: [40, 41, 49, 48, 47, 39],
    17: [42, 43, 51, 50, 49, 41],
    18: [44, 45, 53, 52, 51, 43]
}

EDGE_DEFINITIONS = {
    0: [(1,2), (2,10), (10,9), (9,8), (8,0), (0,1)],
    1: [(3,4), (4,12), (12,11), (11,10), (10,2), (2,3)],
    2: [(5,6), (6,14), (14,13), (13,12), (12,4), (4,5)],
    
    3: [(8,9), (9,19), (19,18), (18,17), (17,7), (7,8)],
    4: [(10,11), (11,21), (21,20), (20,19), (19,9), (9,10)],
    5: [(12,13), (13,23), (23,22), (22,21), (21,11), (11,12)],
    6: [(14,15), (15,25), (25,24), (24,23), (23,13), (13,14)],
    
    7: [(17,18), (18,29), (29,28), (28,27), (27,16), (16,17)],
    8: [(19,20), (20,31), (31,30), (30,29), (29,18), (18,19)],
    9: [(21,22), (22,33), (33,32), (32,31), (31,20), (20,21)],
    10: [(23,24), (24,35), (35,34), (34,33), (33,22), (22,23)],
    11: [(25,26), (26,37), (37,36), (36,35), (35,24), (24,25)],
    
    12: [(29,30), (30,40), (40,39), (39,38), (38,28), (28,29)],
    13: [(31,32), (32,42), (42,41), (41,40), (40,30), (30,31)],
    14: [(33,34), (34,44), (44,43), (43,42), (42,32), (32,33)],
    15: [(35,36), (36,46), (46,45), (45,44), (44,34), (34,35)],
    
    16: [(40,41), (41,49), (49,48), (48,47), (47,39), (39,40)],
    17: [(42,43), (43,51), (51,50), (50,49), (49,41), (41,42)],
    18: [(44,45), (45,53), (53,52), (52,51), (51,43), (43,44)]
}

COSTS = {
    'road': {'drewno': 1, 'cegla': 1},
    'settlement': {'drewno': 1, 'cegla': 1, 'owca': 1, 'zboze': 1},
    'city': {'ruda': 3, 'zboze': 2}
}

def get_vertex_resources(game, vertex_id):
    vertex = game.all_vertices[vertex_id]
    resources = []
    
    for h in game.hexes:
        if vertex in h.vertices and h.resource != 'pustynia':
            resources.append(h.resource)
    
    return resources


def get_vertex_production(game, vertex_id):
    vertex = game.all_vertices[vertex_id]
    production = 0
    
    for h in game.hexes:
        if vertex in h.vertices and h.resource != 'pustynia':
            production += PIPS.get(h.number, 0)
    
    return production


def get_vertex_diversity(game, vertex_id):
    resources = get_vertex_resources(game, vertex_id)
    return len(set(resources))


def has_synergy(resources, synergy_type):
    resource_set = set(resources)
    
    if synergy_type == 'road':
        return 'drewno' in resource_set and 'cegla' in resource_set
    elif synergy_type == 'city':
        return 'zboze' in resource_set and 'ruda' in resource_set
    
    return False


def roll_dice():
    import random
    return random.randint(1, 6) + random.randint(1, 6)