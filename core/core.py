import math
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from core.utils import RES_COLORS, PLAYER_COLORS, HEX_DEFINITIONS


class CatanVisualizer:
    def __init__(self, game):
        self.game = game
        self.hex_centers = {}
        self.vertex_coords = {}
        self.calculate_geometry()
        
        plt.ion()

    def calculate_geometry(self):
        layout = [
            (3, 0),      
            (4, -0.5),   
            (5, -1.0),  
            (4, -0.5),  
            (3, 0)     
        ]

        radius = 1.0
        h_w = math.sqrt(3) * radius  
        v_dist = 1.5 * radius    

        hex_id_counter = 0
        current_y = 4 * v_dist

        for row_idx, (count, x_offset) in enumerate(layout):
            start_x = x_offset * h_w
            
            for col in range(count):
                center_x = start_x + (col * h_w)
                center_y = current_y
                self.hex_centers[hex_id_counter] = (center_x, center_y)

                v_ids = HEX_DEFINITIONS[hex_id_counter]

                for i in range(6):
                    angle = 90 - (i * 60)  
                    rad = math.radians(angle)
                    vx = center_x + radius * math.cos(rad)
                    vy = center_y + radius * math.sin(rad)

                    if v_ids[i] not in self.vertex_coords:
                        self.vertex_coords[v_ids[i]] = (vx, vy)

                hex_id_counter += 1
            
            current_y -= v_dist

    def draw_board(self):
        plt.clf()
        fig = plt.gcf()
        ax = plt.gca()

        ax.set_aspect('equal')
        ax.axis('off')

        for h in self.game.hexes:
            center = self.hex_centers[h.id]
            color = RES_COLORS.get(h.resource, 'white')

            poly = RegularPolygon(
                center,
                6,
                radius=1.0,
                orientation=0,
                facecolor=color,
                edgecolor='white',
                alpha=0.9
            )
            ax.add_patch(poly)

            if h.number != 7:
                ax.text(
                    center[0],
                    center[1],
                    str(h.number),
                    ha='center',
                    va='center',
                    fontsize=11,
                    fontweight='bold',
                    bbox=dict(boxstyle="circle,pad=0.3", fc="white", alpha=0.6)
                )

        for edge in self.game.edges.values():
            if edge.owner != 0:
                v1 = self.vertex_coords[edge.v1_id]
                v2 = self.vertex_coords[edge.v2_id]
                plt.plot(
                    [v1[0], v2[0]],
                    [v1[1], v2[1]],
                    linewidth=5,
                    color=PLAYER_COLORS[edge.owner],
                    zorder=2
                )

        for v in self.game.all_vertices:
            if v.owner != 0:
                x, y = self.vertex_coords[v.id]
                color = PLAYER_COLORS[v.owner]

                if v.type == 'city':
                    ax.plot(
                        x, y,
                        marker='s',
                        markersize=16,
                        color=color,
                        markeredgecolor='white',
                        markeredgewidth=2,
                        zorder=3
                    )
                else:
                    ax.plot(
                        x, y,
                        marker='o',
                        markersize=12,
                        color=color,
                        markeredgecolor='white',
                        markeredgewidth=2,
                        zorder=3
                    )

        self._draw_player_info(ax)

        plt.pause(0.001)

    def _draw_player_info(self, ax):
        p1 = self.game.players[1]
        p2 = self.game.players[2]

        res1 = ", ".join([f"{k[:3]}:{v}" for k, v in p1.resources.items()])
        res2 = ", ".join([f"{k[:3]}:{v}" for k, v in p2.resources.items()])

        info_text = (
            f"GRACZ 1 | Punkty: {p1.points}\n"
            f"   {res1}\n\n"
            f"GRACZ 2 | Punkty: {p2.points}\n"
            f"   {res2}"
        )

        ax.text(
            0.02, 0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            va='top',
            ha='left',
            family='monospace',
            bbox=dict(boxstyle="round", fc="white", alpha=0.9)
        )

    def show_final(self):
        plt.ioff()
        self.draw_board()
        plt.show()

    def close(self):
        plt.close()