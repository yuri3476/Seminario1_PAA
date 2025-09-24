import heapq
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Problema: Empresa de internet conectando bairros com fibra óptica
# Nós = Bairros da cidade
# Arestas = Rotas possíveis de cabeamento 
# Pesos = Custo de instalação 
nodes = ["A", "B", "C", "D", "E", "F"]  # Bairros da cidade

pos = {
    0: (0.0, 1.0),    # A
    1: (-1.2, 0.3),   # B
    2: (1.2, 0.3),    # C
    3: (-0.7, -0.7),  # D
    4: (0.7, -0.7),   # E
    5: (0.0, -1.5)    # F
}

graph = {
    0: [(1, 4), (2, 2), (5, 7)],
    1: [(0, 4), (2, 1), (3, 5)],
    2: [(0, 2), (1, 1), (3, 8), (4, 10)],
    3: [(1, 5), (2, 8), (4, 2), (5, 6)],
    4: [(2, 10), (3, 2), (5, 3)],
    5: [(0, 7), (3, 6), (4, 3)]
}

# Lista única de arestas
unique_edges = {}
for u in graph:
    for v, w in graph[u]:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in unique_edges:
            unique_edges[(a, b)] = w

# Passo a passo do Algoritmo de Prim
def prim_steps(graph, start=0):
    visited = set()
    heap = []
    heapq.heappush(heap, (0, -1, start))
    selected = []
    steps = []

    while heap and len(visited) < len(graph):
        weight, origin, dest = heapq.heappop(heap)
        if dest in visited:
            continue

        visited.add(dest)
        if origin != -1:
            sel = (min(origin, dest), max(origin, dest), weight)
            selected.append(sel)

        for nbr, w in graph[dest]:
            if nbr not in visited:
                heapq.heappush(heap, (w, dest, nbr))

        frontier = []
        seen = set()
        for (w, o, d) in heap:
            a, b = (o, d) if o < d else (d, o)
            if (a, b) not in seen and (a, b) in unique_edges:
                seen.add((a, b))
                frontier.append((a, b, w))

        steps.append({
            "visited": set(visited),
            "selected": list(selected),
            "frontier": list(frontier),
            "picked": (origin, dest, weight)
        })
    return steps

# GIF
def build_animation(steps, nodes, pos, unique_edges, interval_ms=1500):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.2, 1.6)
    ax.set_title("Algoritmo de Prim — Rede de Fibra Óptica\nConectando bairros com custo mínimo", fontsize=14, pad=12)

    xs = [pos[i][0] for i in range(len(nodes))]
    ys = [pos[i][1] for i in range(len(nodes))]
    scat = ax.scatter(xs, ys, s=60, zorder=3)

    # labels dos nós
    for i, name in enumerate(nodes):
        x, y = pos[i]
        ax.text(x, y + 0.12, name, fontsize=10, ha='center', va='bottom', zorder=4)

    edge_lines = {}
    for (u, v), w in unique_edges.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        line, = ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.0, zorder=1)
        edge_lines[(u, v)] = line
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(mx, my, str(w), fontsize=8, ha='center', va='center', color="blue")

    # Caixa de informações
    info_text = ax.text(0.5, -0.08, "", transform=ax.transAxes,
                        ha="center", va="top", fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.6",
                                  facecolor="lightyellow",
                                  edgecolor="darkorange", linewidth=2))

    def update(frame):
        snap = steps[frame]
        visited = snap["visited"]
        selected = set((u, v) for (u, v, w) in snap["selected"])
        frontier = set((u, v) for (u, v, w) in snap["frontier"])
        picked = snap["picked"]

        sizes = [200 if i in visited else 60 for i in range(len(nodes))]
        scat.set_sizes(sizes)

        for (u, v), line in edge_lines.items():
            if (u, v) in selected:
                line.set_linewidth(3.0)
                line.set_color("red")
            elif (u, v) in frontier:
                line.set_linewidth(2.0)
                line.set_color("orange")
                line.set_linestyle("dashed")
            else:
                line.set_linewidth(1.0)
                line.set_color("black")
                line.set_linestyle("solid")

        origin, dest, w = picked
        if origin == -1:
            msg = f"Step {frame+1}/{len(steps)} — Iniciando instalação no bairro {nodes[dest]}"
            cost_msg = "Custo total: R$ 0"
        else:
            msg = f"Step {frame+1}/{len(steps)} — Instalando cabo {nodes[origin]}-{nodes[dest]} (R$ {w})"
            edges_detail = []
            total = 0
            for (u, v, ww) in snap["selected"]:
                edges_detail.append(f"{nodes[u]}-{nodes[v]}({ww})")
                total += ww
            cost_msg = f"Rede: {' + '.join(edges_detail)} = R$ {total}"

        info_text.set_text(msg + "\n" + cost_msg)
        return []

    anim = FuncAnimation(fig, update, frames=len(steps),
                         interval=interval_ms, repeat=False)
    return fig, anim

# Principal
def main():
    out_dir = os.path.join(os.getcwd(), "prim_outputs")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    steps = prim_steps(graph, start=0)
    fig, anim = build_animation(steps, nodes, pos, unique_edges)

    gif_path = os.path.join(out_dir, "prim_animation.gif")
    print("Gerando GIF...")
    anim.save(gif_path, writer=PillowWriter(fps=1))
    print("GIF salvo em:", gif_path)

    print("Gerando frames PNG...")
    # Criar uma nova animação para salvar os frames individualmente
    from matplotlib.animation import FuncAnimation
    
    def save_frame(frame_num):
        # Atualiza o frame
        anim.frame_seq = iter([frame_num])
        anim._draw_frame(frame_num)
        fig.savefig(os.path.join(frames_dir, f"frame_{frame_num+1:02d}.png"),
                    dpi=150, bbox_inches="tight")
    
    for i in range(len(steps)):
        save_frame(i)
    
    print("Frames salvos em:", frames_dir)

if __name__ == "__main__":
    main()
