"""
prim.py

Script que anima o passo-a-passo do algoritmo de Prim para uma Árvore Geradora Mínima (MST).
Gera:
 - prim_animation.gif
 - frames/frame_1.png, frame_2.png, ...

Como usar:
  python prim.py

Dependências:
  - Python 3.8+
  - matplotlib
  - pillow (para salvar gif via PillowWriter; geralmente já instalado com matplotlib)

Estrutura de saída:
  ./prim_outputs/prim_animation.gif
  ./prim_outputs/frames/frame_*.png
"""

import heapq
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------
# Configuração do grafo (exemplo)
# ---------------------------
# Nós representam bairros A..F
nodes = ["A", "B", "C", "D", "E", "F"]

# Posições para desenho (x, y) - ajustadas para melhor visualização
pos = {
    0: (0.0, 1.0),    # A
    1: (-1.2, 0.3),   # B
    2: (1.2, 0.3),    # C
    3: (-0.7, -0.7),  # D
    4: (0.7, -0.7),   # E
    5: (0.0, -1.5)    # F
}

# Grafo não-direcionado ponderado: lista de adjacência
graph = {
    0: [(1, 4), (2, 2), (5, 7)],
    1: [(0, 4), (2, 1), (3, 5)],
    2: [(0, 2), (1, 1), (3, 8), (4, 10)],
    3: [(1, 5), (2, 8), (4, 2), (5, 6)],
    4: [(2, 10), (3, 2), (5, 3)],
    5: [(0, 7), (3, 6), (4, 3)]
}

# Construir lista única de arestas (u < v) -> peso
unique_edges = {}
for u in graph:
    for v, w in graph[u]:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in unique_edges:
            unique_edges[(a, b)] = w

# ---------------------------
# Prim que registra passos
# ---------------------------
def prim_steps(graph, start=0):
    n = len(graph)
    visited = set()
    heap = []
    heapq.heappush(heap, (0, -1, start))  # (peso, origem, destino)
    selected = []
    steps = []

    while heap and len(visited) < n:
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

        # construir fronteira a partir do heap (deduplicada)
        frontier = []
        seen_front = set()
        for (w, o, d) in heap:
            a, b = (o, d) if o < d else (d, o)
            if (a, b) not in seen_front and (a, b) in unique_edges:
                seen_front.add((a, b))
                frontier.append((a, b, w))

        steps.append({
            "visited": set(visited),
            "selected": list(selected),
            "frontier": list(frontier),
            "picked": (origin, dest, weight)
        })

    return steps

# ---------------------------
# Funções de desenho / animação
# ---------------------------
def build_animation(steps, nodes, pos, unique_edges, fps=1, interval_ms=1200):
    # Preparar figura com mais espaço vertical
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Prim's algorithm — MST building (step-by-step)\nCada aresta mostra seu peso", pad=20)
    
    # Definir limites dos eixos para garantir boa visualização
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.5, 1.8)

    xs = [pos[i][0] for i in range(len(nodes))]
    ys = [pos[i][1] for i in range(len(nodes))]
    scat = ax.scatter(xs, ys, s=60, zorder=3)

    # labels dos nós
    for i, name in enumerate(nodes):
        x, y = pos[i]
        ax.text(x, y + 0.12, name, fontsize=10, ha='center', va='bottom', zorder=4)

    # desenhar todas as arestas finas inicialmente; armazenar handles
    edge_lines = {}
    for (u, v), w in unique_edges.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        line, = ax.plot([x1, x2], [y1, y2], linewidth=1.0, linestyle='solid', zorder=1)
        edge_lines[(u, v)] = line
        # peso no ponto médio com pequeno deslocamento
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = (y2 - y1) * 0.08, -(x2 - x1) * 0.06
        ax.text(mx + dx, my + dy, str(w), fontsize=9, ha='center', va='center', zorder=4)

    # Posicionar o texto informativo na parte inferior da figura
    info_text = ax.text(0.0, -0.22, "", transform=ax.transAxes, ha='center', va='top', fontsize=11, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=1.0, 
                                edgecolor='navy', linewidth=2))

    def update(frame):
        snap = steps[frame]
        visited = snap['visited']
        selected = set((u, v) for (u, v, w) in snap['selected'])
        frontier = set((u, v) for (u, v, w) in snap['frontier'])
        picked = snap['picked']

        sizes = [200 if i in visited else 60 for i in range(len(nodes))]
        scat.set_sizes(sizes)

        for (u, v), line in edge_lines.items():
            if (u, v) in selected:
                line.set_linewidth(3.0)
                line.set_linestyle('solid')
                line.set_zorder(2)
            elif (u, v) in frontier:
                line.set_linewidth(2.0)
                line.set_linestyle('dashed')
                line.set_zorder(1.5)
            else:
                line.set_linewidth(1.0)
                line.set_linestyle('solid')
                line.set_zorder(1.0)

        origin, dest, w = picked
        if origin == -1:
            info = f"Step {frame+1}/{len(steps)} — Start em {nodes[dest]}"
            cost_detail = "Custo MST: 0"
        else:
            info = f"Step {frame+1}/{len(steps)} — Escolheu {nodes[origin]}-{nodes[dest]} (w={w})"
            # Mostrar cada aresta selecionada com seu peso
            if snap['selected']:
                edges_detail = []
                total_cost = 0
                for (u, v, weight) in snap['selected']:
                    edges_detail.append(f"{nodes[u]}-{nodes[v]}({weight})")
                    total_cost += weight
                cost_detail = f"MST: {' + '.join(edges_detail)} = {total_cost}"
            else:
                cost_detail = "Custo MST: 0"
        
        info += f"\n{cost_detail}"
        info_text.set_text(info)
        return []

    anim = FuncAnimation(fig, update, frames=len(steps), interval=interval_ms, blit=False, repeat=False)
    return fig, anim, edge_lines, scat

# ---------------------------
# Main: gerar e salvar arquivos
# ---------------------------
def main():
    out_dir = os.path.join(os.getcwd(), "prim_outputs")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("Executando Prim e registrando passos...")
    steps = prim_steps(graph, start=0)
    print(f"Passos registrados: {len(steps)}")

    print("Construindo animação...")
    fig, anim, edge_lines, scat = build_animation(steps, nodes, pos, unique_edges, fps=1, interval_ms=1200)

    gif_path = os.path.join(out_dir, "prim_animation.gif")

    # Salvar GIF (Pillow)
    try:
        print("Salvando GIF...")
        writer = PillowWriter(fps=1)
        anim.save(gif_path, writer=writer)
        print(f"GIF salvo em: {gif_path}")
        gif_saved = True
    except Exception as e:
        print("Falha ao salvar GIF. Erro:")
        print(e)
        gif_saved = False

    # Salvar frames PNG individuais
    print("Salvando frames PNG individuais...")
    def update_frame_for_save(frame_idx):
        """Atualiza a figura para um frame específico e retorna para salvar"""
        snap = steps[frame_idx]
        visited = snap['visited']
        selected = set((u, v) for (u, v, w) in snap['selected'])
        frontier = set((u, v) for (u, v, w) in snap['frontier'])
        picked = snap['picked']

        # Atualizar tamanhos dos nós
        sizes = [200 if i in visited else 60 for i in range(len(nodes))]
        scat.set_sizes(sizes)

        # Atualizar estilo das arestas
        for (u, v), line in edge_lines.items():
            if (u, v) in selected:
                line.set_linewidth(3.0)
                line.set_linestyle('solid')
                line.set_color('red')
            elif (u, v) in frontier:
                line.set_linewidth(2.0)
                line.set_linestyle('dashed')
                line.set_color('orange')
            else:
                line.set_linewidth(1.0)
                line.set_linestyle('solid')
                line.set_color('black')

        # Atualizar texto informativo
        origin, dest, w = picked
        if origin == -1:
            info = f"Step {frame_idx+1}/{len(steps)} — Start em {nodes[dest]}"
            cost_detail = "Custo MST: 0"
        else:
            info = f"Step {frame_idx+1}/{len(steps)} — Escolheu {nodes[origin]}-{nodes[dest]} (w={w})"
            # Mostrar cada aresta selecionada com seu peso
            if snap['selected']:
                edges_detail = []
                total_cost = 0
                for (u, v, weight) in snap['selected']:
                    edges_detail.append(f"{nodes[u]}-{nodes[v]}({weight})")
                    total_cost += weight
                cost_detail = f"MST: {' + '.join(edges_detail)} = {total_cost}"
            else:
                cost_detail = "Custo MST: 0"
        
        info += f"\n{cost_detail}"
        
        # Encontrar e atualizar o texto informativo na figura
        ax = fig.axes[0]
        for text in ax.texts:
            if (hasattr(text, 'get_transform') and 
                text.get_transform() == ax.transAxes and 
                hasattr(text, 'get_position')):
                pos = text.get_position()
                # O texto informativo tem posição y negativa
                if pos[1] < 0:
                    text.set_text(info)
                    break
        else:
            # Se não encontrou, criar um novo texto
            ax.text(0.0, -0.22, info, transform=ax.transAxes, ha='center', va='top', 
                   fontsize=11, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=1.0, 
                            edgecolor='navy', linewidth=2))

    for i in range(len(steps)):
        update_frame_for_save(i)
        fig.savefig(os.path.join(frames_dir, f"frame_{i+1:02d}.png"), dpi=150, bbox_inches='tight')

    print(f"Frames salvos em: {frames_dir}")

    print("\nResumo de arquivos gerados:")
    if gif_saved:
        print(" - GIF:", gif_path)
    else:
        print(" - GIF: NÃO GERADO (ver mensagens acima).")
    print(" - Frames PNG:", frames_dir)

if __name__ == "__main__":
    main()