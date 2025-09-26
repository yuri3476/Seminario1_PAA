import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from time import perf_counter

# Formatador adaptativo de tempo (igual lógica usada no Prim)
def format_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds*1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds*1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    return f"{seconds:.4f} s"


class UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_j] += 1
            return True
        return False


def draw_graph_step(G, pos, title, filename,
                    mst_edges,
                    evaluating_edge=None,
                    rejected_edge=None,
                    all_edges_with_weights=None):
    
    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=16, color='white')
    ax = plt.gca()
    ax.set_facecolor('#202B3B')
    plt.axis('off')

  
    nx.draw_networkx_nodes(G, pos, node_color='white', node_size=1000, edgecolors='gray')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')


    if all_edges_with_weights:
        for u, v, data in all_edges_with_weights:
            
            if (u, v, data['weight']) not in [(e[0], e[1], e[2]) for e in mst_edges] and \
               (u, v, data['weight']) != evaluating_edge and \
               (v, u, data['weight']) != evaluating_edge and \
               (u, v, data['weight']) != rejected_edge and \
               (v, u, data['weight']) != rejected_edge:
                
                nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], edge_color='gray', width=1, style='dashed')
                
                
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                plt.text(mid_x, mid_y, str(data['weight']), color='gray', fontsize=10, ha='center', va='center')



    nx.draw_networkx_edges(G, pos, edgelist=mst_edges, edge_color='#66FFFF', width=3)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    
  
    for u, v, weight in mst_edges:
        if G.has_edge(u,v): 
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            plt.text(mid_x, mid_y, str(weight), color='#66FFFF', fontsize=10, ha='center', va='center', bbox=dict(facecolor='#202B3B', edgecolor='none', boxstyle='round,pad=0.3'))


   
    if evaluating_edge:
        u, v, weight = evaluating_edge
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='yellow', width=3)
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        plt.text(mid_x, mid_y, str(weight), color='yellow', fontsize=10, ha='center', va='center', bbox=dict(facecolor='#202B3B', edgecolor='none', boxstyle='round,pad=0.3'))
        


    if rejected_edge:
        u, v, weight = rejected_edge
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='red', width=3, style='dashed')
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        plt.text(mid_x, mid_y, str(weight), color='red', fontsize=10, ha='center', va='center', bbox=dict(facecolor='#202B3B', edgecolor='none', boxstyle='round,pad=0.3'))


    plt.savefig(filename, facecolor='#202B3B')
    plt.close()


def create_kruskal_animation(G, pos, animation_steps, all_edges_for_drawing, interval_ms=2000, algo_time=None):
    """Cria a animação do algoritmo de Kruskal"""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#202B3B')
    fig.patch.set_facecolor('#202B3B')
    ax.axis('off')
    
    def update(frame):
        ax.clear()
        ax.set_facecolor('#202B3B')
        ax.axis('off')
        
        step = animation_steps[frame]
        
        # Título
        ax.set_title(step['title'], fontsize=16, color='white', pad=20)
        
        # Desenhar nós
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='white', node_size=1000, edgecolors='gray')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold', font_color='black')
        
        # Desenhar arestas não selecionadas (cinza tracejado)
        mst_edge_tuples = [(u, v) for u, v, w in step['mst_edges']]
        eval_edge = step['evaluating_edge']
        reject_edge = step['rejected_edge']
        
        for u, v, data in all_edges_for_drawing:
            edge_tuple = (u, v)
            if (edge_tuple not in mst_edge_tuples and 
                (u, v) != (eval_edge[:2] if eval_edge else (None, None)) and
                (v, u) != (eval_edge[:2] if eval_edge else (None, None)) and
                (u, v) != (reject_edge[:2] if reject_edge else (None, None)) and
                (v, u) != (reject_edge[:2] if reject_edge else (None, None))):
                
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], 
                                     edge_color='gray', width=1, style='dashed')
                
                # Peso da aresta
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                ax.text(mid_x, mid_y, str(data['weight']), color='gray', 
                       fontsize=10, ha='center', va='center')
        
        # Desenhar arestas da MST (ciano)
        if step['mst_edges']:
            mst_edge_list = [(u, v) for u, v, w in step['mst_edges']]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=mst_edge_list, 
                                 edge_color='#66FFFF', width=3)
            
            # Pesos das arestas da MST
            for u, v, weight in step['mst_edges']:
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                ax.text(mid_x, mid_y, str(weight), color='#66FFFF', fontsize=12, 
                       ha='center', va='center', weight='bold',
                       bbox=dict(facecolor='#202B3B', edgecolor='none', boxstyle='round,pad=0.3'))
        
        # Desenhar aresta sendo avaliada (amarelo)
        if eval_edge:
            u, v, weight = eval_edge
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], 
                                 edge_color='yellow', width=4)
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            ax.text(mid_x, mid_y, str(weight), color='yellow', fontsize=12, 
                   ha='center', va='center', weight='bold',
                   bbox=dict(facecolor='#202B3B', edgecolor='none', boxstyle='round,pad=0.3'))
        
        # Desenhar aresta rejeitada (vermelho tracejado)
        if reject_edge:
            u, v, weight = reject_edge
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], 
                                 edge_color='red', width=3, style='dashed')
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            ax.text(mid_x, mid_y, str(weight), color='red', fontsize=12, 
                   ha='center', va='center', weight='bold',
                   bbox=dict(facecolor='#202B3B', edgecolor='none', boxstyle='round,pad=0.3'))
        
        # Informação de custo
        info_text = f"Custo atual da MST: R$ {step['mst_cost']}"
        ax.text(0.5, 0.02, info_text, transform=ax.transAxes, ha='center', va='bottom',
               fontsize=14, color='white', weight='bold',
               bbox=dict(boxstyle="round,pad=0.6", facecolor="darkblue", alpha=0.8))
        if algo_time is not None:
            ax.text(0.98, 0.02, f"Tempo alg.: {format_time(algo_time)}", transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=12, color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#444C77", alpha=0.85))
        
        return []
    
    anim = FuncAnimation(fig, update, frames=len(animation_steps), 
                        interval=interval_ms, repeat=False, blit=False)
    return fig, anim


def save_kruskal_frames(fig, G, pos, animation_steps, all_edges_for_drawing, frames_dir, algo_time=None):
    """Salva cada frame da animação como imagem PNG, incluindo tempo do algoritmo se fornecido."""
    for i, step in enumerate(animation_steps):
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#202B3B')
        ax.axis('off')
        
        # Título
        ax.set_title(step['title'], fontsize=16, color='white', pad=20)
        
        # Desenhar nós
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='white', node_size=1000, edgecolors='gray')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold', font_color='black')
        
        # Desenhar arestas não selecionadas (cinza tracejado)
        mst_edge_tuples = [(u, v) for u, v, w in step['mst_edges']]
        eval_edge = step['evaluating_edge']
        reject_edge = step['rejected_edge']
        
        for u, v, data in all_edges_for_drawing:
            edge_tuple = (u, v)
            if (edge_tuple not in mst_edge_tuples and 
                (u, v) != (eval_edge[:2] if eval_edge else (None, None)) and
                (v, u) != (eval_edge[:2] if eval_edge else (None, None)) and
                (u, v) != (reject_edge[:2] if reject_edge else (None, None)) and
                (v, u) != (reject_edge[:2] if reject_edge else (None, None))):
                
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], 
                                     edge_color='gray', width=1, style='dashed')
                
                # Peso da aresta
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                ax.text(mid_x, mid_y, str(data['weight']), color='gray', 
                       fontsize=10, ha='center', va='center')
        
        # Desenhar arestas da MST (ciano)
        if step['mst_edges']:
            mst_edge_list = [(u, v) for u, v, w in step['mst_edges']]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=mst_edge_list, 
                                 edge_color='#66FFFF', width=3)
            
            # Pesos das arestas da MST
            for u, v, weight in step['mst_edges']:
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                ax.text(mid_x, mid_y, str(weight), color='#66FFFF', fontsize=12, 
                       ha='center', va='center', weight='bold',
                       bbox=dict(facecolor='#202B3B', edgecolor='none', boxstyle='round,pad=0.3'))
        
        # Desenhar aresta sendo avaliada (amarelo)
        if eval_edge:
            u, v, weight = eval_edge
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], 
                                 edge_color='yellow', width=4)
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            ax.text(mid_x, mid_y, str(weight), color='yellow', fontsize=12, 
                   ha='center', va='center', weight='bold',
                   bbox=dict(facecolor='#202B3B', edgecolor='none', boxstyle='round,pad=0.3'))
        
        # Desenhar aresta rejeitada (vermelho tracejado)
        if reject_edge:
            u, v, weight = reject_edge
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], 
                                 edge_color='red', width=3, style='dashed')
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            ax.text(mid_x, mid_y, str(weight), color='red', fontsize=12, 
                   ha='center', va='center', weight='bold',
                   bbox=dict(facecolor='#202B3B', edgecolor='none', boxstyle='round,pad=0.3'))
        
        # Informação de custo
        info_text = f"Custo atual da MST: R$ {step['mst_cost']}"
        ax.text(0.5, 0.02, info_text, transform=ax.transAxes, ha='center', va='bottom',
               fontsize=14, color='white', weight='bold',
               bbox=dict(boxstyle="round,pad=0.6", facecolor="darkblue", alpha=0.8))
        if algo_time is not None:
            ax.text(0.98, 0.02, f"Tempo alg.: {format_time(algo_time)}", transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=12, color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#444C77", alpha=0.85))
        
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def kruskal_com_visualizacao(raw_graph_data):
    start_total = perf_counter()
    # Criar diretório para os outputs
    out_dir = os.path.join(os.getcwd(), "kruskal_outputs")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    nodes = set(v1 for _, v1, _ in raw_graph_data) | set(v2 for _, _, v2 in raw_graph_data)
    num_nodes = len(nodes)
    

    G = nx.Graph()
    for _, u, v in raw_graph_data:
        G.add_node(u)
        G.add_node(v)

    all_edges_for_drawing = []
    for weight, u, v in raw_graph_data:
        G.add_edge(u, v, weight=weight)
        all_edges_for_drawing.append((u, v, {'weight': weight}))

    pos = nx.spring_layout(G, seed=42)
    
    # Lista para armazenar os passos da animação
    animation_steps = []
    

    sorted_edges = sorted(raw_graph_data)
    
    print("-- INÍCIO DO ALGORITMO DE KRUSKAL (Com Visualização) --")
    print(f"Total de vértices na rede: {num_nodes}")
    print("Arestas ordenadas por custo (latência):")
    for weight, u, v in sorted_edges:
        print(f"  ({u} - {v}, Custo: {weight})")
    print("-" * 30)
    
    mst_edges = [] 
    mst_cost = 0
    uf = UnionFind(nodes)
    
    # Passo inicial
    step_counter = 0
    animation_steps.append({
        'title': "Passo 0: Grafo Original (Arestas Ordenadas)",
        'mst_edges': [],
        'evaluating_edge': None,
        'rejected_edge': None,
        'step_counter': step_counter,
        'mst_cost': 0
    })
    step_counter += 1

    print("Avaliando cada aresta para adicionar à Árvore Geradora Mínima (AGM)...")
    
    algo_start = perf_counter()
    for weight, u, v in sorted_edges:
        print(f"\n[{step_counter}] Avaliando aresta: '{u}' - '{v}' com custo {weight}")
        
        # Passo: avaliando aresta
        animation_steps.append({
            'title': f"Passo {step_counter}: Avaliando {u}-{v} (Custo {weight})",
            'mst_edges': list(mst_edges),
            'evaluating_edge': (u, v, weight),
            'rejected_edge': None,
            'step_counter': step_counter,
            'mst_cost': mst_cost
        })
        
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            mst_cost += weight
            print(f"  -> Resultado: Vértices '{u}' e '{v}' estavam em componentes diferentes.")
            print(f"     AÇÃO: Adicionar à AGM. Custo atual da AGM: {mst_cost}")
            
            # Passo: aresta aceita
            animation_steps.append({
                'title': f"Passo {step_counter}: {u}-{v} Aceita (Custo {weight}) - Total: R$ {mst_cost}",
                'mst_edges': list(mst_edges),
                'evaluating_edge': None,
                'rejected_edge': None,
                'step_counter': step_counter,
                'mst_cost': mst_cost
            })
        else:
            print(f"  -> Resultado: Vértices '{u}' e '{v}' já estão conectados no mesmo componente.")
            print(f"     AÇÃO: Descartar. Adicionar esta aresta formaria um CICLO.")
         
            # Passo: aresta rejeitada
            animation_steps.append({
                'title': f"Passo {step_counter}: {u}-{v} Rejeitada (Custo {weight}) - Formaria Ciclo!",
                'mst_edges': list(mst_edges),
                'evaluating_edge': None,
                'rejected_edge': (u, v, weight),
                'step_counter': step_counter,
                'mst_cost': mst_cost
            })
        
        step_counter += 1
        
        if len(mst_edges) == num_nodes - 1:
            print("\n--- A ÁRVORE GERADORA MÍNIMA ESTÁ COMPLETA ---")
            break

    # Passo final
    animation_steps.append({
        'title': f"Resultado Final: AGM Completa (Custo Total: R$ {mst_cost})",
        'mst_edges': list(mst_edges),
        'evaluating_edge': None,
        'rejected_edge': None,
        'step_counter': step_counter,
        'mst_cost': mst_cost
    })
    algo_elapsed = perf_counter() - algo_start

    # Criar animação
    print("\nGerando animação...")
    fig, anim = create_kruskal_animation(G, pos, animation_steps, all_edges_for_drawing, algo_time=algo_elapsed)
    
    # Salvar GIF
    gif_path = os.path.join(out_dir, "kruskal_animation.gif")
    print("Salvando GIF...")
    anim.save(gif_path, writer=PillowWriter(fps=0.8))
    print(f"GIF salvo em: {gif_path}")
    
    # Salvar frames individuais
    print("Salvando frames PNG...")
    save_kruskal_frames(fig, G, pos, animation_steps, all_edges_for_drawing, frames_dir, algo_time=algo_elapsed)
    print(f"Frames salvos em: {frames_dir}")

    total_elapsed = perf_counter() - start_total
    print(f"Tempo (apenas algoritmo Kruskal): {format_time(algo_elapsed)}")
    print(f"Tempo total (inclui geração de GIF e frames): {format_time(total_elapsed)}")
    print("\n--- RESULTADO FINAL ---")
    print("A Árvore Geradora Mínima é composta pelas seguintes conexões:")
    for u, v, weight in mst_edges:
        print(f"  Conexão: {u} - {v}  (Latência: {weight}ms)")
    print("-" * 30)
    
    print(f"O custo (latência) total mínimo para conectar todos os servidores é: R$ {mst_cost}")
    
    print(f"\nArquivos gerados:")
    print(f" - GIF: {gif_path}")
    print(f" - Frames PNG: {frames_dir}")

    return mst_edges, mst_cost, algo_elapsed, total_elapsed




# Problema: Rede de fibra óptica conectando bairros
# Vértices = Bairros da cidade (A, B, C, D, E, F)
# Arestas = Rotas possíveis de cabeamento
# Pesos = Custo de instalação (em milhares de reais)
network_graph = [
    (7, 'A', 'B'), (8, 'A', 'C'), (3, 'B', 'C'), (6, 'B', 'D'),
    (4, 'C', 'D'), (3, 'C', 'E'), (2, 'D', 'E'), (5, 'D', 'F'),
    (2, 'E', 'F'),
]

print("="*60)
print("🌐 ALGORITMO DE KRUSKAL - Rede de Fibra Óptica")
print("="*60)
print("Problema: Conectar todos os bairros com custo mínimo")
print("Algoritmo: Kruskal (ordena arestas e evita ciclos)")
print("="*60)

minimum_spanning_tree, total_cost, algo_time, total_time = kruskal_com_visualizacao(network_graph)
print(f"\nResumo de tempos:\n - Tempo algoritmo (Kruskal): {format_time(algo_time)}\n - Tempo total (com visualizações): {format_time(total_time)}")