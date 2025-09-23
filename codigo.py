import matplotlib.pyplot as plt
import networkx as nx
import os


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


def kruskal_com_visualizacao(raw_graph_data):

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
    

    sorted_edges = sorted(raw_graph_data)
    
    print("--- INÍCIO DO ALGORITMO DE KRUSKAL (Com Visualização) ---")
    print(f"Total de vértices na rede: {num_nodes}")
    print("Arestas ordenadas por custo (latência):")
    for weight, u, v in sorted_edges:
        print(f"  ({u} - {v}, Custo: {weight})")
    print("-" * 30)
    
    mst_edges = [] 
    mst_cost = 0
    uf = UnionFind(nodes)
    
 
    step_counter = 0
    draw_graph_step(G, pos, "Passo 0: Grafo Original (Arestas Ordenadas)", 
                    f"kruskal_step_{step_counter:02d}.png",
                    mst_edges=[],
                    evaluating_edge=None,
                    rejected_edge=None,
                    all_edges_with_weights=all_edges_for_drawing)
    step_counter += 1

    print("Passo 2: Avaliando cada aresta para adicionar à Árvore Geradora Mínima (AGM)...")
    
    for weight, u, v in sorted_edges:
        print(f"\n[{step_counter}] Avaliando aresta: '{u}' - '{v}' com custo {weight}")
        
       
        draw_graph_step(G, pos, f"Passo {step_counter}: Avaliando {u}-{v} (Custo {weight})",
                        f"kruskal_step_{step_counter:02d}.png",
                        mst_edges=mst_edges, 
                        evaluating_edge=(u, v, weight),
                        all_edges_with_weights=all_edges_for_drawing)
        
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            mst_cost += weight
            print(f"  -> Resultado: Vértices '{u}' e '{v}' estavam em componentes diferentes.")
            print(f"     AÇÃO: Adicionar à AGM. Custo atual da AGM: {mst_cost}")
            
            draw_graph_step(G, pos, f"Passo {step_counter}: {u}-{v} Aceita (Custo {weight})",
                            f"kruskal_step_{step_counter:02d}_accepted.png",
                            mst_edges=mst_edges, 
                            evaluating_edge=None,
                            all_edges_with_weights=all_edges_for_drawing)
        else:
            print(f"  -> Resultado: Vértices '{u}' e '{v}' já estão conectados no mesmo componente.")
            print(f"     AÇÃO: Descartar. Adicionar esta aresta formaria um CICLO.")
         
            draw_graph_step(G, pos, f"Passo {step_counter}: {u}-{v} Rejeitada (Custo {weight}) - Ciclo!",
                            f"kruskal_step_{step_counter:02d}_rejected.png",
                            mst_edges=mst_edges, 
                            rejected_edge=(u, v, weight),
                            all_edges_with_weights=all_edges_for_drawing)
        
        step_counter += 1
        
    
        if len(mst_edges) == num_nodes - 1:
            print("\n--- A ÁRVORE GERADORA MÍNIMA ESTÁ COMPLETA ---")
            break


    draw_graph_step(G, pos, f"Resultado Final: AGM (Custo Total: {mst_cost}ms)", 
                    f"kruskal_final_{mst_cost}.png",
                    mst_edges=mst_edges,
                    evaluating_edge=None,
                    rejected_edge=None,
                    all_edges_with_weights=all_edges_for_drawing)

    print("\n--- RESULTADO FINAL ---")
    print("A Árvore Geradora Mínima é composta pelas seguintes conexões:")
    for u, v, weight in mst_edges:
        print(f"  Conexão: {u} - {v}  (Latência: {weight}ms)")
    print("-" * 30)
    

    print(f"O custo (latência) total mínimo para conectar todos os servidores é: {mst_cost}ms")
    
    print(f"\nImagens dos passos geradas na pasta atual como 'kruskal_step_XX.png' e 'kruskal_final_YY.png'")

    return mst_edges, mst_cost




network_graph = [
    (7, 'A', 'B'), (8, 'A', 'C'), (3, 'B', 'C'), (6, 'B', 'D'),
    (4, 'C', 'D'), (3, 'C', 'E'), (2, 'D', 'E'), (5, 'D', 'F'),
    (2, 'E', 'F'),
]


minimum_spanning_tree, total_cost = kruskal_com_visualizacao(network_graph)