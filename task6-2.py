import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import heapq
from matplotlib.lines import Line2D

# ================================
# 2. Реалізація алгоритмів DFS та BFS для пошуку шляхів
# ================================

G = nx.Graph()

# Додаємо українські міста як вузли
cities = [
    "Київ", "Львів", "Одеса", "Харків", "Дніпро",
    "Запоріжжя", "Вінниця", "Івано-Франківськ", "Чернівці", "Полтава", "Миколаїв"
]
G.add_nodes_from(cities)

# Визначаємо логічні транспортні маршрути між містами
realistic_edges = [
    ("Київ", "Львів"), ("Київ", "Вінниця"), ("Київ", "Полтава"), ("Київ", "Дніпро"),
    ("Львів", "Івано-Франківськ"), ("Львів", "Чернівці"), ("Львів", "Вінниця"),
    ("Вінниця", "Чернівці"), ("Вінниця", "Одеса"), ("Одеса", "Миколаїв"), ("Одеса", "Дніпро"),
    ("Харків", "Полтава"), ("Харків", "Дніпро"), ("Дніпро", "Запоріжжя"), ("Дніпро", "Полтава"),
    ("Запоріжжя", "Харків"), ("Запоріжжя", "Дніпро"), ("Івано-Франківськ", "Чернівці"),
    ("Київ", "Харків"), ("Львів", "Дніпро"), ("Запоріжжя", "Одеса")
]
G.add_edges_from(realistic_edges)

# Додаємо ваги до ребер (відстані між містами в км)
weighted_edges = {
    ("Київ", "Львів"): 540, ("Київ", "Вінниця"): 268, ("Київ", "Полтава"): 343, ("Київ", "Дніпро"): 477,
    ("Львів", "Івано-Франківськ"): 134, ("Львів", "Чернівці"): 274, ("Львів", "Вінниця"): 365,
    ("Вінниця", "Чернівці"): 245, ("Вінниця", "Одеса"): 421, ("Одеса", "Миколаїв"): 132, ("Одеса", "Дніпро"): 488,
    ("Харків", "Полтава"): 144, ("Харків", "Дніпро"): 218, ("Дніпро", "Запоріжжя"): 85, ("Дніпро", "Полтава"): 185,
    ("Запоріжжя", "Харків"): 305, ("Запоріжжя", "Дніпро"): 85, ("Івано-Франківськ", "Чернівці"): 110,
    ("Київ", "Харків"): 471, ("Львів", "Дніпро"): 832, ("Запоріжжя", "Одеса"): 598
}
for (city1, city2), weight in weighted_edges.items():
    G[city1][city2]['weight'] = weight

# Встановлюємо реальні координати міст (для фінальної візуалізації)
city_positions = {
    "Київ": (30.5234, 50.4501),
    "Львів": (24.0297, 49.8397),
    "Одеса": (30.7326, 46.4825),
    "Харків": (36.2304, 49.9935),
    "Дніпро": (35.0456, 48.4647),
    "Запоріжжя": (35.1396, 47.8388),
    "Вінниця": (28.4816, 49.2331),
    "Івано-Франківськ": (24.7111, 48.9226),
    "Чернівці": (25.9403, 48.2915),
    "Полтава": (34.5514, 49.5883),
    "Миколаїв": (31.9946, 46.9750)
}


# Функція DFS (пошук у глибину) - реалізовано рекурсивно
def dfs(graph, start, goal, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()
    path = path + [start]
    visited.add(start)
    if start == goal:
        return path
    for neighbor in graph.neighbors(start):
        if neighbor not in visited:
            result = dfs(graph, neighbor, goal, path, visited)
            if result is not None:
                return result
    return None

# Функція BFS (пошук у ширину)
def bfs(graph, start, goal):
    queue = [(start, [start])]
    visited = {start}
    while queue:
        (vertex, path) = queue.pop(0)
        if vertex == goal:
            return path
        for neighbor in graph.neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# Визначаємо початкове та цільове місто для пошуку шляху
start_city = "Київ"
target_city = "Одеса"

dfs_path = dfs(G, start_city, target_city)
bfs_path = bfs(G, start_city, target_city)

print("\nШлях з Києва до Одеси за алгоритмом DFS:")
if dfs_path:
    print(" -> ".join(dfs_path))
else:
    print("Шлях не знайдено.")

print("\nШлях з Києва до Одеси за алгоритмом BFS:")
if bfs_path:
    print(" -> ".join(bfs_path))
else:
    print("Шлях не знайдено.")

# ================================
# Підсвічюємо шляхи DFS та BFS для кращої візуалізації
# ================================

plt.figure(figsize=(10, 7))
# Малюємо вузли та всі ребра графа
nx.draw_networkx_nodes(G, city_positions, node_size=2000, node_color='lightblue')
nx.draw_networkx_labels(G, city_positions, font_size=10)
nx.draw_networkx_edges(G, city_positions, edge_color='gray', width=1)

# Підготовка ребер для підсвічування шляхів
if dfs_path is not None:
    dfs_edges = list(zip(dfs_path, dfs_path[1:]))
    nx.draw_networkx_edges(G, city_positions, edgelist=dfs_edges, edge_color='red', width=3, style='dashed')
if bfs_path is not None:
    bfs_edges = list(zip(bfs_path, bfs_path[1:]))
    nx.draw_networkx_edges(G, city_positions, edgelist=bfs_edges, edge_color='green', width=3, style='dotted')

# Додавання легенди для розрізнення шляхів
legend_elements = [
    Line2D([0], [0], color='red', lw=3, linestyle='dashed', label='DFS шлях'),
    Line2D([0], [0], color='green', lw=3, linestyle='dotted', label='BFS шлях')
]
plt.legend(handles=legend_elements, loc='best')
plt.title("Граф з підсвіченими шляхами: DFS (червоним) та BFS (зеленим)")
plt.show()
