import networkx as nx
import matplotlib.pyplot as plt
import heapq
import pandas as pd

# =======================================
# 3. Алгоритм Дейкстри (найкоротші шляхи)
# =======================================

G = nx.Graph()

# Додаємо українські міста як вузли
cities = [
    "Київ", "Львів", "Одеса", "Харків", "Дніпро",
    "Запоріжжя", "Вінниця", "Івано-Франківськ", "Чернівці", "Полтава", "Миколаїв"
]
G.add_nodes_from(cities)

# Визначаємо логічні транспортні маршрути між містами
edges = [
    ("Київ", "Львів"), ("Київ", "Вінниця"), ("Київ", "Полтава"), ("Київ", "Дніпро"),
    ("Львів", "Івано-Франківськ"), ("Львів", "Чернівці"), ("Львів", "Вінниця"),
    ("Вінниця", "Чернівці"), ("Вінниця", "Одеса"), ("Одеса", "Миколаїв"), ("Одеса", "Дніпро"),
    ("Харків", "Полтава"), ("Харків", "Дніпро"), ("Дніпро", "Запоріжжя"), ("Дніпро", "Полтава"),
    ("Запоріжжя", "Харків"), ("Запоріжжя", "Дніпро"), ("Івано-Франківськ", "Чернівці"),
    ("Київ", "Харків"), ("Львів", "Дніпро"), ("Запоріжжя", "Одеса")
]
G.add_edges_from(edges)

# Додаємо ваги до ребер (відстані між містами в км)
weighted_edges = {
    ("Київ", "Львів"): 540,   ("Київ", "Вінниця"): 268, ("Київ", "Полтава"): 343, ("Київ", "Дніпро"): 477,
    ("Львів", "Івано-Франківськ"): 134, ("Львів", "Чернівці"): 274, ("Львів", "Вінниця"): 365,
    ("Вінниця", "Чернівці"): 245, ("Вінниця", "Одеса"): 421, ("Одеса", "Миколаїв"): 132, ("Одеса", "Дніпро"): 488,
    ("Харків", "Полтава"): 144, ("Харків", "Дніпро"): 218, ("Дніпро", "Запоріжжя"): 85,  ("Дніпро", "Полтава"): 185,
    ("Запоріжжя", "Харків"): 305, ("Запоріжжя", "Дніпро"): 85,  ("Івано-Франківськ", "Чернівці"): 110,
    ("Київ", "Харків"): 471,   ("Львів", "Дніпро"): 832,  ("Запоріжжя", "Одеса"): 598
}

for (u, v), w in weighted_edges.items():
    G[u][v]['weight'] = w

# Встановлюємо реальні координати міст
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


# Додаємо ваги до ребер (відстані між містами в км)

weighted_edges = {
    ("Київ", "Львів"): 540, ("Київ", "Вінниця"): 268, ("Київ", "Полтава"): 343, ("Київ", "Дніпро"): 477,
    ("Львів", "Івано-Франківськ"): 134, ("Львів", "Чернівці"): 274, ("Львів", "Вінниця"): 365,
    ("Вінниця", "Чернівці"): 245, ("Вінниця", "Одеса"): 421, ("Одеса", "Миколаїв"): 132, ("Одеса", "Дніпро"): 488,
    ("Харків", "Полтава"): 144, ("Харків", "Дніпро"): 218, ("Дніпро", "Запоріжжя"): 85, ("Дніпро", "Полтава"): 185,
    ("Запоріжжя", "Харків"): 305, ("Запоріжжя", "Дніпро"): 85, ("Івано-Франківськ", "Чернівці"): 110,
    ("Київ", "Харків"): 471, ("Львів", "Дніпро"): 832, ("Запоріжжя", "Одеса"): 598
}

# Додаємо ваги у граф
for (city1, city2), weight in weighted_edges.items():
    G[city1][city2]['weight'] = weight

# Функція Дейкстри
def dijkstra(graph, start):
    shortest_paths = {node: float('inf') for node in graph.nodes}
    shortest_paths[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_distance + weight

            if distance < shortest_paths[neighbor]:
                shortest_paths[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return shortest_paths

# Обчислення найкоротших шляхів з Києва
start_city = "Київ"
shortest_paths = dijkstra(G, start_city)

# Формування таблиці результатів
df_dijkstra = pd.DataFrame(list(shortest_paths.items()), columns=["Місто", "Найкоротша відстань від Києва (км)"])
df_dijkstra = df_dijkstra.sort_values(by="Найкоротша відстань від Києва (км)")

# Виведення результатів у термінал
print("\nНайкоротші відстані з Києва до інших міст (алгоритм Дейкстри):")
print(df_dijkstra.to_string(index=False))

# === 3. Візуалізація графа ===
plt.figure(figsize=(10, 7))

# Малюємо граф із географічно коректним розташуванням міст
nx.draw(G, city_positions, with_labels=True, node_color='lightblue', node_size=2000, edge_color='gray', font_size=10)
edge_labels = {(u, v): f"{d['weight']} км" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, city_positions, edge_labels=edge_labels, font_size=8)

plt.title("Граф транспортної мережі України (з вагами)")
plt.show()
