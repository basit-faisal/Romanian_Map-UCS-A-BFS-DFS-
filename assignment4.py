#2021126
#Basit faisal
#B31 Data-Science

import matplotlib.pyplot as plt
import time
from collections import deque
import heapq


def min_index(a,b):
    temp_val = 99999999
    index = -1
    for x in range(len(a)):
        if temp_val > a[x]:
            temp_val = a[x]
            index = x
    
    return temp_val,b[index]

def final_cost(dict1_map, dict2_heuristic, key, prev_cost_path, goal, path):
    sum = 0
    try:
        if key == goal:
            if key not in path:
                path.append(key)
                return prev_cost_path, path
        
        if key in dict1_map and key in dict2_heuristic:
            sum = dict1_map[key][goal] + dict2_heuristic[goal] + prev_cost_path
            
    except KeyError:
        if key != goal:
            if key not in path:
                path.append(key)
        temp_sum = []
        key_track = []
        for neighbors in dict1_map[key]:
            ttemp_sum = 0
            ttemp_sum = dict1_map[key][neighbors] + dict2_heuristic[neighbors] + prev_cost_path
            temp_sum.append(ttemp_sum)
            key_track.append(neighbors)
        
        prev_cost_path1, nextkey = min_index(temp_sum, key_track)
        if nextkey != goal:
            path.append(nextkey)
       
        temp_sum, temp_path = final_cost(dict1_map, dict2_heuristic, nextkey, prev_cost_path1, goal, path)
        sum += temp_sum
        if temp_path[-1] != goal and temp_path[-1] not in path:
            path.append(temp_path[-1])
    
    if goal not in path:
        path.append(goal)
    return sum, list(path)


def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor, _ in graph[node].items():
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)


def dfs(graph, start, goal, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = [start]

    if start == goal:
        return path

    visited.add(start)

    for neighbor, _ in graph[start].items():
        if neighbor not in visited:
            new_path = list(path)
            new_path.append(neighbor)
            res = dfs(graph, neighbor, goal, visited, new_path)
            if res is not None:
                return res


def ucs(graph, start, goal):
    heap = [(0, [start])]
    visited = set()

    while heap:
        (cost, path) = heapq.heappop(heap)
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor, neighbor_cost in graph[node].items():
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(heap, (cost + neighbor_cost, new_path))


if __name__ == "__main__":


    #using a nested dictionary to make romanian map graph
    Romanian_map = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

#defining heuristic values in another dictionary
heuristic = {
    'Arad': 366,
    'Bucharest': 0,
    'Craiova': 160,
    'Drobeta': 242,
    'Eforie': 161,
    'Fagaras': 176,
    'Giurgiu': 77,
    'Hirsova': 151,
    'Iasi': 226,
    'Lugoj': 244,
    'Mehadia': 241,
    'Neamt': 234,
    'Oradea': 380,
    'Pitesti': 100,
    'Rimnicu Vilcea': 193,
    'Sibiu': 253,
    'Timisoara': 329,
    'Urziceni': 80,
    'Vaslui': 199,
    'Zerind':374
}
#for A-STAR Algo
current_final_cost = 0
ppath = []
Astar = 0
Astart_starttime = time.time()
current_final_cost = final_cost(Romanian_map,heuristic,'Arad',current_final_cost,'Bucharest',ppath)
Astart_endtime = time.time()
Astar = Astart_endtime-Astart_starttime
print(Astar)

print(current_final_cost)

#for bfs and dfs
start = 'Arad'
goal = 'Bucharest'
BFS = 0
DFS = 0

bfs_start = time.time()
bfs_path = bfs(Romanian_map,start,goal)
bfs_end = time.time()
BFS = bfs_end-bfs_start
print(bfs_path)
print(BFS)
dfs_start = time.time()
dfs_path = dfs(Romanian_map,start,goal)
dfs_end = time.time()
DFS = dfs_start-dfs_end
print(dfs_path)
print(DFS)
#for ucs
UCS = 0
ucs_start = time.time()
ucs_path = ucs(Romanian_map,start,goal)
ucs_end = time.time()
UCS = ucs_end - ucs_start
print(UCS)
print(ucs_path)
#to visualize bfs,dfs,A*,UCS time taken for the romanian map
algos = ['UCS','DFS','BFS','A*']
times = []
times.append(UCS)
times.append(DFS)
times.append(BFS)
times.append(Astar)

plt.bar(algos,times)

plt.title("Time take for search algorithms")
plt.xlabel('Algorithm')
plt.ylabel('Time (ms)')

plt.show()