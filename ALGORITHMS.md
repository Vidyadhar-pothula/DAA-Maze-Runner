# Algorithms Used in Duel of Labyrinth

This document provides the actual Python code implementations used in the game for reference.

## 1. Depth-First Search (DFS)
**Usage:** Maze Generation
**File:** `game_classes.py`

```python
def generate_random(self, width, height):
    # ... (Initialization)
    
    # DFS maze generation (creates graph paths)
    stack = [(0, 0)]
    visited = set([(0, 0)])
    
    while stack:
        current_r, current_c = stack[-1]
        
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        random.shuffle(directions)
        
        found_next = False
        for dr, dc in directions:
            nr, nc = current_r + dr, current_c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                # Carve path (create graph edge)
                wall_r, wall_c = current_r + dr // 2, current_c + dc // 2
                self.grid[wall_r][wall_c].type = '.'
                self.grid[nr][nc].type = '.'
                
                visited.add((nr, nc))
                stack.append((nr, nc))
                found_next = True
                break
        
        if not found_next:
            stack.pop()
```

## 2. Breadth-First Search (BFS)
**Usage:** Structural Analysis (Distance Map)
**File:** `game_classes.py`

```python
def bfs_analysis(self):
    """BFS for Structural Analysis (Distance Map)"""
    if not self.goal_node: return
    
    self.bfs_map = {}
    self.max_bfs_distance = 0
    
    queue = deque([(self.goal_node, 0)])
    self.bfs_map[self.goal_node] = 0
    
    while queue:
        node, dist = queue.popleft()
        self.max_bfs_distance = max(self.max_bfs_distance, dist)
        
        for neighbor in self.get_neighbors(node):
            if neighbor not in self.bfs_map:
                self.bfs_map[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
```

## 3. A* (A-Star) Search
**Usage:** Optimal Path Calculation (Reference)
**File:** `game_classes.py`

```python
def a_star_optimal(self):
    """A* for Optimal Reference Cost"""
    if not self.start_node or not self.goal_node: return 0
    
    frontier = PriorityQueue()
    frontier.put(self.start_node, 0)
    
    cost_so_far = {}
    cost_so_far[self.start_node] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == self.goal_node:
            break
        
        for neighbor in self.get_neighbors(current):
            # Calculate edge cost (1 for cardinal, 1.414 for diagonal)
            edge_cost = 1
            if abs(current.r - neighbor.r) + abs(current.c - neighbor.c) == 2:
                edge_cost = 1.414
            
            penalty = 0
            if neighbor.type == 'T': penalty = 3
            elif neighbor.type == 'P': penalty = -2
            
            new_cost = cost_so_far[current] + max(0.1, edge_cost + penalty)
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + self.heuristic(neighbor)
                frontier.put(neighbor, priority)
    
    return cost_so_far.get(self.goal_node, 0)
```

## 4. Greedy Best-First Search
**Usage:** AI Agent Logic
**File:** `game_classes.py`

```python
def compute_path(self):
    frontier = PriorityQueue()
    frontier.put(self.current_node, 0)
    
    came_from = {}
    came_from[self.current_node] = None
    
    current = None
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == self.goal_node:
            break
        
        for neighbor in self.maze.get_neighbors(current):
            if neighbor not in came_from:
                # Greedy only cares about heuristic (distance to goal)
                priority = self.heuristic(neighbor)
                frontier.put(neighbor, priority)
                came_from[neighbor] = current
```

## 5. Huffman Coding
**Usage:** Post-Game Data Compression Analysis
**File:** `game_classes.py`

```python
class Huffman:
    def build_tree(self, text):
        if not text: return None
        
        freqs = {}
        for char in text:
            freqs[char] = freqs.get(char, 0) + 1
            
        pq = []
        for char, freq in freqs.items():
            heapq.heappush(pq, HuffmanNode(char, freq))
            
        while len(pq) > 1:
            left = heapq.heappop(pq)
            right = heapq.heappop(pq)
            parent = HuffmanNode(None, left.freq + right.freq, left, right)
            heapq.heappush(pq, parent)
            
        root = heapq.heappop(pq)
        self.generate_codes(root, "")
        return root
```
