import math
import random
from collections import deque

# game_classes.py

class Node:
    """Graph node with enhanced tracking for algorithm analysis"""
    def __init__(self, r, c, node_type='.', cost=1):
        self.r = r
        self.c = c
        self.type = node_type  # '.', '#', 'T', 'P', 'S', 'G'
        self.cost = cost
        self.visited_by_player = False
        self.visited_by_ai = False
        
        # Enhanced tracking for graph visualization
        self.explored_by_ai = False  # Considered but not visited
        self.times_evaluated = 0  # How many times AI evaluated this node
        self.heuristic_value = None  # Last calculated heuristic
        
    def __repr__(self):
        return f"Node({self.r}, {self.c}, {self.type})"
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.r == other.r and self.c == other.c
    
    def __hash__(self):
        return hash((self.r, self.c))


class PerformanceMetrics:
    """Track algorithm performance for educational analysis"""
    def __init__(self):
        self.nodes_explored = 0  # Nodes added to consideration
        self.nodes_visited = 0  # Nodes actually moved to
        self.backtrack_count = 0
        self.dead_ends_hit = 0
        self.traps_triggered = 0
        self.powerups_collected = 0
        self.evaluation_history = []  # List of (node, heuristic_value)
        
    def record_evaluation(self, node, heuristic_val):
        """Record when AI evaluates a node"""
        self.nodes_explored += 1
        node.explored_by_ai = True
        node.times_evaluated += 1
        node.heuristic_value = heuristic_val
        self.evaluation_history.append((node, heuristic_val))
    
    def record_visit(self, node):
        """Record actual movement to node"""
        self.nodes_visited += 1
        node.visited_by_ai = True
    
    def record_backtrack(self):
        self.backtrack_count += 1
    
    def record_dead_end(self):
        self.dead_ends_hit += 1
    
    def get_efficiency_ratio(self, optimal_path_length):
        """Calculate how efficient the path was vs optimal"""
        if optimal_path_length == 0:
            return 0.0
        return optimal_path_length / max(self.nodes_visited, 1)
    
    def get_exploration_ratio(self, total_nodes):
        """Percentage of graph explored"""
        return (self.nodes_explored / total_nodes) * 100


class Maze:
    """Graph-based maze with enhanced features"""
    def __init__(self, grid_layout=None, width=10, height=10, seed=None):
        self.grid = []
        self.start_node = None
        self.goal_node = None
        self.width = 0
        self.height = 0
        self.seed = seed or random.randint(0, 999999)
        
        # Graph structure representation
        self.adjacency_list = {}  # node -> [(neighbor, edge_weight)]
        
        if grid_layout:
            self.parse_layout(grid_layout)
        else:
            random.seed(self.seed)
            self.generate_random(width, height)
        
        self.build_graph()
        self.optimal_path_length = self.calculate_optimal_path()
    
    def parse_layout(self, layout_str):
        """Parse string layout into graph nodes"""
        lines = layout_str.strip().split('\n')
        self.height = len(lines)
        self.width = len(lines[0])
        
        for r, line in enumerate(lines):
            row = []
            for c, char in enumerate(line.strip()):
                cost = 1
                if char == 'T': cost = 3
                elif char == 'P': cost = -2
                elif char == '#': cost = float('inf')
                
                node = Node(r, c, char, cost)
                row.append(node)
                
                if char == 'S':
                    self.start_node = node
                    node.cost = 0
                elif char == 'G':
                    self.goal_node = node
            self.grid.append(row)
    
    def generate_random(self, width, height):
        """Generate random maze using DFS (creates graph structure)"""
        self.width = width
        self.height = height
        
        # Initialize all walls
        self.grid = []
        for r in range(height):
            row = []
            for c in range(width):
                node = Node(r, c, '#', float('inf'))
                row.append(node)
            self.grid.append(row)
        
        # DFS maze generation (creates graph paths)
        stack = [(0, 0)]
        visited = set([(0, 0)])
        self.grid[0][0].type = '.'
        self.grid[0][0].cost = 1
        
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
                    self.grid[wall_r][wall_c].cost = 1
                    self.grid[nr][nc].type = '.'
                    self.grid[nr][nc].cost = 1
                    
                    visited.add((nr, nc))
                    stack.append((nr, nc))
                    found_next = True
                    break
            
            if not found_next:
                stack.pop()
        
        # Set start and goal
        self.start_node = self.grid[0][0]
        self.start_node.type = 'S'
        self.start_node.cost = 0
        
        self.goal_node = self.grid[height-1][width-1]
        self.goal_node.type = 'G'
        self.goal_node.cost = 1
        
        # Ensure goal is reachable
        if self.goal_node.type == '#':
            self.goal_node.type = 'G'
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = height-1+dr, width-1+dc
                if 0 <= nr < height and 0 <= nc < width:
                    self.grid[nr][nc].type = '.'
                    self.grid[nr][nc].cost = 1
        
        # Add loops (increase graph connectivity)
        for r in range(1, height-1):
            for c in range(1, width-1):
                if self.grid[r][c].type == '#' and random.random() < 0.15:
                    open_neighbors = sum(1 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                                       if self.grid[r+dr][c+dc].type != '#')
                    if open_neighbors >= 2:
                        self.grid[r][c].type = '.'
                        self.grid[r][c].cost = 1
        
        # Add traps and powerups (weighted graph edges)
        for r in range(height):
            for c in range(width):
                node = self.grid[r][c]
                if node.type == '.' and (r,c) != (0,0) and (r,c) != (height-1, width-1):
                    rand = random.random()
                    if rand < 0.06:
                        node.type = 'T'
                        node.cost = 3
                    elif rand < 0.10:
                        node.type = 'P'
                        node.cost = -2
    
    def build_graph(self):
        """Build explicit adjacency list representation of the graph"""
        self.adjacency_list = {}
        
        for r in range(self.height):
            for c in range(self.width):
                node = self.grid[r][c]
                if node.type == '#':
                    continue
                
                neighbors = self.get_neighbors(node)
                self.adjacency_list[node] = [(neighbor, neighbor.cost) for neighbor in neighbors]
    
    def get_node(self, r, c):
        """Get node at grid position"""
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid[r][c]
        return None
    
    def get_neighbors(self, node):
        """Get all neighbors (graph adjacency)"""
        neighbors = []
        # 8 directions for richer graph connectivity
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinals
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonals
        ]
        
        for dr, dc in directions:
            nr, nc = node.r + dr, node.c + dc
            neighbor = self.get_node(nr, nc)
            if neighbor and neighbor.type != '#':
                neighbors.append(neighbor)
        return neighbors
    
    def heuristic(self, node, heuristic_type='euclidean'):
        """Calculate heuristic for greedy algorithm"""
        if heuristic_type == 'euclidean':
            return math.sqrt((node.r - self.goal_node.r)**2 + (node.c - self.goal_node.c)**2)
        elif heuristic_type == 'manhattan':
            return abs(node.r - self.goal_node.r) + abs(node.c - self.goal_node.c)
        elif heuristic_type == 'chebyshev':
            return max(abs(node.r - self.goal_node.r), abs(node.c - self.goal_node.c))
        return 0
    
    def calculate_optimal_path(self):
        """Calculate optimal path length using BFS (unweighted graph)"""
        if not self.start_node or not self.goal_node:
            return 0
        
        queue = deque([(self.start_node, 0)])
        visited = {self.start_node}
        
        while queue:
            node, dist = queue.popleft()
            
            if node == self.goal_node:
                return dist
            
            for neighbor in self.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return float('inf')  # No path exists
    
    def get_total_walkable_nodes(self):
        """Count total nodes in graph (excluding walls)"""
        return sum(1 for r in range(self.height) for c in range(self.width) 
                  if self.grid[r][c].type != '#')


class Player:
    """Human player with enhanced tracking"""
    def __init__(self, start_node):
        self.current_node = start_node
        self.total_cost = 0
        self.steps = 0
        self.traps_triggered = 0
        self.powerups_collected = 0
        self.path = [start_node]
        self.finished = False
        self.visited_positions = {(start_node.r, start_node.c)}
    
    def move(self, direction, maze):
        """Move in graph (traverse edge)"""
        nr = self.current_node.r + direction[0]
        nc = self.current_node.c + direction[1]
        target = maze.get_node(nr, nc)
        
        if target and target.type != '#':
            self.current_node = target
            self.current_node.visited_by_player = True
            self.steps += 1
            self.path.append(target)
            self.visited_positions.add((target.r, target.c))
            return True
        return False


class GreedyAI:
    """Greedy Best-First Search AI with enhanced metrics"""
    def __init__(self, start_node):
        self.current_node = start_node
        self.total_cost = 0
        self.steps = 0
        self.traps_triggered = 0
        self.powerups_collected = 0
        self.path = [start_node]
        self.finished = False
        self.stack = [start_node]  # For backtracking
        
        # Enhanced metrics
        self.metrics = PerformanceMetrics()
        self.decision_history = []  # Track each decision made
        self.current_candidates = []  # Current nodes being considered
        
    def choose_move(self, maze):
        """Make greedy decision (choose node with best heuristic)"""
        if self.finished:
            return
        
        neighbors = maze.get_neighbors(self.current_node)
        
        # Filter unvisited nodes
        candidates = [n for n in neighbors if n not in self.path]
        
        # Record all candidates being evaluated
        self.current_candidates = []
        for node in candidates:
            heuristic_val = maze.heuristic(node)
            self.metrics.record_evaluation(node, heuristic_val)
            self.current_candidates.append((node, heuristic_val))
        
        if candidates:
            # GREEDY CHOICE: Pick node with minimum heuristic
            candidates.sort(key=lambda n: maze.heuristic(n))
            best_node = candidates[0]
            best_heuristic = maze.heuristic(best_node)
            
            # Record decision
            self.decision_history.append({
                'from': self.current_node,
                'to': best_node,
                'heuristic': best_heuristic,
                'alternatives': [(n, maze.heuristic(n)) for n in candidates[1:]]
            })
            
            self.current_node = best_node
            self.steps += 1
            self.path.append(best_node)
            self.stack.append(best_node)
            self.metrics.record_visit(best_node)
            
        else:
            # Backtrack (graph traversal fallback)
            self.metrics.record_dead_end()
            
            if len(self.stack) > 1:
                self.metrics.record_backtrack()
                self.stack.pop()
                back_node = self.stack[-1]
                
                self.current_node = back_node
                self.steps += 1
                self.path.append(back_node)
            else:
                print("AI stuck: No valid path found")
                self.finished = True
    
    def get_exploration_percentage(self, total_nodes):
        """How much of the graph was explored"""
        return self.metrics.get_exploration_ratio(total_nodes)
    
    def get_efficiency_vs_optimal(self, optimal_length):
        """How efficient was greedy vs optimal solution"""
        return self.metrics.get_efficiency_ratio(optimal_length)