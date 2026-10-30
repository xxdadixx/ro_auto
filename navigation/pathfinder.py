import heapq
import math
from typing import List, Optional, Tuple


class AStarPathfinder:
    """Implements A* pathfinding algorithm over a 2D grid with closed-set tracking."""

    def __init__(self, grid_w: int = 100, grid_h: int = 100) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        # 0 = walkable, 1 = obstacle
        self.grid = [[0 for _ in range(grid_h)] for _ in range(grid_w)]

    def set_obstacle(self, x: int, y: int, is_obstacle: bool = True) -> None:
        if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
            self.grid[x][y] = 1 if is_obstacle else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic for directional pathing."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def find_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Calculates optimal path from start to goal coordinates using A* search."""
        if not (0 <= start[0] < self.grid_w and 0 <= start[1] < self.grid_h):
            return None
        if not (0 <= goal[0] < self.grid_w and 0 <= goal[1] < self.grid_h):
            return None

        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start))

        closed_set = set()
        came_from = {}
        g_score = {start: 0.0}

        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                path = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                return path[::-1]

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor

                if not (0 <= nx < self.grid_w and 0 <= ny < self.grid_h):
                    continue
                if self.grid[nx][ny] == 1 or neighbor in closed_set:
                    continue

                step_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                tentative_g_score = g_score[current] + step_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None
