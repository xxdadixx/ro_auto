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
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Find an optimal 8-direction path without diagonal corner cutting."""
        if (
            len(start) != 2
            or len(goal) != 2
            or not (
                0 <= start[0] < self.grid_w
                and 0 <= start[1] < self.grid_h
            )
            or not (
                0 <= goal[0] < self.grid_w
                and 0 <= goal[1] < self.grid_h
            )
        ):
            return None

        if (
            self.grid[start[0]][start[1]] == 1
            or self.grid[goal[0]][goal[1]] == 1
        ):
            return None

        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )

        open_set = [
            (self._heuristic(start, goal), start)
        ]

        closed_set = set()
        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            if current == goal:
                path = [current]

                while path[-1] != start:
                    path.append(came_from[path[-1]])

                path.reverse()
                return path

            closed_set.add(current)

            current_x, current_y = current

            for dx, dy in neighbors:
                nx = current_x + dx
                ny = current_y + dy

                neighbor = (nx, ny)

                if not (
                    0 <= nx < self.grid_w
                    and 0 <= ny < self.grid_h
                ):
                    continue

                if self.grid[nx][ny] == 1:
                    continue

                if neighbor in closed_set:
                    continue

                # Prevent diagonal movement through obstacle corners.
                if dx != 0 and dy != 0:
                    if self.grid[current_x + dx][current_y] == 1:
                        continue

                    if self.grid[current_x][current_y + dy] == 1:
                        continue

                step_cost = (
                    math.sqrt(2.0)
                    if dx != 0 and dy != 0
                    else 1.0
                )

                tentative_g = (
                    g_score[current] + step_cost
                )

                if tentative_g >= g_score.get(
                    neighbor,
                    math.inf,
                ):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                estimated_total = (
                    tentative_g
                    + self._heuristic(neighbor, goal)
                )

                heapq.heappush(
                    open_set,
                    (estimated_total, neighbor),
                )

        return None
