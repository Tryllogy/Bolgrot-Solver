from ..map import Map
from ..case import CaseType


class BFS:
    """4-directional BFS pathfinder bound to a ``Map``."""

    def __init__(
        self,
        map: Map
    ):
        """Store the map whose cells define walkable positions."""
        self.map: Map = map

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Return the first step of the shortest path from start to goal.

        Runs BFS and, on success, returns the single tile to move to next
        (the step adjacent to ``start``), or ``None`` if no path exists.
        """
        from collections import deque

        queue: deque[tuple[int, int]] = deque([start])
        visited: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while queue:
            current = queue.popleft()

            if current == goal:
                path: list[tuple[int, int]] = []
                node: tuple[int, int] | None = current
                while node is not None:
                    path.append(node)
                    node = visited[node]
                path.pop(-1)
                return path[-1]

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)

        return None

    def get_neighbors(
        self,
        pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Get valid neighboring positions (up, down, left, right)."""
        x, y = pos
        neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1)
        ]
        valid_neighbors = [
            (nx, ny) for nx, ny in neighbors
            if self.is_valid_position((nx, ny))
        ]
        return valid_neighbors

    def is_valid_position(
        self,
        pos: tuple[int, int]
    ) -> bool:
        """Check if the position is within the map and not blocked."""
        case = self.map.cases.get(pos)
        return case is not None and case.case_type != CaseType.WALL
