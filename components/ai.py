from __future__ import annotations

from typing import List, Tuple

import numpy as np  # type: ignore
import tcod

from actions import Action
from components.base_component import BaseComponent


class BaseAI(Action, BaseComponent):
    def perform(self) -> None:
        raise NotImplementedError()

    def get_path_to(self, dest_x: int, dest_y: int) -> List[Tuple[int, int]]:
        """Compute and return a path to the target position.

        If there is no valid path then returns and empty list
        """
        # Copy the walkable array.
        cost = np.array(self.entity.gamemap.tiles["Walkable"], dtype=np.int8)

        for entity in self.entity.gamemap.entities:
            # Check that an entity blocks movement and the cost isn't zero (blocking).
            if entity.blocks_movement and cost[entity.x, entity.y]:
                # Add to the cost of a blocked position
                # A lower number means more enemies with crowd behined each other in
                # hallways. A higher number means enemies will take longer pathes in
                # order to surround the player.
                cost[entity.x, entity.y] += 10

        # Create a graph from the cost array and pass that graph to a new pathfinder
        graph = tcod.path.SimpleGraph(cost=cost, cardinal=2, diagonal=3)
        pathfinder = tcod.path.Pathfinder(graph)

        pathfinder.add_root((self.entity.x, self.entity.y)) # Start position

        # Compute the path to the destination and remove the starting point.
        path: List[List[int]] = pathfinder.path_to((dest_x, dest_y))[1:].tolist()

        # Convert frim List[List[int]] to List[Tuple[int, int]]
        return [(index[0], index[1]) for index in path]

