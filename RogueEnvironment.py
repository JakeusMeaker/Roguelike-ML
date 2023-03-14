import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import engine
import tcod
import actions
import entity_factories
import copy

from procgen import generate_dungeon

MOVE_DIRS = [
    # WASD Keys
    (0, -1),
    (-1, 0),
    (0, 1),
    (1, 0)
]


class RogueEnv(gym.Env):
    metadata = {
        "render.modes": []
    }

    def __init__(self):  # Creates and initializes the environment
        self.screen_width = 80
        self.screen_height = 50

        self.map_width = 80
        self.map_height = 43

        self.room_max_size = 10
        self.room_min_size = 6
        self.max_rooms = 30

        self.max_monsters_per_room = 0

        self.engine = None

        self.context = None
        self.root_console = None

        self.steps = 0

        tilesmax = np.array(
            [
                1
            ] * (12 + (80*43)),
            dtype=np.float32
        )

        tilesmin = np.array(
            [
                -1
            ] * (12 + (80*43)),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(tilesmin, tilesmax, dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.tiles_explored = 0

    def step(self, action):
        assert self.action_space.contains(action), type(action)

        action = actions.BumpAction(self.engine.player, *MOVE_DIRS[action])
        action.perform()

        self.engine.update_fov()  # Update the FOV before the players next action.

        # Counts the number of tiles explored this turn and compares it to the last turn and generates a delta which is
        # given to the model as a reward. Intended to encourage and reward exploration
        explored = sum(sum(self.engine.game_map.explored))
        explored_delta = explored - self.tiles_explored
        self.tiles_explored = explored
        reward = explored_delta

        # Rewards the model for staying alive by allowing it to remain alive for longer. The longer it stays alive then
        # the more reward it can get
        self.steps += 1
        if explored_delta != 0:
            self.steps -= explored_delta
        else:
            reward = -1

        return self.generate_obs(), reward, self.engine.player.fighter.hp == 0 or self.steps > 5, {}

    def reset(self):
        player = copy.deepcopy(entity_factories.player)

        self.engine = engine.Engine(player=player)
        self.engine.game_map = generate_dungeon(
            max_rooms=self.max_rooms,
            room_min_size=self.room_min_size,
            room_max_size=self.room_max_size,
            map_width=self.map_width,
            map_height=self.map_height,
            max_monsters_per_room=self.max_monsters_per_room,
            engine=self.engine
        )
        self.engine.update_fov()
        self.tiles_explored = sum(sum( self.engine.game_map.explored))
        self.steps = 0

        return self.generate_obs()

    def generate_obs(self):
        observations = np.empty(80 * 43 + 12, dtype=np.float32)

        # Adds the visible tiles to the observations
        unexplored = 0
        for x in range(self.engine.game_map.width):
            for y in range(self.engine.game_map.height):
                if self.engine.game_map.explored[x][y]:
                    observations[x * self.engine.game_map.height + y] = int(self.engine.game_map.tiles[x][y][0])
                else:
                    observations[x * self.engine.game_map.height + y] = -1
                    unexplored += 1

        # Determines the players current position for the observations
        observations[80 * 43] = self.engine.player.x / self.map_width
        observations[80 * 43 + 1] = self.engine.player.y / self.map_height

        # Adds the unexplored tiles to the observations
        observations[80 * 43 + 2] = unexplored / (80 * 43)

        x = np.array(observations, dtype=np.float32)
        return x

    def render(self, mode="human"):
        if self.root_console is None:
            tileset = tcod.tileset.load_tilesheet(
                "dejavu10x10.png", 32, 8, tcod.tileset.CHARMAP_TCOD
            )

            self.context = tcod.context.new_terminal(
                self.screen_width,
                self.screen_height,
                tileset=tileset,
                title="Roguelike ML",
                vsync=True,
            )

            self.root_console = tcod.Console(self.screen_width, self.screen_height, order="F")

        self.root_console.clear()
        self.engine.event_handler.on_render(console=self.root_console)
        self.context.present(self.root_console)


gym.envs.register(id='RogueLearning-v0', entry_point='RogueEnvironment:RogueEnv', )
