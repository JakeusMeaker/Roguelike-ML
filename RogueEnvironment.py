import gym
import numpy
from gym import spaces
from gym.utils import seeding
from typing import Optional, Union

import numpy as np
import engine
import tcod
import actions
import entity_factories
import copy

import tile_types
from procgen import generate_dungeon

MOVE_DIRS = [
    # WASD Keys
    (0, -1),
    (-1, 0),
    (0, 1),
    (1, 0),
    #(-1, -1),
    #(1, -1),
    #(-1, 1),
    #(1, 1)
]


class RogueEnv(gym.Env):
    metadata = {
        "render.modes": []
    }

    def __init__(self):
        self.screen_width = 80
        self.screen_height = 50

        self.map_width = 80
        self.map_height = 43

        self.room_max_size = 10
        self.room_min_size = 6
        self.max_rooms = 30

        self.max_monsters_per_room = 2

        self.engine = None

        self.seed()

        self.context = None
        self.root_console = None


        tilesmax = np.array(
            [
                255
            ] * (2 + (80*43)),
            dtype=np.float32
        )

        tilesmin = np.array(
            [
                0
            ] * (2 + (80*43)),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(tilesmin, tilesmax, dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.tiles_explored = 0

    def step(self, action):
        """"""
        # TODO take action, generate reward and sucessor state
        assert self.action_space.contains(action), type(action)

        action = actions.BumpAction(self.engine.player, *MOVE_DIRS[action])
        action.perform()

        self.engine.handle_enemy_turns()
        self.engine.update_fov()  # Update the FOV before the players next action.

        healthdelta = self.engine.player.fighter.max_hp - self.engine.player.fighter.hp

        explored = sum(sum(self.engine.game_map.explored))
        explored_delta = explored - self.tiles_explored

        reward = explored_delta - healthdelta

        return self.generate_obs(), reward, self.engine.player.fighter.hp == 0, {}

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

        return self.generate_obs()

    def generate_obs(self):
        #obs = self.np_random.uniform(low=0.0, high=255.0, size=(2,))
        #x = np.array(obs, dtype=np.float32)
        #return x

        # x = np.array([[self.engine.player.x, self.engine.player.y] + [0] * (80*50) ], dtype=np.float32)

        visibletiles = np.empty(80 * 43 + 2, dtype=np.float32)

        for x in range(self.engine.game_map.width):
            for y in range(self.engine.game_map.height):
                if self.engine.game_map.explored[x][y]:
                    visibletiles[x * self.engine.game_map.height + y] = int(self.engine.game_map.tiles[x][y][0])
                else:
                    visibletiles[x * self.engine.game_map.height + y] = 2
                # numpy.append(visibletiles, self.engine.game_map.tiles[x][y])
                # x[ 2 + (x * self.map_height + y ) ] = int(self.engine.game_map.tiles[y][x].walkable)

        visibletiles[80 * 43] = self.engine.player.x
        visibletiles[80 * 43 + 1] = self.engine.player.y
        x = np.array(visibletiles, dtype=np.float32)

        #print(self.np_random.uniform(low=0, high=255, size=(2,)))
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
                title="Yet Another Roguelike Tutorial",
                vsync=True,
            )

            self.root_console = tcod.Console(self.screen_width, self.screen_height, order="F")

        self.root_console.clear()
        self.engine.event_handler.on_render(console=self.root_console)
        self.context.present(self.root_console)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


gym.envs.register(id='RogueLearning-v0', entry_point='RogueEnvironment:RogueEnv', )
