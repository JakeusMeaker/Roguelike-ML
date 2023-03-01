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

        self.max_monsters_per_room = 0

        self.engine = None

        self.seed()

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
        """"""
        # TODO take action, generate reward and sucessor state
        self.steps += 1

        #print(action)
        assert self.action_space.contains(action), type(action)

        action = actions.BumpAction(self.engine.player, *MOVE_DIRS[action])
        action.perform()

        self.engine.handle_enemy_turns()
        self.engine.update_fov()  # Update the FOV before the players next action.

        explored = sum(sum(self.engine.game_map.explored))
        explored_delta = explored - self.tiles_explored
        self.tiles_explored = explored

        reward = explored_delta

        for x_ in [-1, 0, 1]:
            for y_ in [-1, 0, 1]:
                actornearby = self.engine.game_map.get_actor_at_location(self.engine.player.x + x_, self.engine.player.y + y_)
                if actornearby is not None and not actornearby.is_alive:
                    reward += 1000
        reward = reward - (self.previoushealth - self.engine.player.fighter.hp) - 1
        self.previoushealth = self.engine.player.fighter.hp

        return self.generate_obs(), reward, self.engine.player.fighter.hp == 0 or self.steps > 150, {}

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
        self.previoushealth = player.fighter.max_hp
        self.steps = 0

        return self.generate_obs()

    def generate_obs(self):
        #obs = self.np_random.uniform(low=0.0, high=255.0, size=(2,))
        #x = np.array(obs, dtype=np.float32)
        #return x

        # x = np.array([[self.engine.player.x, self.engine.player.y] + [0] * (80*50) ], dtype=np.float32)

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
                # numpy.append(observations, self.engine.game_map.tiles[x][y])
                # x[ 2 + (x * self.map_height + y ) ] = int(self.engine.game_map.tiles[y][x].walkable)

        observations[80 * 43] = self.engine.player.x / self.map_width
        observations[80 * 43 + 1] = self.engine.player.y / self.map_height

        index = 80 * 43 + 1
        for x_ in [-1, 0, 1]:
            for y_ in [-1, 0, 1]:
                observations[index] = self.engine.game_map.get_actor_at_location(self.engine.player.x + x_, self.engine.player.y + y_) is not None
                index += 1

        observations[80 * 43 + 10] = self.engine.player.fighter.hp / 30
        observations[80 * 43 + 11] = unexplored / (80 * 43)

        x = np.array(observations, dtype=np.float32)

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
