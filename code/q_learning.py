#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load

with open("params.yaml", "r") as f:
    config: dict = safe_load(f)


def read_map_file(path):
    map_file = open(path)

    rows = map_file.readlines()

    map_file.close()

    for i in range(len(rows)):
        rows[i] = rows[i].strip()
        rows[i] = rows[i].split(' ')

        rows[i] = list(map(lambda x: int(x), rows[i]))

    return np.array(rows)


tileMap = read_map_file('./Map')
testMap = read_map_file('./testMap')


class Env:
    def __init__(self, tile_map, testMap):
        self.tileMap = tile_map
        self.testMap = testMap
        self.printMap = np.array(tile_map)
        self.tileMapCopy = np.array(tile_map)
        self.env_rows = len(self.tileMap)
        self.env_columns = len(self.tileMap[0])

        self.agent_position = [0, 0]
        self.actions = ['up', 'right', 'down', 'left']

        self.trained = False

        dims = []
        for i in range(8):
            dims.append(2)
        dims.append(self.env_rows)
        dims.append(self.env_columns)
        dims.append(self.env_rows)
        dims.append(self.env_columns)
        dims.append(len(self.actions))

        self.q_values = np.zeros(dims)
        self.q_values[:, 0, :, :, :, :, :, :, :, :, :, :, 0] = -10000
        self.q_values[:, :, :, :, 0, :, :, :, :, :, :, :, 1] = -10000
        self.q_values[:, :, :, :, :, :, 0, :, :, :, :, :, 2] = -10000
        self.q_values[:, :, :, 0, :, :, :, :, :, :, :, :, 3] = -10000

    def train(self, epsilon, min_epsilon, epsilon_decay, discount_factor,
              learning_rate, min_learning_rate, learning_rate_decay, num_episodes):
        training_paths = []

        rewards = []
        rewards.append(0)

        plot_episodes = []
        plot_episodes.append(0)

        map_change = []

        for k in range(num_episodes // 1000):
            map_copy = np.array(self.tileMap)
            for i in range(1, self.env_rows - 1):
                for j in range(1, self.env_columns - 1):
                    if np.random.random() < 0.3:
                        map_copy[i][j] = 0
            map_change.append(map_copy)

        map_index = 0

        for episode in range(num_episodes):
            if config.get("stop_exploring_after") and episode >= config.get("stop_exploring_after"):
                learning_rate = 0.9
                epsilon = -1

            if episode % 1000 == 0:
                self.tileMap = np.array(map_change[map_index])
                map_index += 1

            print(episode, ' : ', epsilon)
            # start training -> starting location of the agent,target location and box location
            row_index, column_index = self.get_starting_location()
            target_index_x, target_index_y = self.get_target_location(row_index, column_index)

            start_row, start_column = row_index, column_index

            # add locations to path
            training_path = []
            training_path.append([start_row, start_column])
            training_path.append([target_index_x, target_index_y])

            move_counter = 0
            total_reward = 0

            # iteration -> agent tries to find a path to the box,and then to the target location
            while not self.is_terminal_state(row_index, column_index, target_index_x, target_index_y):
                # train time move limit
                if config.get('move_limit') and move_counter > config.get("move_limit"):
                    break
                move_counter += 1

                view = self.get_view(row_index, column_index)

                action_index = self.get_next_action(view, row_index, column_index, target_index_x, target_index_y,
                                                    epsilon)

                training_path.append(self.actions[action_index][0])

                old_row_index, old_column_index = row_index, column_index
                old_view = list(view)

                view, row_index, column_index, rew = self.next_state(view, row_index, column_index,
                                                                     target_index_x, target_index_y,
                                                                     action_index)

                total_reward += rew

                # Bellman's equation -> temporal difference & q value update
                old_q_value = self.q_values[
                    tuple(old_view + [old_row_index, old_column_index, target_index_x, target_index_y, action_index])]

                temporal_difference = rew + discount_factor * np.max(
                    self.q_values[
                        tuple(view + [row_index, column_index, target_index_x, target_index_y])]) - old_q_value

                new_q_value = old_q_value + learning_rate * temporal_difference

                env.q_values[tuple(
                    old_view + [old_row_index, old_column_index, target_index_x,
                                target_index_y, action_index])] = new_q_value

            print(total_reward)
            if episode % config.get("iteration_plot_cycle") == 0:
                rewards.append(total_reward)
                plot_episodes.append(episode)

            # eps = min_eps + (max_eps - min_eps) * np.exp(-eps_decay*episode)
            epsilon *= epsilon_decay
            if epsilon < min_epsilon:
                epsilon = min_epsilon

            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            # every xth episode is exported
            if (episode + 1) % config.get("iteration_export_cycle") == 0:
                training_paths.append(training_path)

        self.trained = True
        print('Training complete!')
        np.save('q_values', env.q_values)
        # Path export to a format compatible with our unity simulation
        if config.get("export_training_paths"):
            export_path_to_simulation_format(training_paths, "training-paths.txt")

        plt.plot(plot_episodes, rewards)
        plt.show()

    def reset_env(self):
        self.tileMap = np.array(self.tileMapCopy)
        self.printMap = np.array(self.tileMapCopy)

    def get_view(self, row_index, column_index):
        view = np.zeros(8, dtype='int').tolist()

        for i in range(3):
            view[i] = self.tileMap[row_index - 1][column_index + i - 1]

        view[3] = self.tileMap[row_index][column_index - 1]
        view[4] = self.tileMap[row_index][column_index + 1]

        for i in range(3):
            view[i + 5] = self.tileMap[row_index + 1][column_index + i - 1]

        return view

    def get_shortest_path(self, start_x, start_y, target_x, target_y, adaptation_enabled=False):
        path = []
        simulation_format = []
        if self.tileMap[start_x, start_y] == 0:
            return []

        row_index, column_index, \
        target_index_x, target_index_y = start_x, start_y, target_x, target_y

        view = self.get_view(row_index, column_index)

        path.append([start_x, start_y])
        simulation_format.append([start_x, start_y])
        simulation_format.append([target_index_x, target_index_y])

        move_counter = 0
        reward = 0

        while not self.is_terminal_state(row_index, column_index,
                                         target_index_x, target_index_y):
            if move_counter == config.get('move_limit'):
                break

            move_counter += 1

            action_index = self.get_next_action(view, row_index, column_index,
                                                target_index_x, target_index_y, -1)

            old_view = list(view)
            old_row_index, old_column_index, old_target_index_x, \
            old_target_index_y = row_index, column_index, target_index_x, target_index_y

            view, row_index, column_index, rew = self.next_state(view, row_index, column_index,
                                                                 target_index_x, target_index_y,
                                                                 action_index)

            old_q_value = self.q_values[
                tuple(old_view + [old_row_index, old_column_index, target_index_x, target_index_y, action_index])]

            temporal_difference = rew + 0.9 * np.max(
                self.q_values[
                    tuple(view + [row_index, column_index, target_index_x, target_index_y])]) - old_q_value

            new_q_value = old_q_value + 0.9 * temporal_difference

            if adaptation_enabled:
                env.q_values[tuple(
                    old_view + [old_row_index, old_column_index, target_index_x,
                                target_index_y, action_index])] = new_q_value

            path.append([row_index, column_index])
            simulation_format.append(self.actions[action_index][0])
            reward += rew
            print(move_counter, ':', reward)

        return path, simulation_format, reward

    # test agent performance (accuracy)
    def test(self, test_episodes, adaptation_enabled):
        if not self.trained:
            self.q_values = np.load('q_values.npy')

        test_paths = []

        self.tileMap = np.array(self.testMap)
        self.env_rows = len(self.tileMap)
        self.env_columns = len(self.tileMap[0])

        passed = 0

        for test_episode in range(test_episodes):

            row_index, column_index = self.get_starting_location()
            target_index_x, target_index_y = self.get_target_location(row_index, column_index)

            path, simulation_format, reward = self.get_shortest_path(row_index, column_index,
                                                                     target_index_x, target_index_y, adaptation_enabled)

            if (test_episode + 1) % config.get("test_export_cycle") == 0:
                test_paths.append(simulation_format)

            if reward > 0:
                passed += 1
            else:
                print('------------------------------------------------------------------------------')
                print('Test: ' + str(test_episode))
                print(f'Starting location: [{row_index}, {column_index}]')
                print(f'Target location: [{target_index_x}, {target_index_y}]')
                print(path)
                print(reward)

        # every xth episode is exported
        if config.get('export_test_paths'):
            export_path_to_simulation_format(test_paths, 'test-paths.txt')

        print(f'Accuracy: {(passed / test_episodes) * 100}%')

    def set_agent_position(self, x, y):
        if x < 0 or x >= self.env_rows:
            return
        if y < 0 or y >= self.env_columns:
            return

        self.agent_position = [x, y]

    # terminal states are walls and target location if carry_state equals to 1 (i.e. if agent carries a box)
    def is_terminal_state(self, row_index, column_index, target_index_x, target_index_y):
        if self.tileMap[row_index, column_index] == 0 \
                or (row_index == target_index_x and column_index == target_index_y):
            return True
        else:
            return False

    def get_next_action(self, view, row_index, column_index, target_index_x, target_index_y, epsilon):
        if np.random.random() > epsilon:
            return np.argmax(self.q_values[tuple(view + [row_index, column_index, target_index_x, target_index_y])])

        else:
            possible_actions = []
            if view[1] == 1:
                possible_actions.append(0)
            if view[4] == 1:
                possible_actions.append(1)
            if view[3] == 1:
                possible_actions.append(2)
            if view[6] == 1:
                possible_actions.append(2)

            if len(possible_actions) != 0:
                random_index = np.random.randint(len(possible_actions))
                return possible_actions[random_index]

            return np.random.randint(len(self.actions))

    # initial agent starting location,excludes walls
    def get_starting_location(self):
        row_index = np.random.randint(1, self.env_rows - 1)
        column_index = np.random.randint(1, self.env_columns - 1)

        while self.tileMap[row_index, column_index] == 0:
            row_index = np.random.randint(1, self.env_rows - 1)
            column_index = np.random.randint(1, self.env_columns - 1)

        return row_index, column_index

    # initial target starting location,excludes walls
    def get_target_location(self, agent_x, agent_y):
        target_index_x, target_index_y = self.get_starting_location()

        while target_index_x == agent_x and target_index_y == agent_y:
            target_index_x, target_index_y = self.get_starting_location()

        return target_index_x, target_index_y
        # target_index_x = np.random.randint(self.env_rows)
        # target_index_y = np.random.randint(self.env_columns)
        #
        # while self.tileMap[target_index_x, target_index_y] == 0:
        #     target_index_x = np.random.randint(self.env_rows)
        #     target_index_y = np.random.randint(self.env_columns)
        #
        # return target_index_x, target_index_y

    def get_box_location(self, target_index_x, target_index_y):
        box_index_x = np.random.randint(self.env_rows)
        box_index_y = np.random.randint(self.env_columns)

        while self.tileMap[box_index_x, box_index_y] == 0 \
                or (box_index_x == target_index_x and box_index_y == target_index_y):
            box_index_x = np.random.randint(self.env_rows)
            box_index_y = np.random.randint(self.env_columns)

        return box_index_x, box_index_y

    def next_state(self, view, row_index, column_index, target_index_x,
                   target_index_y, action_index):
        new_row_index = row_index
        new_column_index = column_index
        new_view = list(view)

        reward = 0
        wall = False

        if self.actions[action_index] == 'up':
            new_row_index -= 1

        elif self.actions[action_index] == 'right':
            new_column_index += 1

        elif self.actions[action_index] == 'down':
            new_row_index += 1

        elif self.actions[action_index] == 'left':
            new_column_index -= 1

        if self.tileMap[new_row_index, new_column_index] == 0:
            wall = True
            reward += config.get("wall_reward")

        elif target_index_x == new_row_index and target_index_y == new_column_index:
            reward += config.get('target_reward')

        reward += config.get("path_reward")

        if not wall:
            new_view = self.get_view(new_row_index, new_column_index)

        self.agent_position = [new_row_index, new_column_index]
        return new_view, new_row_index, new_column_index, reward

    def visual_update(self):
        self.printMap = np.array(self.tileMap)
        self.printMap[self.agent_position[0], self.agent_position[1]] = 8

    def print_env(self):
        print(self.printMap)


def export_path_to_simulation_format(paths, file):
    try:
        with open(file, "w") as F:
            lines = []
            for path in paths:
                for i in range(0, 3):
                    lines += str(path[i][0]) + ' ' + str(path[i][1]) + '\n'

                for action_i in range(3, len(path)):
                    lines += path[action_i] + '\n'

                lines += 'x' + '\n'

            lines.pop(-1)
            F.writelines(lines)
    except IOError:
        print("error writing to file")


eps = config.get("epsilon")
min_eps = config.get("min_epsilon")
eps_decay = config.get("epsilon_decay")
disc_factor = config.get("discount_factor")
learn_rate = config.get("learning_rate")
learn_rate_decay = config.get("learning_rate_decay")
min_learn_rate = config.get("min_learning_rate")
n_episodes = config.get("episodes")
n_test_episodes = config.get('test_episodes')

print(tileMap)
env = Env(tileMap, testMap)
#env.train(eps, min_eps, eps_decay, disc_factor, learn_rate, min_learn_rate, learn_rate_decay, n_episodes)

env.test(n_test_episodes, adaptation_enabled=False)
