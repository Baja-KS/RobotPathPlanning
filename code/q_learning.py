#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load

with open("params.yaml", "r") as f:
    config: dict = safe_load(f)


def read_map_file(path):
    map_file = open('./Map')

    rows = map_file.readlines()

    map_file.close()

    for i in range(len(rows)):
        rows[i] = rows[i].strip()
        rows[i] = rows[i].split(' ')

        rows[i] = list(map(lambda x: int(x), rows[i]))

    return np.array(rows)


tileMap = read_map_file('./Map')


class Env:
    def __init__(self, tile_map):
        self.tileMap = tile_map
        self.printMap = np.array(tile_map)
        self.tileMapCopy = np.array(tile_map)
        self.env_rows = len(self.tileMap)
        self.env_columns = len(self.tileMap[0])

        self.agent_position = [0, 0]
        self.actions = ['up', 'right', 'down', 'left']

        self.trained = False

        self.q_values = np.zeros(
            (self.env_rows, self.env_columns, self.env_rows, self.env_columns,
             self.env_rows, self.env_columns, 2, len(self.actions)))

    def train(self, epsilon, epsilon_decay, discount_factor, learning_rate, num_episodes):
        training_paths = []

        rewards = []

        for episode in range(num_episodes):
            if config.get("stop_exploring_after") and episode == config.get("stop_exploring_after"):
                epsilon = 0

            print(episode, ' : ', epsilon)

            # start training -> starting location of the agent,target location and box location
            row_index, column_index = env.get_starting_location()
            target_index_x, target_index_y = env.get_target_location()
            box_index_x, box_index_y = env.get_box_location(target_index_x, target_index_y)
            carry_index = 0

            start_row, start_column = row_index, column_index

            # add locations to path
            training_path = []
            training_path.append([start_row, start_column])
            training_path.append([target_index_x, target_index_y])
            training_path.append([box_index_x, box_index_y])

            move_counter = 0
            total_reward = 0

            # iteration -> agent tries to find a path to the box,and then to the target location
            while not env.is_terminal_state(row_index, column_index, target_index_x, target_index_y, carry_index):
                # train time move limit
                if config.get('move_limit') and move_counter > config.get("move_limit"):
                    break
                move_counter += 1

                action_index = env.get_next_action(row_index, column_index, target_index_x, target_index_y,
                                                   box_index_x, box_index_y, carry_index, epsilon)

                training_path.append(self.actions[action_index][0])

                old_row_index, old_column_index, old_carry_index = row_index, column_index, carry_index

                row_index, column_index, carry_index, rew = env.next_state(row_index, column_index,
                                                                           target_index_x, target_index_y,
                                                                           box_index_x, box_index_y,
                                                                           carry_index, action_index)

                total_reward += rew

                # Bellman's equation -> temporal difference & q value update
                old_q_value = env.q_values[old_row_index, old_column_index, target_index_x, target_index_y,
                                           box_index_x, box_index_y,
                                           old_carry_index, action_index]

                temporal_difference = rew + discount_factor * np.max(
                    env.q_values[row_index, column_index, target_index_x, target_index_y,
                                 box_index_x, box_index_y, carry_index]) - old_q_value

                new_q_value = old_q_value + learning_rate * temporal_difference

                env.q_values[old_row_index, old_column_index, target_index_x, target_index_y,
                             box_index_x, box_index_y, old_carry_index, action_index] = new_q_value

            print(total_reward)
            rewards.append(total_reward)
            env.reset_env()
            # eps = min_eps + (max_eps - min_eps) * np.exp(-eps_decay*episode)
            epsilon *= epsilon_decay
            # every xth episode is exported
            if (episode + 1) % config.get("iteration_export_cycle") == 0:
                training_paths.append(training_path)

        self.trained = True
        print('Training complete!')
        np.save('q_values', env.q_values)
        # Path export to a format compatible with our unity simulation
        if config.get("export_training_paths"):
            export_path_to_simulation_format(training_paths, "training-paths.txt")

        plt.plot(range(0, num_episodes), rewards)
        plt.show()

    def reset_env(self):
        self.tileMap = np.array(self.tileMapCopy)
        self.printMap = np.array(self.tileMapCopy)

    def get_shortest_path(self, start_x, start_y, target_x, target_y, box_x, box_y):
        path = []
        simulation_format = []
        if self.tileMap[start_x, start_y] == 0:
            return []

        row_index, column_index, target_index_x, target_index_y, \
        box_index_x, box_index_y, \
        carry_index = start_x, start_y, target_x, target_y, box_x, box_y, 0

        path.append([start_x, start_y])
        simulation_format.append([start_x, start_y])
        simulation_format.append([target_index_x, target_index_y])
        simulation_format.append([box_index_x, box_index_y])

        move_counter = 0
        reward = 0

        while not self.is_terminal_state(row_index, column_index,
                                         target_index_x, target_index_y, carry_index):
            if move_counter == config.get('move_limit'):
                break

            move_counter += 1

            action_index = self.get_next_action(row_index, column_index,
                                                target_index_x, target_index_y,
                                                box_index_x, box_index_y, carry_index, -1)

            row_index, column_index, carry_index, reward = self.next_state(row_index, column_index,
                                                                           target_index_x, target_index_y,
                                                                           box_index_x, box_index_y,
                                                                           carry_index, action_index)

            path.append([row_index, column_index])
            simulation_format.append(self.actions[action_index][0])
            reward += reward

        return path, simulation_format, reward

    # test agent performance (accuracy)
    def test(self, test_episodes):
        if not self.trained:
            self.q_values = np.load('q_values.npy')

        test_paths = []

        passed = 0

        for test_episode in range(test_episodes):

            row_index, column_index = self.get_starting_location()
            target_index_x, target_index_y = self.get_target_location()
            box_index_x, box_index_y = self.get_box_location(target_index_x, target_index_y)

            path, simulation_format, reward = self.get_shortest_path(row_index, column_index,
                                                                     target_index_x, target_index_y,
                                                                     box_index_x, box_index_y)

            if (test_episode+1) % config.get("iteration_export_cycle") == 0:
                test_paths.append(simulation_format)

            if reward > 0:
                passed += 1
            else:
                print('------------------------------------------------------------------------------')
                print('Test: ' + str(test_episode))
                print(f'Starting location: [{row_index}, {column_index}]')
                print(f'Target location: [{target_index_x}, {target_index_y}]')
                print(f'Box location: [{box_index_x}, {box_index_y}]')
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
    def is_terminal_state(self, row_index, column_index, target_index_x, target_index_y, carry_state):
        if self.tileMap[row_index, column_index] == 0 \
                or (row_index == target_index_x and column_index == target_index_y and carry_state == 1):
            return True
        else:
            return False

    def get_next_action(self, row_index, column_index, target_index_x, target_index_y,
                        box_index_x, box_index_y, carry_index, epsilon):
        if np.random.random() > epsilon:
            return np.argmax(self.q_values[row_index, column_index, target_index_x, target_index_y,
                                           box_index_x, box_index_y, carry_index])

        else:
            return np.random.randint(4)

    # initial agent starting location,excludes walls
    def get_starting_location(self):
        row_index = np.random.randint(self.env_rows)
        column_index = np.random.randint(self.env_columns)

        while self.tileMap[row_index, column_index] == 0:
            row_index = np.random.randint(self.env_rows)
            column_index = np.random.randint(self.env_columns)

        return row_index, column_index

    # initial target starting location,excludes walls
    def get_target_location(self):
        self.get_starting_location()
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

    def next_state(self, row_index, column_index, target_index_x, target_index_y,
                   box_index_x, box_index_y, carry_index, action_index):
        new_row_index = row_index
        new_column_index = column_index
        new_carry_index = carry_index

        reward = 0

        if self.actions[action_index] == 'up' and row_index > 0:
            new_row_index -= 1

        elif self.actions[action_index] == 'right' and column_index < self.env_columns - 1:
            new_column_index += 1

        elif self.actions[action_index] == 'down' and row_index < self.env_rows - 1:
            new_row_index += 1

        elif self.actions[action_index] == 'left' and column_index > 0:
            new_column_index -= 1

        if self.tileMap[new_row_index, new_column_index] == 0:
            reward += config.get("wall_reward")

        elif target_index_x == new_row_index and target_index_y == new_column_index:
            if carry_index == 1:
                reward += config.get('target_reward')

        elif carry_index == 0 and box_index_x == new_row_index and box_index_y == new_column_index:
            new_carry_index = 1
            reward += config.get('pickup_reward')

        reward += config.get("path_reward")

        self.agent_position = [new_row_index, new_column_index]
        return new_row_index, new_column_index, new_carry_index, reward

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
eps_decay = config.get("epsilon_decay")
disc_factor = config.get("discount_factor")
learn_rate = config.get("learning_rate")
n_episodes = config.get("episodes")
n_test_episodes = config.get('test_episodes')

env = Env(tileMap)
env.train(eps, eps_decay, disc_factor, learn_rate, n_episodes)

env.test(n_test_episodes)
