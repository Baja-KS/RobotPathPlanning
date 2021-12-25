#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load

with open("params.yaml", "r") as f:
    config: dict = safe_load(f)


# print(config)
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
    def __init__(self, tileMap, epsilon, discount_factor, learning_rate):
        self.tileMap = tileMap
        self.printMap = np.array(tileMap)
        self.tileMapCopy = np.array(tileMap)
        self.env_rows = len(self.tileMap)
        self.env_columns = len(self.tileMap[0])

        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.agent_position = [0, 0]
        self.actions = ['up', 'right', 'down', 'left']

        self.q_values = np.zeros(
            (self.env_rows, self.env_columns, self.env_rows, self.env_columns,
             self.env_rows, self.env_columns, 2, len(self.actions)))

    def reset_env(self):
        self.tileMap = np.array(self.tileMapCopy)
        self.printMap = np.array(self.tileMapCopy)

    def get_shortest_path(self, start_x, start_y, target_x, target_y, box_x, box_y):
        path = []
        if self.tileMap[start_x, start_y] == 0:
            return []

        row_index, column_index, target_index_x, target_index_y, \
        box_index_x, box_index_y, \
        carry_index = start_x, start_y, target_x, target_y, box_x, box_y, 0

        path.append([start_x, start_y])

        m_counter = 0
        t_reward = 0

        while not self.is_terminal_state(row_index, column_index,
                                         target_index_x, target_index_y, carry_index):
            if m_counter == config.get('move_limit'):
                break

            m_counter += 1

            action = self.get_next_action(row_index, column_index,
                                          target_index_x, target_index_y,
                                          box_index_x, box_index_y, carry_index, -1)

            row_index, column_index, carry_index, reward = self.next_state(row_index, column_index,
                                                                           target_index_x, target_index_y,
                                                                           box_index_x, box_index_y,
                                                                           carry_index, action)

            path.append([row_index, column_index])
            t_reward += reward

        return path, t_reward

    def test(self):
        test_episodes = config.get('test_episodes')

        passed = 0

        for test_episode in range(test_episodes):

            row_index, column_index = self.get_starting_location()
            target_index_x, target_index_y = self.get_target_location()
            box_index_x, box_index_y = self.get_box_location(target_index_x, target_index_y)

            path, reward = self.get_shortest_path(row_index, column_index,
                                                  target_index_x, target_index_y,
                                                  box_index_x, box_index_y)

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

        print(f'Accuracy: {(passed / test_episodes) * 100}%')

    def set_agent_position(self, x, y):
        if x < 0 or x >= self.env_rows:
            return
        if y < 0 or y >= self.env_columns:
            return

        self.agent_position = [x, y]

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

    def get_starting_location(self):
        row_index = np.random.randint(self.env_rows)
        column_index = np.random.randint(self.env_columns)

        while self.tileMap[row_index, column_index] == 0:
            row_index = np.random.randint(self.env_rows)
            column_index = np.random.randint(self.env_columns)

        return row_index, column_index

    def get_target_location(self):
        target_index_x = np.random.randint(self.env_rows)
        target_index_y = np.random.randint(self.env_columns)

        while self.tileMap[target_index_x, target_index_y] == 0:
            target_index_x = np.random.randint(self.env_rows)
            target_index_y = np.random.randint(self.env_columns)

        return target_index_x, target_index_y

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


def action_to_short(previous_position, current_position):
    action_to_short_map = {
        -1: {
            -1: "ul",
            0: "l",
            1: "dl",
        },
        0: {
            -1: "u",
            0: "-",
            1: "d",
        },
        1: {
            -1: "ur",
            0: "r",
            1: "dr",
        }
    }

    x_diff = current_position[0] - previous_position[0]
    y_diff = current_position[1] - previous_position[1]

    return action_to_short_map[y_diff][x_diff]


def export_to_lines(path, is_last=False):
    lines = []
    start_row, start_column = path[0][0], path[0][1]
    lines.append(f"{start_row} {start_column}\n")
    for i in range(1, len(path)):
        lines.append(action_to_short(path[i - 1], path[i]) + "\n")
    if not is_last:
        lines.append("x\n")
    return lines


def export_path_to_simulation_format(paths, file):
    try:
        with open(file, "w") as f:
            lines = []
            for i in range(len(paths)):
                lines += export_to_lines(paths[i], i == len(paths) - 1)
            f.writelines(lines)
    except IOError:
        print("error writing to file")


eps = config.get("epsilon")
min_eps = config.get('min_epsilon')
max_eps = config.get('max_epsilon')
eps_decay = config.get("epsilon_decay")
discount_factor = config.get("discount_factor")
learning_rate = config.get("learning_rate")

num_episodes = config.get("episodes")
training_paths = []

env = Env(tileMap, eps, discount_factor, learning_rate)

rewards = []

for episode in range(num_episodes):
    if config.get("stop_exploring_after") and episode == config.get("stop_exploring_after"):
        eps = 0

    print(episode, ' : ', eps)
    r_index, c_index = env.get_starting_location()
    t_index_x, t_index_y = env.get_target_location()
    b_index_x, b_index_y = env.get_box_location(t_index_x, t_index_y)
    carr_index = 0

    s_row, s_column = r_index, c_index

    training_path = []
    training_path.append([s_row, s_column])

    move_counter = 0
    total_reward = 0

    while not env.is_terminal_state(r_index, c_index, t_index_x, t_index_y, carr_index):
        if config.get('move_limit') and move_counter > config.get("move_limit"):
            break
        move_counter += 1

        a_index = env.get_next_action(r_index, c_index, t_index_x, t_index_y,
                                      b_index_x, b_index_y, carr_index, eps)

        old_row_index, old_column_index, old_carry_index = r_index, c_index, carr_index

        r_index, c_index, carr_index, rew = env.next_state(r_index, c_index,
                                                           t_index_x, t_index_y,
                                                           b_index_x, b_index_y,
                                                           carr_index, a_index)

        training_path.append([r_index, c_index])

        total_reward += rew

        old_q_value = env.q_values[old_row_index, old_column_index, t_index_x, t_index_y,
                                   b_index_x, b_index_y,
                                   old_carry_index, a_index]

        temporal_difference = rew + discount_factor * np.max(
            env.q_values[r_index, c_index, t_index_x, t_index_y,
                         b_index_x, b_index_y, carr_index]) - old_q_value

        new_q_value = old_q_value + learning_rate * temporal_difference

        env.q_values[old_row_index, old_column_index, t_index_x, t_index_y,
                     b_index_x, b_index_y, old_carry_index, a_index] = new_q_value

    print(total_reward)
    rewards.append(total_reward)
    env.reset_env()
    # eps = min_eps + (max_eps - min_eps) * np.exp(-eps_decay*episode)
    eps *= eps_decay
    training_paths.append(training_path)

print('Training complete!')
np.save('q_values', env.q_values)

if config.get("export_training_paths"):
    export_path_to_simulation_format(training_paths, "training-paths.txt")

plt.plot(range(0, num_episodes), rewards)
plt.show()

env.test()

#print(env.get_shortest_path(3, 7, 2, 3, 6, 8))
#print(env.get_shortest_path(2, 1, 4, 7, 5, 8))
#print(env.get_shortest_path(5, 8, 7, 7, 3, 1))
