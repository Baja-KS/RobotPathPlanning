import numpy as np
import matplotlib.pyplot as plt


def read_map_file(path):
    f = open('./Map')

    rows = f.readlines()

    f.close()

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
        self.q_values = np.zeros((self.env_rows, self.env_columns, 3, len(self.actions)))

    def reset_env(self):
        self.tileMap = np.array(self.tileMapCopy)
        self.printMap = np.array(self.tileMapCopy)

    def set_agent_position(self, x, y):
        if x < 0 or x >= self.env_rows:
            return
        if y < 0 or y >= self.env_columns:
            return

        self.agent_position = [x, y]

    def is_terminal_state(self, row_index, column_index, carry_index):
        if self.tileMap[row_index, column_index] == 0 or self.tileMap[row_index, column_index] == 5:
            return True
        else:
            return False

    def get_next_action(self, row_index, column_index, carry_index, epsilon):
        if np.random.random() > epsilon:
            return np.argmax(self.q_values[row_index, column_index, carry_index])

        else:
            return np.random.randint(4)

    def get_starting_location(self):
        row_index = np.random.randint(self.env_rows)
        column_index = np.random.randint(self.env_columns)

        while self.is_terminal_state(row_index, column_index, 0):
            row_index = np.random.randint(self.env_rows)
            column_index = np.random.randint(self.env_columns)

        return row_index, column_index

    def next_state(self, row_index, column_index, carry_index, action_index):
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
            reward -= 100

        elif self.tileMap[new_row_index, new_column_index] == 2:
            self.tileMap[new_row_index, new_column_index] = 1
            new_carry_index += 1
            reward += 15

        elif self.tileMap[new_row_index, new_column_index] == 5:
            if carry_index == 2:
                reward += 100
            else:
                reward -= 30

        reward -= 1

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


eps = 1
eps_decay = 0.991
discount_factor = 0.9
learning_rate = 0.9

num_episodes = 1800
training_paths = []

env = Env(tileMap, eps, discount_factor, learning_rate)

rewards = []

for episode in range(num_episodes):
    if episode == 1700:
        eps = 0
    print(episode, ' : ', eps)
    r_index, c_index = env.get_starting_location()
    carr_index = 0
    s_row, s_column = r_index, c_index

    # print(start_row, start_column)

    training_path = []
    training_path.append([s_row, s_column])

    move_counter = 0
    total_reward = 0

    while not env.is_terminal_state(r_index, c_index, carr_index):
        if move_counter > 50:
            break
        move_counter += 1

        action_index = env.get_next_action(r_index, c_index, carr_index, eps)

        old_row_index, old_column_index, old_carry_index = r_index, c_index, carr_index
        r_index, c_index, carr_index, rew = env.next_state(r_index, c_index, carr_index, action_index)

        training_path.append([r_index, c_index])

        total_reward += rew

        old_q_value = env.q_values[old_row_index, old_column_index, old_carry_index, action_index]
        temporal_difference = rew + discount_factor * np.max(
            env.q_values[r_index, c_index, carr_index]) - old_q_value

        new_q_value = old_q_value + learning_rate * temporal_difference

        env.q_values[old_row_index, old_column_index, old_carry_index, action_index] = new_q_value

        env.visual_update()
        env.print_env()

    print(total_reward)
    rewards.append(total_reward)
    env.reset_env()
    eps *= eps_decay
    training_paths.append(training_path)

print('Training complete!')

with open("Q-matrix.txt", "w") as f:
    f.write(env.q_values.__str__())

export_path_to_simulation_format(training_paths, "training-paths.txt")
# path = get_shortest_path(1, 0)
# export_path_to_simulation_format([path], "final-path.txt")
# print(path)

plt.plot(range(0, num_episodes, 1), rewards)
plt.show()