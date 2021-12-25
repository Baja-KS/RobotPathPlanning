import numpy as np
from yaml import safe_load

with open("params.yaml", "r") as f:
    config: dict = safe_load(f)

def read_map_file(path):
    f = open('./Map')

    rows = f.readlines()

    f.close()

    for i in range(len(rows)):
        rows[i] = rows[i].strip()
        rows[i] = rows[i].split(' ')

        rows[i] = list(map(lambda x: int(x), rows[i]))

    return np.array(rows)


tileMap = read_map_file('asdsadsad')
q_values = np.load('q_values')


def is_terminal_state(row_index, column_index, target_index_x, target_index_y, carry_state):
    if tileMap[row_index, column_index] == 0 \
            or (row_index == target_index_x and column_index == target_index_y and carry_state == 1):
        return True
    else:
        return False

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


def get_shortest_path(start_x, start_y, target_x, target_y, box_x, box_y):
    path = []
    if tileMap[start_x, start_y] == 0:
        return []

    row_index, column_index, target_index_x, target_index_y, \
    box_index_x, box_index_y, \
    carry_index = start_x, start_y, target_x, target_y, box_x, box_y, 0

    path.append([start_x, start_y])

    m_counter = 0
    t_reward = 0

    while not is_terminal_state(row_index, column_index,
                                     target_index_x, target_index_y, carry_index):
        if m_counter == 200:
            break

        m_counter += 1

        action = q_values[row_index, column_index,
                                      target_index_x, target_index_y,
                                      box_index_x, box_index_y, carry_index, -1]

        row_index, column_index, carry_index, reward = next_state(row_index, column_index,
                                                                       target_index_x, target_index_y,
                                                                       box_index_x, box_index_y,
                                                                       carry_index, action)

        path.append([row_index, column_index])
        t_reward += reward

    return path, t_reward


def test():
    test_episodes = 50000
    passed = 0

    for test_episode in range(test_episodes):
        # print('------------------------------------------------------------------------------')
        # print('Test: ' + str(test_episode))

        row_index, column_index = get_starting_location()
        target_index_x, target_index_y = get_target_location()
        box_index_x, box_index_y = get_box_location(target_index_x, target_index_y)

        # print(f'Starting location: [{row_index}, {column_index}]')
        # print(f'Target location: [{target_index_x}, {target_index_y}]')
        # print(f'Box location: [{box_index_x}, {box_index_y}]')

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

