import numpy as np


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

print(tileMap)

environment_rows = len(tileMap)
environment_columns = len(tileMap[0])

print(environment_rows, environment_columns)

q_values = np.zeros((environment_rows, environment_columns, 4))

actions = ['up', 'right', 'down', 'left']

rewards = np.full((environment_rows, environment_columns), -100.)

for i in range(environment_rows):
    for j in range(environment_columns):
        if tileMap[i][j] == 1:
            rewards[i][j] = -1

rewards[0, 3] = 100

print(rewards)


def is_terminal_state(current_row_index, current_column_index):
    if rewards[current_row_index, current_column_index] == -1.:
        return False
    else:
        return True


def get_starting_location():
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)

    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)
    return current_row_index, current_column_index


def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)


def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index


def get_shortest_path(start_row_index, start_column_index):
    if is_terminal_state(start_row_index, start_column_index):
        return []
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])

        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, 0.9)
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index,
                                                                        action_index)
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path


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
        lines.append(action_to_short(path[i - 1], path[i])+"\n")
    if not is_last:
        lines.append("---\n")
    return lines


def export_path_to_simulation_format(paths, file):
    try:
        with open(file, "w") as f:
            lines = []
            for i in range(len(paths)):
                lines += export_to_lines(paths[i], i == len(paths) - 1)
            f.writelines(lines)
    except:
        print("error writing to file")


epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9

# training_export=open("training.txt","w")
training_paths=[]
for episode in range(1000):
    row_index, column_index = get_starting_location()

    start_row, start_column=row_index,column_index

    while not is_terminal_state(row_index, column_index):
        action_index = get_next_action(row_index, column_index, epsilon)

        old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
        row_index, column_index = get_next_location(row_index, column_index, action_index)

        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value

    training_paths.append(get_shortest_path(start_row,start_column))
    # path=get_shortest_path(start_row,start_column)
    # training_export.writelines()

print('Training complete!')
export_path_to_simulation_format(training_paths,"training-paths.txt")
path=get_shortest_path(4, 0)
export_path_to_simulation_format([path],"final-path.txt")
print(path)
