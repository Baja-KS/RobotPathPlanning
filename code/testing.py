import numpy as np


def manhatn(row_index, column_index, target_index_x, target_index_y, box_index_x, box_index_y):
    lr_target = column_index - target_index_y
    ud_target = row_index - target_index_x

    lr_box = abs(column_index - box_index_y)
    ud_box = abs(row_index - box_index_x)

    t = 0
    b = 0

    if lr_target >= 0:
        if ud_target >= 0:
            if lr_target >= ud_target:
                t = 3
            else:
                t = 0
        else:
            if lr_target >= abs(ud_target):
                t = 3
            else:
                t = 2
    else:
        if ud_target >= 0:
            if abs(lr_target) >= ud_target:
                t = 1
            else:
                t = 0
        else:
            if abs(lr_target) >= abs(ud_target):
                t = 1
            else:
                t = 2

    if lr_box >= 0:
        if ud_box >= 0:
            if lr_box >= ud_box:
                b = 3
            else:
                b = 0
        else:
            if lr_box >= abs(ud_box):
                b = 3
            else:
                b = 2
    else:
        if ud_target >= 0:
            if abs(lr_box) >= ud_box:
                b = 1
            else:
                b = 0
        else:
            if abs(lr_box) >= abs(ud_box):
                b = 1
            else:
                b = 2

    return t, b

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


def get_view(row_index, column_index):
    view = np.zeros(8, dtype='int').tolist()

    for i in range(3):
        view[i] = tileMap[row_index - 1][column_index + i - 1]

    view[3] = tileMap[row_index][column_index - 1]
    view[4] = tileMap[row_index][column_index + 1]

    for i in range(3):
        view[i + 5] = tileMap[row_index + 1][column_index + i - 1]

    return view


vm = get_view(1, 3)

dims = []
for i in range(8):
    dims.append(2)
dims.append(20)
dims.append(20)
dims.append(20)
dims.append(20)
dims.append(4)

q = np.zeros(dims, dtype='short')
print(q.shape)
x = np.array([1, 2, 3, 4], dtype='int')
print(np.random.choice(x, 1))
#q[0, :, :, :, :, :, :, :, :, :, :, :, 0] = -10000
#q[:, :, 0, :, :, :, :, :, :, :, :, :, 1] = -10000
#q[:, :, :, :, 0, :, :, :, :, :, :, :, 2] = -10000
#q[:, :, :, :, :, :, 0, :, :, :, :, :, 3] = -10000
