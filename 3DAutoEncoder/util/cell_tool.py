import numpy as np

@staticmethod
def vox2cell(vox, cell_size):
    """
    voxelをcell_size*cell_size*cell_sizeのセルに分解する
    第一にx方向に分解,次にy,最後にz
    Parameters
    ----------
    vox: np.ndarray(height, width, depth, channels)
        expected: height == width == depth, channels == 1
    cell_size: tuple(int, int, int, int) : (x,y,z,ch)

    Returns
    -------
    cells: np.ndarray(cell_num, cell_size, cell_size, cell_size, channels)
    cell_num: tuple(int, int, int, int)

    Examples
    -------
    voxel = np.zeros(128,128,128,1)
    cells = vox2cell(voxel, 32)
    cells: np.ndarray(4**3, 128,128,128,1)
    """
    cells = []
    vox_size = vox.shape
    cell_dim = cell_size
    # 各方向のcellの個数
    cell_x = vox_size[0] // cell_size[0]
    cell_y = vox_size[1] // cell_size[1]
    cell_z = vox_size[2] // cell_size[2]

    for z in range(cell_z):
        for y in range(cell_y):
            for x in range(cell_x):
                cell = vox[x * cell_dim[0]: (x + 1) * cell_dim[0],
                       y * cell_dim[1]: (y + 1) * cell_dim[1],
                       z * cell_dim[2]: (z + 1) * cell_dim[2],
                       :]
                cells.append(cell)
    return np.array(cells)

@staticmethod
def cell2vox(cells, cell_num):
    """
    cellをvoxに戻す
    Parameters
    ----------
    cells: np.ndarray(cell_num, cell_size, cell_size, cell_size, channels)
    cell_num: tuple(x, y, z)
        cellが各方向に何個並ぶか

    Returns
    -------
    voxel: np.ndarray(height, width, depth, channels)
    """
    cell_size = cells[0].shape
    voxel = np.zeros([cell_size[0] * cell_num[0], cell_size[1] * cell_num[1], cell_size[2] * cell_num[2], 1],
                     dtype='float').astype(np.float32)
    for z in range(cell_num[2]):
        for y in range(cell_num[1]):
            for x in range(cell_num[0]):
                cell = cells[z * cell_num[1] * cell_num[0] + y * cell_num[0] + x]
                voxel[x * cell_size[0]: (x + 1) * cell_size[0],
                y * cell_size[1]: (y + 1) * cell_size[1],
                z * cell_size[2]: (z + 1) * cell_size[2],
                :] = cell

    return voxel


