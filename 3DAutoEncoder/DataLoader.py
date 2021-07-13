import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat
import os


class DataLoader:
    def __init__(self, path, im_dim):
        """set train_path

        Parameters
        ----------
        path: str
            dataset path.
            (expected: receive from args.train_noise_path or args.train_real_path)

        """
        self.path = path
        self.im_dim = im_dim
        # self.dataset = args.dataset

    def load_data(self):
        """load data from train_path.
        *.mat must have "voxel" attr.

        Returns
        -------
        data_set: ndarray
            ndarray of 3d data. (index, dim, dim, dim)
        """

        data_set = []
        for file in Path(self.path).glob('*.mat'):
            print('Loading volume... ' + str(file))
            data = loadmat(str(file))
            if data["voxel"].shape != self.im_dim[:3]:
                print(data["voxel"].shape, self.im_dim[:3])
                raise ValueError("Loaded data shape doesn't matches.")
            data_set.append(data["voxel"])

        print("Loaded data from", os.path.abspath(self.path))
        return data_set
        # get file length
    # raw = np.load(self.train_path+'/'+self.dataset+'.mat')
    # print('Loaded data with '+str(raw.shape[0])+'objects')

if __name__ == "__main__":
    dataloader = DataLoader("../../dataset3D/ReconData5set/train/image", 128)
    data = dataloader.load_data()
    pass
