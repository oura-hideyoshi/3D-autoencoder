from pathlib import Path

import cv2
from cv2 import imread
import numpy as np
import os
from scipy.io import savemat


class Converter:
    def __init__(self):
        pass

    @staticmethod
    def tiff2mat(data_path, save_path):
        """set many tiff images to a set of .mat volume
        dir
        ┣1.tif
        ┣2.tif
        ┣3.tif
        when each tiff has (128*128)pix, shape this method returns is (128*128*images_number)


        Examples
        ----------
        Converter.tiff2mat("./dir1/dir2", "./save_dir")
        load from dir2,
        and save ./save_dir/dir2.mat
        so, .mat name will be set automatically from data_path name

        Parameters
        ----------
        data_path: str
            data directory path including tiff images
        save_path: str
            save directory path.

        Returns
        -------
        volume: ndarray

        """
        volume = []
        if not os.path.exists(data_path):
            raise NotADirectoryError(data_path + " is not found.")
        mat_name = os.path.splitext(os.path.basename(data_path))[0]

        print("Getting tiff files from : " + data_path + "/*.tiff or /*.tif", end="")
        for file in Path(data_path).glob('*.tiff'):
            data = imread(str(file), cv2.IMREAD_GRAYSCALE)
            volume.append(data)
        for file in Path(data_path).glob('*.tif'):
            data = imread(str(file), cv2.IMREAD_GRAYSCALE)
            volume.append(data)

        volume = np.array(volume)
        volume = np.transpose(volume, [1, 2, 0])
        savemat(save_path + os.sep + mat_name + ".mat", {"voxel": volume})
        print(" -> Saved " + mat_name + ".mat")
        # print("Loaded", volume.shape, " dataset.")
        return volume

    @staticmethod
    def dirs2mat(_data_path, _save_path):
        for _dir in os.listdir(data_path):
            # print(data_path + os.sep + _dir)
            if os.path.isdir(data_path + os.sep + _dir):
                data_dir = data_path + os.sep + _dir
                Converter.tiff2mat(data_dir, save_path)

    @staticmethod
    def list2xlsx(_list, _save_path):
        import pandas as pd
        df = pd.DataFrame(data=_list, columns=None)
        df.to_excel(_save_path)


if __name__ == "__main__":
    data_path = "../../dataset3D/ReconData5set_shortBone/train/label"
    save_path = data_path
    Converter.dirs2mat(data_path, save_path)
