import warnings

from tensorflow.keras.utils import Sequence
import numpy as np


class generator3D(Sequence):
    """
    データセットからバッチサイズ分だけデータを取得する.
    さらに、clip_sizeを指定することで、各データからランダムにclip_sizeの領域(りっぽい)を切り抜いたものを返す.
    すなわち、__getitem__ごとにclip_sizeの小さい領域をbatch_size*clip_sizeだけ生成する.
    """

    def __init__(self, x_set, y_set, batch_size, clip_dim=None, clip_num=1, under=0.0, upper=1.0, is_rot=False,
                 is_rand_choice=False, rand_choice=0.1):
        """

        Parameters
        ----------
        x_set: np.ndarray
            x dataset. [sample_num, height, width, depth, channels]
        y_set: np.ndarray
            y dataset. [sample_num, height, width, depth, channels]
        batch_size: int
            get_itemごとに取り出すサンプルの数. もしサンプル数よりもバッチの方が大きい場合はサンプル数に合わせる
        clip_dim: tuple
            サンプルから切り取る領域の大きさ.
        clip_num: int
            clipする数
        under: float
            y_setから切り取った領域に含まれる強度の下限.下限以下であった場合、そのセットは破棄される
        upper: float
            y_setから切り取った領域に含まれる強度の上限.上限以上であった場合、そのセットは破棄される
        is_rot: bool
            回転の変形を加えるかどうか
        is_rand_choice: bool
            下限・上限によって破棄されたデータでも確率的に有効にするかどうか
        rand_choice: float
            is_rand_choiceによって有効にする確率
        """
        if len(x_set) != len(y_set) or x_set[0].shape[:3] != y_set[0].shape[:3]:
            print("x data shape:", x_set[0].shape, "\n""y data shape:", y_set[0].shape)
            raise ValueError("Loaded data shape doesn't matches.")
        self.x, self.y = x_set, y_set
        self.data_dim = x_set[0].shape
        if len(x_set) < batch_size:
            warnings.warn("batch size must be under sample numbers. So, set sample number to batch size automatically.")
            self.batch_size = len(x_set)
        else:
            self.batch_size = batch_size
        if clip_dim is None:
            self.clip_dim = self.data_dim
        self.clip_dim = clip_dim
        self.clip_num = clip_num
        self.under = under
        self.upper = upper
        self.is_rot = is_rot
        self.is_rand_choice = is_rand_choice
        self.rand_choice_th = rand_choice
        self.max_v = self.clip_dim[0] * self.clip_dim[1] * self.clip_dim[2] * self.clip_dim[3]

    def __len__(self):
        """
        batchを何個渡すか
        Returns
        -------
        もしサンプル数が100個でbatch_sizeが16なら、
        ceil(100 / 16) = ceil(6.25) = 7
        """
        return int(np.ceil(len(self.x) // float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Generatorがバッチを作成するたびに呼ばれる関数.
        (恐らく)同一epoch内で__getitem__が呼ばれる度にidxが1ずつ増える
        Parameters
        ----------
        idx

        Returns
        -------
        batch_x: np.ndarray
            one batch. [batch_size*clip_num, [clip_size], 1]
        """
        # sampleからbatch_size分だけsampleを順に取得
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.is_rot:
            rand_rot = list(np.random.randint(low=0, high=4, size=self.batch_size * self.clip_num))
        else:
            rand_rot = list(np.zeros(self.batch_size * self.clip_num).astype("uint"))
        if self.is_rand_choice:
            rand_choice = list(np.random.rand(self.batch_size * self.clip_num))
        else:
            rand_choice = list(np.ones(self.batch_size * self.clip_num))

        # 切り取る領域の小さい方の値を生成
        clip_list = []
        for i in range(self.clip_num):
            clip_list.append([np.random.randint(low=self.data_dim[idx] - self.clip_dim[idx]) for idx in range(3)])

        clip_batch_x = []
        clip_batch_y = []
        count_rand = 0
        for i in range(self.batch_size):
            x = batch_x[i]
            y = batch_y[i]
            for j, c_a in enumerate(clip_list):
                clip_x = x[c_a[0]:c_a[0] + self.clip_dim[0], c_a[1]:c_a[1] + self.clip_dim[1],
                         c_a[2]:c_a[2] + self.clip_dim[2], :]
                clip_y = y[c_a[0]:c_a[0] + self.clip_dim[0], c_a[1]:c_a[1] + self.clip_dim[1],
                         c_a[2]:c_a[2] + self.clip_dim[2], :]
                if self.under <= (np.sum(clip_y) / self.max_v) <= self.upper:
                    rot = rand_rot.pop()

                    rot_x = np.rot90(clip_x, rot)
                    clip_batch_x.append(rot_x)
                    rot_y = np.rot90(clip_y, rot)
                    clip_batch_y.append(rot_y)

                elif rand_choice.pop() < self.rand_choice_th:
                    rot = rand_rot.pop()

                    rot_x = np.rot90(clip_x, rot)
                    clip_batch_x.append(rot_x)
                    rot_y = np.rot90(clip_y, rot)
                    clip_batch_y.append(rot_y)

                    count_rand += 1

        if len(clip_batch_x) == 0:
            clip_batch_x.append(clip_x)
            clip_batch_y.append(clip_y)

        if self.is_rand_choice or self.under != 0.0 or self.upper != 1.0:
            print("batch size:", len(clip_batch_x), '(' + str(count_rand) + ')', "/", self.batch_size * self.clip_num)
        return np.array(clip_batch_x), np.array(clip_batch_y)

    def on_epoch_end(self):
        pass
