import warnings

from scipy.io import savemat
from model import ModelGenerator
from keras.utils import plot_model
from DataLoader import DataLoader
from util import rescale, Converter
from util.cell_tool import *
import os

from loss import *

# TODO GPUからCPUへの切り替えを実現
# GPUの設定
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


def test(cfg):
    # -------- create dirs --------
    if not os.path.isdir(cfg.save_root_path):
        os.makedirs(cfg.save_root_path)
        os.makedirs(cfg.checkpoints_path)
        os.makedirs(cfg.sample_path)
        os.makedirs(cfg.activation_path)
        os.makedirs(cfg.input_sample_path)
        os.makedirs(cfg.save_root_path + "/logs/train/plugins/profile")
        os.makedirs(cfg.save_root_path + "/screenshot")

    # -------- build model --------
    model = ModelGenerator().build_UNet_3L(im_dim=cfg.im_dim, clip_dim=cfg.clip_dim, ini_f=4, k_size=[4, 4, 8])
    print('<test> Generating model ...')
    model.summary()
    plot_model(model, to_file=cfg.save_root_path + "/model.png")
    try:
        model.load_weights(cfg.test_weight_path)
        print("Sat weight to model.")
    except (FileNotFoundError, OSError) as e:
        print(e, "cfg.test_weight_path:", cfg.test_weight_path)
        raise
    print("<test> Generating model ... Complete !")

    # -------- load & create dataset --------
    print("<test> Loading dataset ...")
    data_loader_x = DataLoader(cfg.test_x_path, cfg.im_dim)
    test_x = np.array(data_loader_x.load_data()).astype(np.float32)
    test_x = rescale.rescale(test_x)
    test_x = np.expand_dims(test_x, axis=4)
    print("train X shape :", test_x.shape)
    if cfg.test_y_path is not None:
        data_loader_y = DataLoader(cfg.test_y_path, cfg.im_dim)
        test_y = np.array(data_loader_y.load_data()).astype(np.float32)
        test_y = rescale.rescale(test_y)
        test_y = np.expand_dims(test_y, axis=4)
        print("train Y shape :", test_y.shape)
    print("<test> Loading dataset ... Complete !")

    if "save_input" in cfg.mode:
        savemat(cfg.input_sample_path + "/test_x.mat", {"test_x": test_x})

    # -------- predict --------
    print("<test>  predicting ...")
    if cfg.clip_dim is None:
        predicted = model.predict(test_x, verbose=1, batch_size=1)
        predicted = np.squeeze(predicted)
        make_intermediate_images(cfg.activation_path, model, test_x[[1]])
    else:
        cell_size = cfg.clip_dim
        celled_test_X = [vox2cell(test_x[i], cell_size) for i in range(test_x.shape[0])]

        celled_generated_volumes = [model.predict(celled_test_X[i], verbose=1) for i in range(test_x.shape[0])]
        predicted = [cell2vox(celled_generated_volumes[i], (4, 4, 4, 1)) for i in range(test_x.shape[0])]
        predicted = np.squeeze(predicted)
        make_intermediate_images(cfg.activation_path, model, celled_test_X[0][[5]])
    if not os.path.isdir(cfg.sample_path):
        os.makedirs(cfg.sample_path)
    for i in range(len(test_x)):
        savemat(cfg.sample_path + '/test_' + str(i) + '.mat', {"voxel": predicted[i]})


def make_intermediate_images(path, model, voxel):
    from keras import models
    from keras.layers import MaxPooling3D, PReLU, BatchNormalization
    import math

    # create model
    layers = model.layers[1:]
    layer_outputs = [layer.output for layer in layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activation_model.summary()

    activations = activation_model.predict(voxel)
    for i, activation in enumerate(activations):
        print("%2d: %s" % (i, str(activation.shape)))

    activations = [(layer.name, activation) for layer, activation in zip(layers, activations)
                   if isinstance(layer, BatchNormalization)
                   ]

    # 出力層ごとに特徴画像を並べてヒートマップ画像として出力
    if not os.path.isdir(path):
        os.makedirs(path)
    for i, (name, activation) in enumerate(activations):
        savemat(path + "/" + str(i) + "_" + name + ".mat", {"voxel": np.squeeze(activation[0])})

        # max = np.max(activation[0])
        # fs = activation[0].transpose(2, 0, 1)   # (intensity, width, height)
        # fs = sorted(fs, key=lambda x: -np.mean(x))[:5]
        # for j, f in enumerate(fs):
        #   plt.figure()
        #   sns.heatmap(f, vmin=0, vmax=max, xticklabels=False, yticklabels=False, square=True, cbar=False)
        #   plt.savefig("%s-%d.png" % (name, j+1))
