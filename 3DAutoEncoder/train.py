import warnings

import keras.callbacks
from scipy.io import savemat

from model import ModelGenerator
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
from keras.metrics import Precision, Recall
from DataLoader import DataLoader
from util import rescale, Converter
from util.Converter import Converter
import os
import numpy as np
import tensorflow as tf
from loss import *
import pandas as pd
from DataGenerator import generator3D

# TODO GPUからCPUへの切り替えを実現
# GPUの設定
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


def train(cfg):
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
    model = ModelGenerator().build_point_spread2(im_dim=cfg.im_dim, clip_dim=cfg.clip_dim)
    print('<train> Generating model ...')
    model.summary()
    plot_model(model, to_file=cfg.save_root_path + "/model.png")
    print('<train> Generating model ... Complete !')

    # -------- compile model --------
    print("<train> Compiling model ...")
    try:
        if cfg.train_weight_path is not None:
            model.load_weights(cfg.train_weight_path)
            print("Sat weight to model.")
    except FileNotFoundError as e:
        print(e, "cfg.train_weight_path:", cfg.train_weight_path)
        raise
    optimizer = Adam(lr=cfg.Adam_lr, beta_1=cfg.Adam_beta)
    metrics = ['accuracy', 'mse', ssim_loss, LWE]
    model.compile(loss=imbalanced_loss, optimizer=optimizer, metrics=metrics)
    print("<train> Compiling model ... Complete !")

    # -------- load & create dataset --------
    print("<train> Loading dataset ...")
    data_loader_x = DataLoader(cfg.train_x_path, cfg.im_dim)
    data_loader_y = DataLoader(cfg.train_y_path, cfg.im_dim)
    train_x = np.array(data_loader_x.load_data()).astype(np.float32)
    train_x = rescale.rescale(train_x)
    train_x = np.expand_dims(train_x, axis=4)
    train_y = np.array(data_loader_y.load_data()).astype(np.float32)
    train_y = rescale.rescale(train_y)
    train_y = np.expand_dims(train_y, axis=4)
    # from keras.utils.np_utils import to_categorical
    # train_y = to_categorical(train_y)
    print("train X shape :", train_x.shape)
    print("train Y shape :", train_y.shape)
    print("<train> Loading dataset ... Complete !")

    if "save_input" in cfg.mode:
        savemat(cfg.input_sample_path + "/train_x.mat", {"train_x": train_x})
        savemat(cfg.input_sample_path + "/train_y.mat", {"train_y": train_y})

    # -------- fitting model --------
    if train_x.shape[0] < cfg.batch_size:
        raise ValueError("config 'batch_size' must be under the train samples number")
    callback_tensorBoard = keras.callbacks.TensorBoard(log_dir=cfg.log_path, histogram_freq=1,
                                                       batch_size=cfg.batch_size,
                                                       write_graph=True, write_grads=True, write_images=True,
                                                       embeddings_freq=0, embeddings_layer_names=None,
                                                       embeddings_metadata=None
                                                       )
    # tensorboard --logdir=/full_path_to_your_logs
    callback_checkpoint = keras.callbacks.ModelCheckpoint(filepath=cfg.save_root_path + '/weight.h5', monitor='loss',
                                                          save_best_only=True, save_weights_only=True)
    callbacks = [callback_tensorBoard, callback_checkpoint]

    if cfg.clip_dim is None:
        hist = model.fit(train_x, train_y, batch_size=cfg.batch_size, epochs=cfg.epochs,
                         callbacks=[callback_tensorBoard, callback_checkpoint], validation_split=cfg.validation_split)
    else:
        dataGenerator = generator3D(x_set=train_x, y_set=train_y, batch_size=cfg.batch_size, clip_dim=cfg.clip_dim,
                                    clip_num=cfg.clip_num)
        hist = model.fit_generator(generator=dataGenerator, epochs=cfg.epochs, callbacks=callbacks)

    model.evaluate(train_x, train_y, batch_size=1, verbose=1, sample_weight=None, steps=None)

    # -------- save history --------
    save_history_path = os.path.join(cfg.save_root_path, "metrics.xlsx")
    data = []
    index = []
    for metrics in hist.history:
        index.append(metrics)
        data.append(hist.history.get(metrics))
    df = pd.DataFrame(data=data, index=index, columns=None)
    df.to_excel(save_history_path)
