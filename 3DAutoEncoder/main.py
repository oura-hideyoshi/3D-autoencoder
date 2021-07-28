from os.path import join
from train import train
from test import test


def main(cfg):
    if "train" in cfg.mode:
        train(cfg)

    if "test" in cfg.mode:
        test(cfg)


class config:
    mode = ["train", "test", "_save_input"]

    # save path
    save_root_path = "result/0728_spreadModel"
    checkpoints_path = join(save_root_path, 'checkpoints')
    sample_path = join(save_root_path, "sample")
    activation_path = join(save_root_path, "activation")
    log_path = join(save_root_path, "logs")
    input_sample_path = join(save_root_path, "input_sample")
    xlsx_path = join(save_root_path, "history.xlsx")

    # dataset
    train_x_path = '../dataset3D/ReconData5set_shortBone/z=32/image'
    train_y_path = '../dataset3D/ReconData5set_shortBone/z=32/label'
    test_x_path = '../dataset3D/ReconData5set_shortBone/z=32/image'
    test_y_path = None

    # for train parameter
    im_dim = (128, 128, 32, 1)  # 入力データサイズ
    epochs = 1000
    batch_size = 5
    Adam_lr = 1e-4
    Adam_beta = 0.9
    # clip_dim = (32, 32, 32, 1)  # 入力データからクリップするサイズ = モデルの入力サイズ
    clip_dim = None
    clip_num = 10
    train_weight_path = None
    validation_split = 0.1

    # use only for testing
    test_weight_path = join(save_root_path, "weight.h5")


if __name__ == "__main__":
    cfg = config()
    main(cfg)
