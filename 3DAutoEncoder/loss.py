import tensorflow as tf
import keras.backend as K


def imbalanced_loss(y_true, y_pred):
    """
    不均衡データ対策loss
    y_true内に 1 が少ないため、以下のようなイメージの損失関数を実装する
    y_true(x,y,z,c) = 1 and y_pred(x,y,z,c) = 0 -> 損失を大きく
    y_true(x,y,z,c) = 0 and y_pred(x,y,z,c) = 1 -> 損失を小さく
    実際のデータはbinaryではないので、損失の計算は以下のようにする
    y_true, y_predのうち、y_true >= threshold となるvoxelの集合とそうでない集合に分ける.
    (2Dの場合のイメージ) threshold = 0.1, mul = 10
    y_true                  y_pred                  ->  bin_y_true                  error
    [0.046, 0.041, 0.155],  [0.146, 0.276, 0.020],      [False, False,  True],      [0.100, 0.235, 0.134],
    [0.144, 0.127, 0.198],  [0.134, 0.990, 0.132],      [ True,  True,  True],      [0.009, 0.863, 0.065],
    [0.011, 0.199, 0.084]   [0.698, 0.527, 0.660]]      [False,  True, False]       [0.687, 0.328, 0.576]

    T1P0_error              T0P1_error              ->  imbalanced_error        ->  loss : 15.617
    [0.   , 0.   , 0.134],  [0.100, 0.235, 0.   ],      [0.100, 0.235, 1.345],      MSE  : 3.002
    [0.009, 0.863, 0.065],  [0.   , 0.   , 0.   ],      [0.096, 8.639, 0.653],
    [0.   , 0.328, 0.   ]   [0.687, 0.   , 0.576]       [0.687, 3.282, 0.576]

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------

    """
    threshold = 0.1
    mul = 1000  # 倍率

    bin_y_true = y_true >= threshold
    bin_y_true = K.cast_to_floatx(bin_y_true)
    error = K.abs(y_true - y_pred)
    T1P0_error = error * bin_y_true
    T0P1_error = error * (1 - bin_y_true)
    imbalanced_error = T1P0_error * mul + T0P1_error
    loss = K.mean(imbalanced_error)

    return loss


def LWE(y_true, y_pred):
    weight = K.abs(y_true + y_pred)
    mag = 1 / K.sum(weight)

    e = K.abs(y_true - y_pred)
    weight_e = weight * e
    weight_e_sum = K.sum(weight_e)
    loss = weight_e_sum * mag
    return loss


def custom_ssim(y_true, y_pred):
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)


def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


if __name__ == '__main__':
    pass
