from keras.layers.advanced_activations import ReLU, LeakyReLU, PReLU, Softmax
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dropout, Concatenate
from keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
from keras.layers.pooling import MaxPooling3D
from keras.models import Sequential, Model

"""
Generator and Discriminator
reference:
    Vox2Vox: 3D-GAN for Brain Tumour Segmentation
    https://arxiv.org/abs/2003.13653
    Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling
    https://arxiv.org/abs/1610.07584
"""


class ModelGenerator:
    """
    モデルを返すクラス
    モデルのプロパティはモデルごとで決定
    """

    def build_UNet_3L(self, im_dim, clip_dim=None, ini_f=8, k_size=[3, 3, 6]):
        """
        U-Netを模倣した、深さ３層のU-Net的なモデル
        1層目のフィルター数はini_fで指定され、下の層に進むたびに２倍、上がるたびに1/2倍される
        Parameters
        ----------
        im_dim: tuple
        clip_dim: tuple
        clip_num: int
        ini_f: int
        k_size: list

        Returns
        -------

        """

        layer_stack = []

        if clip_dim is None:
            input_layer = Input(shape=im_dim)
        else:
            input_layer = Input(shape=clip_dim)

        # model = BatchNormalization(name='batchNormE1-1')(input_layer)
        model = Conv3D(ini_f, k_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal',
                       name='ConvE1-1')(input_layer)
        # model = BatchNormalization(name='batchNormE1-2')(model)
        model = Conv3D(ini_f, k_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal',
                       name='ConvE1-2')(model)
        layer_stack.append(model)

        model = MaxPooling3D(pool_size=2, name='Pool1-2')(model)
        model = Conv3D(ini_f * 2, k_size, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal',
                       name='ConvE2-1')(model)
        # model = BatchNormalization(name='batchNormE2-1')(model)
        model = Conv3D(ini_f * 2, k_size, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal',
                       name='ConvE2-2')(model)
        # model = BatchNormalization(name='batchNormE2-2')(model)
        layer_stack.append(model)

        model = MaxPooling3D(pool_size=2, name='Pool2-3')(model)
        model = Conv3D(ini_f * 4, k_size, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal',
                       name='ConvE3-1')(model)
        # model = BatchNormalization(name='batchNormE3-1')(model)
        model = Conv3D(ini_f * 4, k_size, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal',
                       name='ConvE3-2')(model)
        # model = BatchNormalization(name='batchNormE3-2')(model)

        model = UpSampling3D(size=(2, 2, 2), name='up_sampling3-2')(model)
        model = Conv3D(ini_f * 2, k_size, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal')(model)
        model = Concatenate()([model, layer_stack.pop()])
        model = Conv3D(ini_f * 2, k_size, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal',
                       name='ConvD2-1')(model)
        # model = BatchNormalization(name='batchNormD2-1')(model)
        model = Conv3D(ini_f * 2, k_size, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal',
                       name='ConvD2-2')(model)
        model = BatchNormalization(name='batchNormD2-2')(model)

        model = UpSampling3D(size=(2, 2, 2), name='up_sampling2-1')(model)
        model = Conv3D(ini_f, k_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(
            model)
        model = Concatenate()([model, layer_stack.pop()])
        model = Conv3D(ini_f, k_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal',
                       name='ConvD1-1')(model)
        # model = BatchNormalization(name='batchNormD1-1')(model)
        model = Conv3D(ini_f * 2, k_size, strides=1, padding='same', activation='relu',
                       kernel_initializer='he_normal',
                       name='ConvD1-2')(model)
        # model = BatchNormalization(name='batchNormD1-2')(model)

        # model = Conv3D(2, k_size, strides=1, padding='same', kernel_initializer='he_normal',
        #                name='output')(model)
        # model = Softmax(axis=-1)(model)
        model = Conv3D(1, k_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal',
                       name='output')(model)

        return Model(input=input_layer, outputs=model, name='UNet_3L')


if __name__ == "__main__":
    m_g = ModelGenerator()
    model = m_g.build_UNet_3L([128, 128, 128, 1])
    model.summary()
