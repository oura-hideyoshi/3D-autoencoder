# 3D-autoencoder
3D オートエンコーダ

## 実行環境
[env_inf.yml](./env_inf.yml)　参照
## 仕様
大抵のconfigはmain.py内のconfigで変更可能
### モードの選択

`mode = ["train", "test", "save_input"]`

|モード|内容|
|:---:|:---|
|train|学習モード. 学習データセットパスからデータを読み込み、モデルの学習を行う|
|test|テストモード. テストデータセットパスからデータを読み込み、予測を行う|
|save_input|入力データセットを.matファイルにしてまとめて保存する. デバッグ用|

### データの保存先
基本的にデフォルトのままでよい
``` 
# save path
save_root_path = "result/test"
checkpoints_path = join(save_root_path, 'checkpoints')
sample_path = join(save_root_path, "sample")
activation_path = join(save_root_path, "activation")
log_path = join(save_root_path, "logs")
input_sample_path = join(save_root_path, "input_sample")
xlsx_path = join(save_root_path, "history.xlsx")
```
|パス|保存内容|
|:---:|---|
|save_root_path|保存パスのルートパス. 学習保存先はここのみ変更で可能|
|checkpoints_path|学習モデルを一定エポックごとに保存する. **未実装**|
|sample_path|モデルの出力|
|activation_path|入力データセットを.matファイルにしてまとめて保存する. デバッグ用|
|log_path|keras.callbacks.TensorBoardのlog_dir|
|input_sample_path|modeでsave_inputを指定している場合の、入力データの保存先|
|xlsx_path|学習経過ファイルの保存先.ディレクトリ名ではなくファイル名.|

### データセット
```
# dataset
train_x_path = '../dataset2D/ReconData5set/256x256pix/train/image'
train_y_path = '../dataset2D/ReconData5set/256x256pix/train/label'
test_x_path = '../dataset2D/ReconData5set/256x256pix/test/image'
test_y_path = '../dataset2D/ReconData5set/256x256pix/test/label'
```
|パス|内容|
|:---:|---|
|train_x_path|トレーニングXデータ|
|train_y_path|トレーニングYデータ|
|test_x_path|テストXデータ|
|test_y_path|テストYデータ|

### パラメータ
```
# for train parameter
im_dim = (256, 256, 1)  # 入力データサイズ
epochs = 1000
batch_size = 40
Adam_lr = 1e-4
Adam_beta = 0.9
# clip_dim = (32, 32, 1)  # 入力データからクリップするサイズ = モデルの入力サイズ
clip_dim = None
clip_num = 10
train_weight_path = None
validation_split = 0.1
```
|パラメータ名|内容|
|:---:|---|
|im_dim|入力データサイズ. グレースケールは1ch扱い|
|apochs|エポック数|
|batch_size|バッチサイズ|
|Adam_lr|Adamの学習係数|
|Adam_beta|Adamの学習係数β1|
|clip_dim|画像の一部の領域を切り抜いて学習させる際のサイズ. Noneなら画像そのまま|
|clip_num|バッチごとに画像をいくつ切り抜くか|
|train_weight_path|学習の際に用いるモデルの重みファイルのパス. Noneの場合は1から学習|
|validation_split|トレーニングデータセットのうち一部を評価用にする際の割合|

### テストのときのみ使うやつ
```
# use only for testing
test_weight_path = join(save_root_path, "weight.h5")
```
|変数|内容|
|:---:|---|
|test_weight_path|テストモードの際に用いる重みファイル|

## データ詳細
**tensorboard を使った、学習過程の可視化**

`tensorboard --logdir=/full_path_to_your_logs`