# Image Splitter

## これは何？

画像をパーツ毎に分けたいツールです。

## 用意するもの

- pytorch

必須。学習するのに使います。

- python3-opencv

任意。学習に使用する画像を生成するのに使います。

- python3-onnx

任意。PytorchのモデルをONNXに変換するのに使います。

- onnxruntime

python3-onnxで作ったモデルを利用するのに使います。

- stb

C++で画像を読み書きするのに使います。

## How to build

```
$ git clone https://github.com/Tatakinov/image_splitter
$ cd image_splitter
$ cmake -S . -B build
$ cmake --build build
```

# Learning

1. `learn`ディレクトリ以下に`test`と`train`ディレクトリを作る

2. `learn`ディレクトリで`./learn.py`

3. 気に入ったモデルが出来上がったら`../utils/convert.py 気に入ったモデル`でonnxを作る
