#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e


EXAMPLE=examples/mnist # lmdb生成路径
DATA=data/mnist # 原始数据路径
BUILD=build/examples/mnist # 可执行二进制文件路径

BACKEND="lmdb" # 选择后端类型

echo "Creating ${BACKEND}..."

# 如果已经存在lmdb文件，则先删除
rm -rf $EXAMPLE/mnist_train_${BACKEND}
rm -rf $EXAMPLE/mnist_test_${BACKEND}

# 创建训练集lmdb文件
$BUILD/convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $EXAMPLE/mnist_train_${BACKEND} --backend=${BACKEND}

# 创建测试集lmdb文件
$BUILD/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $EXAMPLE/mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."
