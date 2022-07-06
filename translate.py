#!/usr/bin/env python3

# Based on https://github.com/geohot/tinygrad/blob/master/examples/transformer.py
import os
import numpy as np
import random

from tinygrad.tensor import Device
from extra.utils import get_parameters
from extra.training import train, evaluate
from models.transformer import Transformer
from tinygrad.optim import Adam

# dataset idea from https://github.com/karpathy/minGPT/blob/master/play_math.ipynb
def make_dataset():
    ds = []
    for i in range(100):
        for j in range(100):
            s = i + j
            ds.append(
                [i // 10, i % 10, j // 10, j % 10, s // 100, (s // 10) % 10, s % 10]
            )
    random.shuffle(ds)
    ds = np.array(ds)
    ds_X = ds[:, 0:6]
    ds_Y = np.copy(ds[:, 1:])
    ds_X_train, ds_X_test = ds_X[0:8000], ds_X[8000:]
    ds_Y_train, ds_Y_test = ds_Y[0:8000], ds_Y[8000:]

    return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test


MAX_LEN = 6
NUM_CLASSES = 255

def int_of_char(c):
    i = ord(c)
    if i >= NUM_CLASSES:
        i = 88 # "X" character
    return i

def ints_of_str(s):
    return [int_of_char(c) for c in s]

def make_translation_dataset():
    """
    x = list()
    for i in range(10000):
        x.append("HelloX")
        x.append("DogXXX")

    y = list()
    for i in range(10000):
        y.append("HolaXX")
        y.append("PerroX")
    """
    x = list()
    with open("data/source") as source_file:
        for line in source_file:
            while len(line) < MAX_LEN:
                line += " "
            x.append(line)
    y = list()
    with open("data/target") as target_file:
        for line in target_file:
            while len(line) < MAX_LEN:
                line += " "
            y.append(line)

    for i in range(len(x)):
        x[i] = x[i][:MAX_LEN]
        y[i] = y[i][:MAX_LEN]

    x = list(map(ints_of_str, x))
    y = list(map(ints_of_str, y))

    test_size = 2000

    ds_X_train = x[test_size:]
    ds_Y_train = y[test_size:]
    ds_X_test = x[:test_size]
    ds_Y_test = y[:test_size]

    ds_X_train = np.array(ds_X_train)
    ds_Y_train = np.array(ds_Y_train)
    ds_X_test = np.array(ds_X_test)
    ds_Y_test = np.array(ds_Y_test)

    return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test


from tinygrad.optim import Adam

if __name__ == "__main__":
    model = Transformer(NUM_CLASSES, MAX_LEN, 2, 128, 4, 32)

    #X_train, Y_train, X_test, Y_test = make_dataset()
    X_train, Y_train, X_test, Y_test = make_translation_dataset()
    lr = 0.003
    for i in range(10):
        optim = Adam(get_parameters(model), lr=lr)
        train(model, X_train, Y_train, optim, 50, BS=64)
        acc, Y_test_preds = evaluate(
            model, X_test, Y_test, num_classes=NUM_CLASSES, return_predict=True
        )
        lr /= 1.2
        print(f"reducing lr to {lr:.4f}")
        if acc > 0.998:
            wrong = 0
            for k in range(len(Y_test_preds)):
                if (Y_test_preds[k] != Y_test[k]).any():
                    wrong += 1
                    a, b, c, x = (
                        X_test[k, :2],
                        X_test[k, 2:4],
                        Y_test[k, -3:],
                        Y_test_preds[k, -3:],
                    )
                    print(
                        f"{a[0]}{a[1]} + {b[0]}{b[1]} = {x[0]}{x[1]}{x[2]} (correct: {c[0]}{c[1]}{c[2]})"
                    )
            print(f"Wrong predictions: {wrong}, acc = {acc:.4f}")
