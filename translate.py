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

MAX_LEN = 10
NUM_CLASSES = 255

LAYERS = 8
NUM_HEADS = 4
EMBED_DIM = NUM_HEADS * 16

TRAIN_STEPS = 100
TRAIN_LOOPS = 100


def int_of_char(c):
    i = ord(c)
    if i >= NUM_CLASSES:
        i = 88  # "X" character
    return i


def ints_of_str(s):
    return [int_of_char(c) for c in s]


def str_of_ints(ints):
    to_return = ""
    for i in ints:
        to_return += chr(i)
    return to_return


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
    model = Transformer(NUM_CLASSES, MAX_LEN, LAYERS, EMBED_DIM, NUM_HEADS, 32)

    # X_train, Y_train, X_test, Y_test = make_dataset()
    X_train, Y_train, X_test, Y_test = make_translation_dataset()
    lr = 0.003
    for i in range(TRAIN_LOOPS):
        optim = Adam(get_parameters(model), lr=lr)
        train(model, X_train, Y_train, optim, TRAIN_STEPS, BS=64)
        acc, Y_test_preds = evaluate(
            model, X_test, Y_test, num_classes=NUM_CLASSES, return_predict=True
        )
        lr /= 1.2
        print(f"reducing lr to {lr:.4f}")
        k = random.randint(0, len(Y_test_preds))
        print("Source: " + str_of_ints(X_test[k]))
        print("Target: " + str_of_ints(Y_test[k]))
        print("Pred:   " + str_of_ints(Y_test_preds[k]))
