import binascii

import numpy as np


def transform(x):
    y = x.copy()
    y[x == 1] = -1
    y[x == 0] = +1
    return y


def inv_transform(x):
    y = x.copy()
    y[x == +1] = 0
    y[x == -1] = 1
    return y


def str2bin(string):
    word_int_arr = [int(binascii.hexlify(char.encode('ascii')), 16) for char in string]
    binary = []
    for char_int in word_int_arr:
        binary.extend(list(map(int, list("{0:08b}".format(char_int)))))
    return np.array(binary)


def bin2str(binary):
    try:
        return binascii.unhexlify('%x' % int(binary[(len(binary) % 8):], 2)).decode('ascii')
    except UnicodeDecodeError:
        return "*"
