import random
import string

import numpy as np

import conv_codec
import utils
from multiprocessing import Pool

def random_word(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


L = 10000
I = 0
W = 80
R = 5

M = np.array([[1, 0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1, 1],
              [1, 0, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 0, 1, 1]])

conv_codec = conv_codec.ConvCodec(M)

print("W=", W, " I=", I, " R=", R, " M0=", M.shape[0], " N=", R * (2 ** I) * M.shape[0] * (8 * W) // (I + 1))


def test_run():
    word = random_word(W)
    word_bin = utils.str2bin(word)
    word_symbol_arr = utils.transform(word_bin)

    conv_encoded_symbols = conv_codec.encode(word_symbol_arr)
    repetition_encoded = np.repeat(conv_encoded_symbols, R)

    received = repetition_encoded + np.random.normal(0, np.sqrt(12), len(repetition_encoded))

    repetition_decoded = np.mean(np.reshape(received, (-1, R)), axis=1)
    conv_decoded = conv_codec.decode(repetition_decoded)

    return np.sum(conv_decoded[:len(word_symbol_arr)] != word_symbol_arr)


wrong_msg = 0
wrong_bit = 0
test_count = 0

def printer(res):
    global wrong_bit
    global wrong_msg
    if res != 0:
        wrong_msg += 1
        wrong_bit += res
    print("\rWrong message count:", wrong_msg, " out of ", test_count, "Wrong bit count:", wrong_bit, " out of ", 8*W*test_count, end="")


workers = 5
pool = Pool(workers)
for i in range(L):
    processes = []
    for w in range(workers):
        test_count += 1
        p = pool.apply_async(test_run, callback=printer)
        processes.append(p)
    for p in processes:
        p.wait()
