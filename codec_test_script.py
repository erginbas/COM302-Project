import random
import string

import numpy as np

import bio_codec
import conv_codec
import utils


def random_word(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


L = 10000
I = 0
W = 80
R = 5

#bio_codec = codec.Codec(I)

M = np.array([[1, 0, 0, 0, 1, 1, 1, 1],[1, 0, 0, 1, 1, 1, 0, 1],[1, 0, 1, 0, 1, 0, 0, 1],[1, 0, 1, 1, 0, 1, 1, 1],[1, 0, 1, 1, 1, 1, 1, 1],
              [1, 1, 0, 0, 0, 1, 0, 1],[1, 1, 0, 1, 0, 1, 1, 1],[1, 1, 0, 1, 1, 1, 1, 1],[1, 1, 1, 1, 0, 1, 0, 1],[1, 1, 1, 1, 1, 0, 1, 1]])

#M = np.array([[1, 0, 1, 0, 0, 1, 1, 1],[1, 0, 1, 1, 1, 0, 0, 1],[1, 1, 0, 1, 1, 0, 1, 1],[1, 1, 1, 1, 1, 1, 0, 1]])
conv_codec = conv_codec.ConvCodec(M)

print("W=", W, " I=", I, " R=", R, " M0=",  M.shape[0], " N=", R * (2 ** I) * M.shape[0] * (8 * W) // (I + 1))

err_text_count = 0
err_block_count = 0
err_bit_count = 0
for j in range(L):
    word = random_word(W)
    word_bin = utils.str2bin(word)
    word_symbol_arr = utils.transform(word_bin)

    conv_encoded_symbols = conv_codec.encode(word_symbol_arr)

    # encoded_messages = []
    # for k in range(0, len(conv_encoded_symbols), I + 1):
    #     message = conv_encoded_symbols[k:k + I + 1]
    #     encoded_msg = bio_codec.encode(message)
    #     encoded_messages.append(encoded_msg)
    # bio_encoded = np.concatenate(encoded_messages)

    repetition_encoded = np.repeat(conv_encoded_symbols, R)

    received = repetition_encoded + np.random.normal(0, np.sqrt(12), len(repetition_encoded))

    repetition_decoded = np.mean(np.reshape(received, (-1, R)), axis=1)

    # decoded_messages = []
    # for k in range(0, len(repetition_decoded), (2 ** I)):
    #     block = repetition_decoded[k:k + (2 ** I)]
    #     decoded_msg = bio_codec.soft_decode(block)
    #     decoded_messages.append(decoded_msg)
    # bio_decoded = np.concatenate(decoded_messages)

    conv_decoded = conv_codec.decode(repetition_decoded)

    if np.any(conv_decoded[:len(word_symbol_arr)] != word_symbol_arr):
        err_text_count += 1
        # err_block_count += np.sum(repetition_decoded != encoded)
        err_bit_count += np.sum(conv_decoded[:len(word_symbol_arr)] != word_symbol_arr)

    print("\rIteration: ", j, end="\t")
    print("\tRatio of trials with error: {:4f}".format(err_text_count / (j + 1)), end="\t")
    # print("\tBlock error probability: {:4f}".format(err_block_count / (len(repetition_decoded) * (j + 1))), end="\t")
    print("\tBit error probability: {:8f}".format(err_bit_count / ((j + 1) * (8 * W))), end="")
