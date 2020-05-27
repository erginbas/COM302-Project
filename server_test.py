import numpy as np
from conv_codec import ConvCodec
import utils
import os

M = np.array([[1, 0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1, 1],
              [1, 0, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 0, 1, 1]])

conv_codec = ConvCodec(M)
text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque sed quam augue.'

R = 5

word_bin = utils.str2bin(text)
word_symbol_arr = utils.transform(word_bin)
conv_encoded_symbols = conv_codec.encode(word_symbol_arr)

repetition_encoded = np.repeat(conv_encoded_symbols, R)

np.savetxt('input.txt', repetition_encoded)

os.system("python3 client.py --input_file=input.txt --output_file=output.txt "
          "--srv_hostname=iscsrv72.epfl.ch --srv_port=80")

output = np.loadtxt('output.txt')

repetition_decoded = np.mean(np.reshape(output, (-1, R)), axis=1)

conv_decoded_symbol_arr = conv_codec.decode(repetition_decoded)
conv_decoded = utils.inv_transform(conv_decoded_symbol_arr)

bin_output = "".join(conv_decoded.astype(str).tolist())

str_output = utils.bin2str(bin_output)
str_output_array = np.array([c for c in str_output])
text_array = np.array([c for c in text])
error_letter_cnt = np.sum(str_output_array != text_array)
error_bit_cnt = np.sum(conv_decoded_symbol_arr != word_symbol_arr)

print(text)
print(str_output)
print('# of character errors = ' + str(error_letter_cnt) + ' out of ' + str(len(text_array)) + ' letters.')
print('# of bit errors = ' + str(error_bit_cnt) + ' out of ' + str(len(conv_decoded)) + ' bits.')
