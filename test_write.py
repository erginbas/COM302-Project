import random
import string
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from bio_codec import BioCodec


def run(*args):
    """Runs a shell command"""
    subprocess.run(' '.join(args), shell=True, check=True)


L = 10000
I = 6
W = 80
Nmax = 51200
margin = 5
R = int(np.floor(Nmax / ((2 ** I) * (margin + W) * 8 / (I + 1))))
c = BioCodec(I)
N = int(R*(2**I)*np.ceil(8*W/(I+1)))


def random_word(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


print("I=", I, ", W=", W, ", margin=", margin, ", R=", R, ", N=", N)

word = random_word(W)
word_binary = c.str2bin(word)

sent = []
for k in range(0, W * 8, I + 1):
    message = word_binary[k:k + I + 1]
    sent.extend(c.encode(message))

input_arr = np.repeat(sent, R)

np.savetxt("input.txt", input_arr)

run("python", "client.py", "--input_file=input.txt", "--output_file=output.txt",
    "--srv_hostname=iscsrv72.epfl.ch", "--srv_port=80")

#output_arr = input_arr + np.sqrt(12) * np.random.randn(len(input_arr))

output_arr = np.loadtxt("output.txt")


received = np.mean(np.reshape(output_arr, (-1, R)), axis=1)

output = ""
for k in range(0, len(received), (2**I)):
    output += c.decode(received[k:k+(2**I)])

print("sent bits:\t\t", word_binary)
print("received bits:\t", output)
print(len(word_binary), len(output))

print("sent:\t\t ", word)
print("received:\t ", c.bin2str(output))
print(c.bin2str(output) == word)

# A = np.fft.fft(input_arr)
# B = np.fft.fft(output_arr)
# A = np.fft.fftshift(A)
# B = np.fft.fftshift(B)
#
# plt.subplot(121)
# plt.plot(np.abs(A), label="input")
# plt.plot(np.abs(B), label="output")
# plt.legend()
# plt.subplot(122)
# plt.plot(input_arr, label="input")
# plt.plot(output_arr, label="output")
# plt.legend()
# plt.show()
