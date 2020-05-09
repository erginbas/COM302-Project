import numpy as np

import utils


class BioCodec:
    def __init__(self, i=0):
        self.i = i
        self.M = BioCodec.generate_m(i)

    def encode(self, y):
        num_missing_elements = self.i - (len(y) - 1) % (self.i + 1)

        if num_missing_elements > 0:
            missing_elements = np.repeat(+1, num_missing_elements)
            y = np.concatenate([y, missing_elements])

        ind = int("".join(map(str, utils.inv_transform(y[1:]))), 2)
        sign = y[0]
        return sign * self.M[ind, :]

    def decode(self, y):
        u = np.dot(self.M, y)
        max_ind = int(np.argmax(np.abs(u)))
        sign_ele = np.sign(u[max_ind])
        binary_str = ("{0:0" + str(self.i + 1) + "b}").format(max_ind)
        symbol_array = utils.transform(np.array(list(map(int, list(binary_str)))))
        symbol_array[0] = sign_ele
        return symbol_array

    def soft_decode(self, y):
        u = np.dot(self.M, y)
        abs_metrics = np.abs(u)
        weighted_sum = np.zeros(self.i + 1)
        for idx, feasible_output in enumerate(u):
            binary_str = ("{0:0" + str(self.i + 1) + "b}").format(idx)
            symbol_array = utils.transform(np.array(list(map(int, list(binary_str)))))
            symbol_array[0] = np.sign(u[idx])
            weighted_sum = weighted_sum + abs_metrics[idx] * symbol_array
        return weighted_sum

    @staticmethod
    def generate_m(i: int) -> np.ndarray:
        if i == 0:
            return np.array([1])
        else:
            return np.block([[BioCodec.generate_m(i - 1), BioCodec.generate_m(i - 1)], [BioCodec.generate_m(i - 1), -BioCodec.generate_m(i - 1)]])
