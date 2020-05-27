import numpy as np

import utils


class ConvCodec():

    def __init__(self, M):
        self.M = M
        self.c_count = np.shape(M)[0]
        self.c_length = np.shape(M)[1]
        self.transitions = [[] for _ in range(2)]
        self.pre_computed_output = self.pre_compute_outputs()

    def pre_compute_outputs(self):
        pre_comp_out = []
        for state in range(2 ** self.c_length):
            state_arr = np.array(list(map(int, list(("{0:0" + str(self.c_length) + "b}").format(state)))))
            pre_comp_out.append(self.compute_output(state_arr))
        return pre_comp_out

    def compute_output(self, full_state):
        """prev_state: shape = (m0-1,)"""
        return utils.transform(np.remainder(np.dot(self.M, full_state), 2))

    def decode(self, input_seq):
        if self.c_length > 1:
            path_and_metrics_template = [{"path": [], "metric": -np.inf} for _ in range(2 ** (self.c_length - 1))]
            path_and_metrics_template[0]["metric"] = 0
            path_and_metrics_template_2 = [{"path": [], "metric": -np.inf} for _ in range(2 ** (self.c_length - 1))]
            path_and_metrics_template_2[0]["metric"] = 0

            double_paths = [path_and_metrics_template, path_and_metrics_template_2]

            for idx in range(len(input_seq)//self.c_count):
                msg_block = input_seq[idx * self.c_count:(idx+1) * self.c_count]
                new_path_and_metrics = double_paths[idx % 2]
                path_and_metrics = double_paths[(idx+1) % 2]

                if len(msg_block) < self.c_count:
                    break

                get_full_state_metrics = np.inner(self.pre_computed_output, msg_block)

                for state in range(2 ** (self.c_length - 1)):
                    new_bit = int(state % 2)
                    prev_state_plus = state >> 1
                    prev_state_minus = (state >> 1) + 2 ** (self.c_length - 2)
                    edge_metric_plus = get_full_state_metrics[2 * prev_state_plus + new_bit]
                    edge_metric_minus = get_full_state_metrics[2 * prev_state_minus + new_bit]
                    future_metric_plus = path_and_metrics[prev_state_plus]["metric"] + edge_metric_plus
                    future_metric_minus = path_and_metrics[prev_state_minus]["metric"] + edge_metric_minus

                    if future_metric_minus > future_metric_plus:
                        new_path_and_metrics[state]["path"] = path_and_metrics[prev_state_minus]["path"] + [new_bit]
                        new_path_and_metrics[state]["metric"] = future_metric_minus
                    else:
                        new_path_and_metrics[state]["path"] = path_and_metrics[prev_state_plus]["path"] + [new_bit]
                        new_path_and_metrics[state]["metric"] = future_metric_plus

            result = utils.transform(np.array(new_path_and_metrics[0]["path"][:(-(self.c_length - 1))]))
        else:
            redundant_terms = len(input_seq) % self.c_count
            if redundant_terms > 0:
                result = np.sign(np.sum(np.reshape(input_seq[:(-redundant_terms)], (-1, self.c_count)), 1))
            else:
                result = np.sign(np.sum(np.reshape(input_seq, (self.c_count, -1)), 0))

        return result

    def encode(self, symbol_seq):
        bit_seq = np.concatenate([utils.inv_transform(symbol_seq), np.zeros(self.c_length - 1, dtype=np.int32)])
        "M = matrix w/ shape (n0xk0xm0)"
        "n0 =# of symbols exiting per epoch"
        "k0 = # of bits entering per epoch"
        "m0 = # of bits used to produce an output"
        "b = input vector"
        "assume initial state (1...1), add (1...1) to the end"
        x = np.zeros(len(bit_seq) * self.c_count, dtype=np.int32)
        state = 0
        for k in range(0, len(bit_seq)):
            full_state = 2*state + bit_seq[k]
            x[k * self.c_count: (k+1) * self.c_count] = self.pre_computed_output[full_state]
            state = full_state % (2**(self.c_length-1))

        return x
