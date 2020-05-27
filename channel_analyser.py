import numpy as np
import subprocess
import time


def run(*args):
    """Runs a shell command"""
    subprocess.run(' '.join(args), shell=True, check=True)


L = 10000
minutes = 5

num_iter = int(minutes * 2)

input_arr = np.zeros(L)

np.savetxt("input.txt", input_arr)

means = np.zeros(num_iter)
variances = np.zeros(num_iter)

for i in range(num_iter):
    run("python", "client.py", "--input_file=input.txt", "--output_file=output.txt",
        "--srv_hostname=iscsrv72.epfl.ch", "--srv_port=80")

    output_arr = np.loadtxt("output.txt")
    means[i] = np.mean(output_arr)
    variances[i] = np.var(output_arr)
    print("iteration: ", i+1)
    print("mean: ", means[i])
    print("var: ", variances[i])
    if i != (num_iter-1):
        time.sleep(30)
        print()

np.savetxt("logs/means_test.txt", means)
np.savetxt("logs/vars_test.txt", variances)