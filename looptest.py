import subprocess

import subprocess

with open("output.log", "w") as f:
    for i in range(8):
        subprocess.Popen(["python", "train.py"], stdout=f)