import subprocess

import subprocess

with open("output.log", "w") as f:
    for i in range(30):
        subprocess.Popen(["python", "basic_cls.py"], stdout=f)