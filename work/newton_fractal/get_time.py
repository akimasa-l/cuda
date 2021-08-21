import time
import subprocess

# compile
subprocess.run(["nvcc", "./main.cu", "-O2", "--generate-code", "arch=compute_61,code=sm_61",
               "--generate-code", "arch=compute_75,code=sm_75", "-o", "./main.out"], cwd="./cu/")
# subprocess.run(["g++", "./main.cpp", "-O2", "-o", "./main.out"], cwd="./cpp/")

""" t = time.time()
a = subprocess.run(["./main.out"],stdout=subprocess.PIPE,
                   cwd="./cpp/", universal_newlines=True)
print("cpp:", time.time()-t)
with open("./cpp/main.log",mode="w") as f:
    f.write(a.stdout) """

t = time.time()
a = subprocess.run(["nvprof", "./main.out"],stdout=subprocess.PIPE,
                   cwd="./cu/", universal_newlines=True)
print("CUDA :", time.time()-t)
with open("./cu/main.log",mode="w") as f:
    f.write(a.stdout)

# write graph
# subprocess.run(["python3", "./main.py"], cwd="./cpp/")
subprocess.run(["python3", "./main.py"], cwd="./cu/")
