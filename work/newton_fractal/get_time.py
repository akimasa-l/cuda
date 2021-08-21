import time
import subprocess
t=time.time()
a=subprocess.run("./cpp/main.out",stdout=subprocess.PIPE,universal_newlines=True)
print("cpp :",time.time()-t)
t=time.time()
a=subprocess.run("./cu/main.out",stdout=subprocess.PIPE,universal_newlines=True)
print("CUDA :",time.time()-t)