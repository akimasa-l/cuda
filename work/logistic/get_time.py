import time
import subprocess
t=time.time()
a=subprocess.run("./logistic.out",stdout=subprocess.PIPE,universal_newlines=True)
print("Logistic:",time.time()-t)
t=time.time()
a=subprocess.run("./logistic_cpp.out",stdout=subprocess.PIPE,universal_newlines=True)
print("Logistic_cpp:",time.time()-t)