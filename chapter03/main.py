import subprocess
a=subprocess.run(["nvprof","./simpleDivergence.out"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
print("stdout:",a.stdout)#->none
print("stderr:",a.stderr)