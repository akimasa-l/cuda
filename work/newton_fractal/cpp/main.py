import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

result = pd.read_csv("./main.log").values

result = (result + 2*np.pi) % (2*np.pi) * 180/np.pi

print(result[0].dtype)

# 初期値z0の計算範囲
real_min = -1  # 実部最小値
real_max = 1  # 実部最大値
im_min = -1  # 虚部最小値
im_max = 1  # 虚部最大値


im = plt.imshow(result, origin="lower", cmap="hsv", extent=[
                real_min, real_max, im_min, im_max])
plt.xlabel("Real part")
plt.ylabel("Imaginary part")

cbar = plt.colorbar(im)
cbar.set_label('Argument (°)', size=10)

plt.savefig("./fig_cpp.png",dpi=800)
