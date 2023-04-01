import matplotlib.pyplot as plt
import numpy as np

err_data = open("gtotal/err_gtotal", "r")
nums = list(map(lambda s: float(s), err_data.readline().split()))
# print(nums)
y = np.array(nums)
size = len(y)
step = 100
x = np.arange(0, size * step, step)

fig = plt.figure()
plt.plot(x, y)
plt.title('Train results')
plt.ylabel('Errs')
plt.xlabel('Epochs')

plt.show()
