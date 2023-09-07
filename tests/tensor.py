import matplotlib.pyplot as plt
from deeprai.tools.noise import GaussianNoise
from deeprai.tools.toolkit import normalize
from Datasets.MNIST import mnist as db

speckle_noise = GaussianNoise(std=0.06)

arr = db.load_x(1)
noisy_arr = normalize(speckle_noise.noise(arr))


fig, (ax1, ax2) = plt.subplots(1, 2)


ax1.imshow(arr.reshape(28, 28), cmap='gray_r')
ax1.set_title('Original')
ax1.axis('off')


ax2.imshow(noisy_arr.reshape(28, 28), cmap='gray_r')
ax2.set_title('Noisy')
ax2.axis('off')

plt.show()
