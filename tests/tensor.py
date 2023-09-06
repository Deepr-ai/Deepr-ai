import matplotlib.pyplot as plt
from deeprai.tools.noise import SaltPepperNoise
from deeprai.tools.toolkit import normalize
from Datasets.MNIST import mnist as db

speckle_noise = SaltPepperNoise(s_vs_p=0.5, amount=0.02)

arr = db.load_x(1)
noisy_arr = normalize(speckle_noise.noise(arr))

# Visualizing the original and noisy images
fig, (ax1, ax2) = plt.subplots(1, 2)

# Display original
ax1.imshow(arr.reshape(28, 28), cmap='gray_r') # using gray_r colormap so that 0 is white and 1 is black
ax1.set_title('Original')
ax1.axis('off')

# Display noisy image
ax2.imshow(noisy_arr.reshape(28, 28), cmap='gray_r')
ax2.set_title('Noisy')
ax2.axis('off')

plt.show()
