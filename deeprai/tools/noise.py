import numpy as np
import threading


class GaussianNoise:
    """
    GaussianNoise class applies Gaussian noise to numpy arrays.

    Attributes:
    - mean: Mean of the Gaussian distribution.
    - std: Standard deviation of the Gaussian distribution.

    Usage:
    ```
    gaussian_noise = GaussianNoise(mean=0, std=25)
    noisy_images = gaussian_noise.noise(list_of_images)
    ```
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def compute(self):
        def add_gaussian_noise(image):
            row, col, ch = image.shape
            gauss = np.random.normal(self.mean, self.std, (row, col, ch))
            noisy = image + gauss
            return np.clip(noisy, 0, 255)

        return add_gaussian_noise

    def noise(self, arrays):
        func = self.compute()
        threads = []
        results = [None] * len(arrays)

        def worker(index, arr):
            results[index] = func(arr)

        for i, arr in enumerate(arrays):
            t = threading.Thread(target=worker, args=(i, arr))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results


class SaltPepperNoise:
    """
    SaltPepperNoise class introduces salt and pepper noise to numpy arrays.

    Attributes:
    - s_vs_p: Proportion of salt vs. pepper noise.
    - amount: Overall amount of noise to introduce.

    Usage:
    ```
    sp_noise = SaltPepperNoise(s_vs_p=0.5, amount=0.04)
    noisy_images = sp_noise.noise(list_of_images)
    ```
    """

    def __init__(self, s_vs_p=0.5, amount=0.04):
        self.s_vs_p = s_vs_p
        self.amount = amount

    def compute(self):
        def add_sp_noise(image):
            out = np.copy(image)
            num_salt = np.ceil(self.amount * image.size * self.s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            out[coords[0], coords[1], :] = 1

            num_pepper = np.ceil(self.amount * image.size * (1. - self.s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[coords[0], coords[1], :] = 0
            return out

        return add_sp_noise

    def noise(self, arrays):
        func = self.compute()
        threads = []
        results = [None] * len(arrays)

        def worker(index, arr):
            results[index] = func(arr)

        for i, arr in enumerate(arrays):
            t = threading.Thread(target=worker, args=(i, arr))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results


class SpeckleNoise:
    """
    SpeckleNoise class introduces speckle noise to numpy arrays.

    Usage:
    ```
    speckle_noise = SpeckleNoise()
    noisy_images = speckle_noise.noise(list_of_images)
    ```
    """

    def compute(self):
        def add_speckle_noise(image):
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            noisy = image + image * gauss
            return np.clip(noisy, 0, 255)

        return add_speckle_noise

    def noise(self, arrays):
        func = self.compute()
        threads = []
        results = [None] * len(arrays)

        def worker(index, arr):
            results[index] = func(arr)

        for i, arr in enumerate(arrays):
            t = threading.Thread(target=worker, args=(i, arr))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results
