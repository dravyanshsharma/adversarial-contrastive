import PIL.Image as Image
import numpy as np
import cv2

class RandomGaussianBlur(object):
    def __init__(self, kernel_size, min_val=0.1, max_val=2.0, p=0.5):
        self.kernel_size = int(kernel_size / 2) * 2 + 1
        self.min = min_val
        self.max = max_val
        self.p = p

    def __call__(self, sample):
        if np.random.random_sample() < self.p:
            sample = np.asarray(sample)
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
            sample = Image.fromarray(sample)

        return sample
