import random
import numpy as np
import math
import cv2 as cv
import torch
import torch.nn.functional as F


class Transform:
    """ Class for applying various image transformations."""
    def __call__(self, *args):
        rand_params = self.roll()
        if rand_params is None:
            rand_params = ()
        elif not isinstance(rand_params, tuple):
            rand_params = (rand_params,)
        output = [self.transform(img, *rand_params) for img in args]
        if len(output) == 1:
            return output[0]
        return output

    def roll(self):
        return None

    def transform(self, img, *args):
        """Must be deterministic"""
        raise NotImplementedError


class Compose:
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            if not isinstance(args, tuple):
                args = (args,)
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensorAndJitter(Transform):
    """ Convert to a Tensor and jitter brightness"""
    def __init__(self, brightness_jitter=0.0):
        self.brightness_jitter = brightness_jitter

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def transform(self, img, brightness_factor):
        # handle numpy array
        img = torch.from_numpy(img.transpose((2, 0, 1)))

        # backward compatibility
        return img.float().mul(brightness_factor/255.0).clamp(0.0,1.0)


class ToGrayscale(Transform):
    """Converts image to grayscale with probability"""
    def __init__(self, probability = 0.5):
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    def transform(self, img, do_grayscale):
        if do_grayscale:
            if isinstance(img, torch.Tensor):
                raise NotImplementedError('Implement torch variant.')
            img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            return np.stack([img_gray, img_gray, img_gray], axis=2)
            # return np.repeat(np.sum(img * self.color_weights, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
        return img


class RandomHorizontalFlip(Transform):
    """Horizontally flip the given NumPy Image randomly with a probability p."""
    def __init__(self, probability = 0.5):
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform(self, img, do_flip):
        if do_flip:
            if isinstance(img, torch.Tensor):
                return img.flip((2,))
            return np.fliplr(img).copy()
        return img


class Blur(Transform):
    """ Blur the image by applying a gaussian kernel with given sigma"""
    def __init__(self, sigma):
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        self.sigma = sigma
        self.filter_size = [math.ceil(2*s) for s in self.sigma]
        x_coord = [torch.arange(-sz, sz+1, dtype=torch.float32) for sz in self.filter_size]
        self.filter = [torch.exp(-(x**2)/(2*s**2)) for x, s in zip(x_coord, self.sigma)]
        self.filter[0] = self.filter[0].view(1,1,-1,1) / self.filter[0].sum()
        self.filter[1] = self.filter[1].view(1,1,1,-1) / self.filter[1].sum()

    def transform(self, img):
        if isinstance(img, torch.Tensor):
            sz = img.shape[2:]
            im1 = F.conv2d(img.view(-1, 1, sz[0], sz[1]), self.filter[0], padding=(self.filter_size[0], 0))
            return F.conv2d(im1, self.filter[1], padding=(0,self.filter_size[1])).view(-1,sz[0],sz[1])
        else:
            raise NotImplementedError
