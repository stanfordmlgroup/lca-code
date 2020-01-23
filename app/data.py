from PIL import Image

import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_img(img_path):
    """
    Args
    img_path (str): path to the image to load
    """
    return Image.open(img_path).convert('RGB')


def transform_img(img, use_gpu=False):
    
    SCALE = 320
    CROP = 320

    transforms_list = [transforms.Resize((SCALE))]
    transforms_list += [transforms.CenterCrop((CROP, CROP)) if CROP else None]
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    transforms_list += [transforms.ToTensor(), normalize]

    transform = transforms.Compose([t for t in transforms_list if t])

    img_tensor = transform(img)

    img_tensor = img_tensor.unsqueeze(0)

    if use_gpu:

        return img_tensor.cuda()

    else:

       return img_tensor.cpu()


class UnNormalize(object):
    def __init__(self, mean, std, use_gpu=False):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
        if use_gpu:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        assert(len(tensor.size()) == 3), f'Image tensor should have 3 dimensions. Got tensor with {len(tensor.size())} dimenions.'

        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        return tensor


def un_normalize(img_tensor, use_gpu=False):
    """Un-normalize a PyTorch Tensor seen by the model into a NumPy array of
    pixels fit for visualization. If using raw Hounsfield Units, window the input.

    Args:
        img_tensor: Normalized tensor using mean and std. Tensor with pixel values in range (-1, 1).

    Returns:
        unnormalized_img: Numpy ndarray with values between 0 and 1.
    """

    unnormalizer = UnNormalize(IMAGENET_MEAN, IMAGENET_STD, use_gpu=use_gpu)
    # Make a copy, as we don't want to un_normalize in place. The unnormalizer affects the inputted  tensor.
    img_tensor_copy = img_tensor.clone()
    unnormalized_img = unnormalizer(img_tensor_copy)
    unnormalized_img = unnormalized_img.cpu().float().numpy()
    return unnormalized_img


def _normalize_png(img):
    """Normalizes img to be in the range 0-255."""
    img -= np.amin(img)
    img /= (np.amax(img) + 1e-7)
    img *= 255
    return img


def add_heat_map(original_image, intensities_np, alpha_img=0.33, color_map='magma', normalize=True):
    """Add a CAM heat map as an overlay on a PNG image.

    Args:
        original_image: Pixels to add the heat map on top of. Must be in range (0, 1).
        intensities_np: Intensity values for the heat map. Must be in range (0, 1).
        alpha_img: Weight for image when summing with heat map. Must be in range (0, 1).
        color_map: Color map scheme to use with PyPlot.
        normalize: If True, normalize the intensities to range exactly from 0 to 1.

    Returns:
        Original pixels with heat map overlaid.
    """
    assert(np.max(intensities_np) <= 1 and np.min(intensities_np) >= 0)
    assert(np.max(original_image) <= 1 and np.min(original_image) >= 0), f'np.max: {np.max(original_image)} and np.min: {np.min(original_image)}'
    color_map_fn = plt.get_cmap(color_map)


    if normalize:
        # Returns pixel values between 0 and 255
        intensities_np = _normalize_png(intensities_np)
    else:
        intensities_np *= 255

    # Get heat map (values between 0 and 1
    heat_map = color_map_fn(intensities_np.astype(np.uint8))
    if len(heat_map.shape) == 3:
        heat_map = heat_map[:, :, :3]
    else:
        heat_map = heat_map[:, :, :, :3]

    new_img = (alpha_img * original_image.astype(np.float32)
            + (1. - alpha_img) * heat_map.astype(np.float32))

    new_img = np.uint8(_normalize_png(new_img))

    return new_img



