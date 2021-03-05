from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import json
import os
from os.path import join
import numpy as np
import cv2


class BlenderDataset(Dataset):
    def __init__(self, root_dir, csv_fname, class_fname='class.csv', render_transform=None, depth_transform=None, albedo_transform=None, train=False):
        self.render_transform = render_transform
        self.depth_transform = depth_transform
        self.albedo_transform = albedo_transform
        self.root_dir = root_dir
        self.train = train

        self.classes = []
        with open(join(root_dir, class_fname), 'r') as fd:
            self.classes = ["background"] + fd.read().splitlines()
        self.json_fnames = []
        with open(join(root_dir, csv_fname), 'r') as fd:
            self.json_fnames = fd.read().splitlines()

        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.json_fnames)

    def __getitem__(self, idx):
        json_fname = self.json_fnames[idx]
        with open(join(self.root_dir, json_fname), 'r') as json_file:
            data = json.load(json_file)

        if not data:
            return None

        fname = json_fname.split('.')[0]
        rgb_fname = fname + '_render.png'
        depth_fname = fname + '_depth.exr'
        albedo_fname = fname + '_albedo.exr'
        label = self.classes.index(data['label'])
        bbox = np.array(data['bbox'])

        rgb_path = join(self.root_dir, rgb_fname)
        rgb_data = np.array(Image.open(rgb_path).convert('RGB'))

        depth_path = join(self.root_dir, depth_fname)
        depth_data = exr2depth(depth_path)

        albedo_path = join(self.root_dir, albedo_fname)
        albedo_data = exr2segmap(albedo_path) * label

        if self.render_transform is not None:
            rgb_data = self.render_transform(rgb_data)

        if self.depth_transform is not None:
            depth_data = self.depth_transform(depth_data)

        if self.albedo_transform is not None:
            albedo_data = self.albedo_transform(albedo_data)

        return rgb_data, depth_data, label, bbox, albedo_data


def exr2depth(path):
    """Read depth image as numpy array

    Args:
        path (str): The path to the file

    Returns:
        ndarray: Returns an array with the shape WxHx1
    """
    if not os.path.isfile(path):
            return None
        
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    
    # get the maximum value from the array, aka the most distant point
    # everything above that value is infinite, thus i clamp it to maxvalue
    # then divide by maxvalue to obtain a normalized map
    # multiply by 255 to obtain a colormap from the depthmap
    maxvalue = np.max(np.where(np.isinf(img), -np.Inf, img))
    img[img > maxvalue] = maxvalue
    img = img / maxvalue * 255

    img = np.array(img).astype(np.uint8).reshape(img.shape[0], img.shape[1], -1)

    return img


def exr2segmap(path):
    """Read segmentation map image as numpy array

    Args:
        path (str): The path to the file

    Returns:
        ndarray: Returns an array with the shape WxHx1
    """
    if not os.path.isfile(path):
            return None    

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    img = img[..., 0] + img[..., 1] + img[..., 2]
    img[img <= 0] = 0
    img[img > 0] = 1
    img = np.array(img).astype(np.int64).reshape(img.shape[0], img.shape[1], -1)

    return img


def segmentation2rgb(segmentations, nc=54):
    import random

    def id_to_random_color(number):
        if number == 0:
            return (0, 0, 0)
        random.seed(number)
        r = random.random()
        g = random.random()
        b = random.random()
        return r, g, b

    rgbs = []

    for segmentation in segmentations.squeeze():
        r = np.zeros_like(segmentation).astype(np.float32)
        g = np.zeros_like(segmentation).astype(np.float32)
        b = np.zeros_like(segmentation).astype(np.float32)

        for l in range(1, nc + 1):
            idx = segmentation == l
            r[idx], g[idx], b[idx] = id_to_random_color(l)

        rgb = torch.tensor(np.stack([r, g, b], axis=0))
        rgbs.append(rgb)

    return torch.stack(rgbs, 0)


def test_dataset():
    root = 'C:/dev/blenderRenderer/output'
    from torchvision import transforms
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    blender_data = BlenderDataset(
        root, "train.csv", "class.csv", render_transform=transform_val, depth_transform=transform_val, albedo_transform=transform_val, train=True)
    dataloader = DataLoader(blender_data, batch_size=2, shuffle=True)
    for data in dataloader:
        images, depth_images, labels, bboxes, segmentations = data
        print(images.shape, depth_images.shape, labels.shape,
              bboxes.shape, segmentations.shape)
        break

    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import torchvision
    fig: Figure = plt.figure()

    npimg = torchvision.utils.make_grid(images).numpy()
    fig.add_subplot(3, 1, 1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    npdepth = torchvision.utils.make_grid(depth_images).numpy()
    fig.add_subplot(3, 1, 2)
    plt.imshow(np.transpose(npdepth, (1, 2, 0)))

    npseg = torchvision.utils.make_grid(
        segmentation2rgb(segmentations)).numpy()
    fig.add_subplot(3, 1, 3)
    plt.imshow(np.transpose(npseg, (1, 2, 0)))

    plt.show()


if __name__ == '__main__':
    test_dataset()
