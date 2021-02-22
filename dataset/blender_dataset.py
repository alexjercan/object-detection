from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json
from os.path import join
import OpenEXR
import Imath
import array
import numpy as np


class BlenderDataset(Dataset):
    def __init__(self, root_dir, csv_fname, class_fname='class.csv', render_transform=None, depth_transform=None, albedo_transform=None, train=False):
        self.render_transform = render_transform
        self.depth_transform = depth_transform
        self.albedo_transform = albedo_transform
        self.root_dir = root_dir
        self.train = train

        self.classes = []
        with open(join(root_dir, class_fname), 'r') as fd:
            self.classes = fd.read().splitlines()
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
        albedo_data = exr2segmentation(albedo_path, label)

        if self.render_transform is not None:
            rgb_data = self.render_transform(rgb_data)

        if self.depth_transform is not None:
            depth_data = self.depth_transform(depth_data)
            
        if self.albedo_transform is not None:
            albedo_data = self.albedo_transform(albedo_data)

        return rgb_data, depth_data, label, bbox, albedo_data


def exr2depth(exr):
    file = OpenEXR.InputFile(exr)

    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    R = [array.array('f', file.channel(Chan, FLOAT)).tolist()
           for Chan in "R"]

    img = np.zeros((sz[1], sz[0], 3), np.float64)

    data = np.array(R)
    minvalue = np.min(data)
    data -= minvalue
    maxvalue = np.max(np.where(np.isinf(data), -np.Inf, data))
    data[data > maxvalue] = maxvalue
    data /= maxvalue
    

    img = np.array(data).astype(np.float32).reshape(img.shape[0], -1)

    return img


def exr2segmentation(exr, label):
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    R, G, B = [array.array('f', file.channel(Chan, FLOAT)).tolist()
                 for Chan in ("R", "G", "B")]

    seg = np.array(R) + np.array(G) + np.array(B)
    seg[seg <= 0] = -1
    seg[seg > 0] = label
    seg = seg.astype(np.float32).reshape(sz[1], -1)

    return seg


def test_dataset():
    root = 'C:/dev/blenderRenderer/output'
    from torchvision import transforms
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    blender_data = BlenderDataset(
        root, "train.csv", "class.csv", render_transform=transform_val, depth_transform=transform_val, albedo_transform=transform_val, train=True)
    print(blender_data.num_classes)
    dataloader = DataLoader(blender_data, batch_size=2, shuffle=True)
    for data in dataloader:
        images, depth_images, labels, bboxes, segmentations = data
        print(images.shape, depth_images.shape, labels.shape, bboxes.shape, segmentations.shape)
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
    
    npseg = torchvision.utils.make_grid(segmentations).numpy()
    fig.add_subplot(3, 1, 3)
    plt.imshow(np.transpose(npseg, (1, 2, 0)))
    
    plt.show()


if __name__ == '__main__':
    test_dataset()
