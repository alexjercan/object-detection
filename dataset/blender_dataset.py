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
    def __init__(self, root_dir, cvs_fname, render_transform=None, depth_transform=None, train=False):
        self.rgb_images = []
        self.depth_images = []
        self.ids = {}
        self.labels = []
        self.root_dir = root_dir
        self.render_transform = render_transform
        self.depth_transform = depth_transform
        self.num_classes = 0
        self.train = train

        json_fnames = []
        with open(join(root_dir, cvs_fname), 'r') as fd:
            json_fnames = fd.read().splitlines()

        idx = 0
        for json_fname in json_fnames:
            with open(join(root_dir, json_fname), 'r') as json_file:
                fname = json_fname.split('.')[0]
                data = json.load(json_file)
                self.rgb_images.append(fname + '_render.png')
                self.depth_images.append(fname + '_depth.exr')
                if data['label'] not in self.ids:
                    self.ids[data['label']] = idx
                    idx = idx + 1
                self.labels.append(self.ids[data['label']])
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        render_name = self.rgb_images[idx]
        render_path = join(self.root_dir, render_name)
        render_data = np.array(Image.open(render_path).convert('RGB'))

        depth_name = self.depth_images[idx]
        depth_path = join(self.root_dir, depth_name)
        depth_data = exr2depth(depth_path)

        if self.render_transform:
            data = self.render_transform(render_data)

        if self.depth_transform:
            depth_data = self.depth_transform(depth_data)

        return data, depth_data, self.labels[idx]


def exr2depth(exr):
    file = OpenEXR.InputFile(exr)

    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist()
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


def test_dataset():
    root = 'C:/dev/blenderRenderer/output'
    from torchvision import transforms
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    blender_data = BlenderDataset(root, "train.csv", transform_val, transform_val, True)
    print(blender_data.num_classes)
    dataloader = DataLoader(blender_data, batch_size=2, shuffle=True)
    for data in dataloader:
        images, depth_images, labels = data
        print(images.size(), depth_images.size(), labels)
        break

    import matplotlib.pyplot as plt
    import torchvision
    fig=plt.figure(figsize=(2, 1))
    npimg = torchvision.utils.make_grid(images).numpy()
    fig.add_subplot(2, 1, 1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    npdepth = torchvision.utils.make_grid(depth_images).numpy()
    fig.add_subplot(2, 1, 2)
    plt.imshow(np.transpose(npdepth, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    test_dataset()
