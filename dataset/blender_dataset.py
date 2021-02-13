from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from os import listdir
import json
from os.path import isfile, join
import OpenEXR
import Imath
import array
import numpy as np

class BlenderDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=False):
        self.rgb_images = []
        self.depth_images = []
        self.labels = []
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = 0
        self.train = train

        json_fnames = [f for f in listdir(root_dir) if (isfile(join(root_dir, f)) and f.endswith('.json'))]
        for json_fname in json_fnames:
            with open(join(root_dir, json_fname), 'r') as json_file:
                fname = json_fname.split('.')[0]
                data = json.load(json_file)
                self.rgb_images.append(fname + '_render.png')
                self.depth_images.append(fname + '_depth.exr')
                self.labels.append(data['label'])
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

        if self.transform:
            data = self.transform(render_data)
            depth_data = self.transform(depth_data)

        return data, depth_data, self.labels[idx]


def exr2depth(exr, maxvalue=1.,normalize=True):                                                               
    file = OpenEXR.InputFile(exr)

    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in "R" ]

    img = np.zeros((sz[1], sz[0], 3), np.float64)

    data = np.array(R)
    data[data > maxvalue] = maxvalue

    if normalize:
        data /= np.max(data)

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
    blender_data = BlenderDataset(root, transform_val, True)
    print(blender_data.num_classes)
    dataloader = DataLoader(blender_data, batch_size=2, shuffle=True)
    for data in dataloader:
        images, depth_images, labels = data
        print(images.size(), depth_images.size(), labels)

    transforms.ToPILImage()(images[0]).show()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(depth_images.numpy()[0][0])
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    test_dataset()
