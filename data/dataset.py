from util.general import L_RGB, load_data, load_img_paths, load_label_paths
from util.plots import plot_images
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


def create_dataloader(img_dir, label_dir, image_size, batch_size, S, anchors, transform, used_layers=[L_RGB]):
    dataset = BDataset(img_dir, label_dir, image_size=image_size, S=S,
                       anchors=anchors, transform=transform, used_layers=used_layers)
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=BDataset.collate_fn)
    return dataset, loader


class BDataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
        used_layers=[L_RGB],
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.used_layers = used_layers

        self.layer_files = load_img_paths(self.img_dir, self.used_layers)
        self.label_files = load_label_paths(self.label_dir)

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        label_path = self.label_files[index]
        labels = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
        layers, hw0, hw = load_data(self, index)
        # labels Nx[C,X,Y,W,H]

        # Convert
        if L_RGB in layers:
            layers[L_RGB] = layers[L_RGB][:, :, ::-1]  # to RGB
        layers = {k: layers[k].transpose(2, 0, 1) for k in layers}
        img0 = layers[self.used_layers[0]]
        layers = [layers[k] for k in self.used_layers]
        layers = np.concatenate(layers, axis=0)

        img0 = np.ascontiguousarray(img0)
        layers = np.ascontiguousarray(layers)

        labels_out = torch.zeros((len(labels), 6))
        labels_out[:, 1:] = torch.from_numpy(labels)

        return torch.from_numpy(img0), torch.from_numpy(layers), labels_out

    @staticmethod
    def collate_fn(batch):
        img, layer, label = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.stack(layer, 0), torch.cat(label, 0)
