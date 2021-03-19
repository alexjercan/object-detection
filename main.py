import test
import yaml
import logging

from model.model import Model
from utils.common import check_dataset, check_file, set_logging
from utils.torch import select_device, torch_distributed_zero_first
from utils.datasets import create_dataloader

logger = logging.getLogger(__name__)

def model_create():
    cfg = check_file('yolov5s.yaml')
    device = select_device()
    model = Model(cfg).to(device)
    model.train()
    
    return model


def data_loader_create():
    with open(check_file("bdataset.yaml")) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    check_dataset(data_dict)  # check
    train_path = data_dict['train']
    return create_dataloader(train_path, imgsz=640, batch_size=1, stride=32)


if __name__ == '__main__':
    set_logging()
    model = model_create()
    dataloader, dataset = data_loader_create()
