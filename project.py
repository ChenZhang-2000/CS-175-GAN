import time

import yaml
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from utils.build import *
from utils.configuration import Configuration
from model.trainer import Trainer


def test(num_generation):
    with open(r"config/config.yaml") as stream:
        config = yaml.safe_load(stream)
    config = Configuration(**config)

    t = time.time()
    train_loader, valid_loader = build_dataloader(config)
    generator = build_generator(config)
    generator.load_state_dict(torch.load(rf".\runs\pretrained\weights\g_290.pt", map_location='cpu'))
    generator.cuda()

    for i, data in enumerate(train_loader):
        if i >= num_generation:
            break
        img, annot = data
        img = img[0].permute(1, 2, 0).cpu()
        plt.imshow(img)
        plt.show()
        with torch.no_grad():
            for j in range(3):
                img_f = generator(annot[:1].cuda())[0]
                img_f = img_f.cpu()
                vutils.save_image(img_f.data, f"output/{i}_{j}.jpg", range=(-1, 1), normalize=True)


if __name__ == "__main__":
    test(5)
