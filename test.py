import time

import yaml
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from utils.build import *
from utils.configuration import Configuration
from model.trainer import Trainer


def test():
    with open(r"config/config.yaml") as stream:
        config = yaml.safe_load(stream)
    config = Configuration(**config)
    # print(config.GLOBAL.TRAIN_DIR)

    t = time.time()
    train_loader, valid_loader = build_dataloader(config)
    generator = build_generator(config)
    generator.load_state_dict(torch.load(rf".\runs\pretrained\weights\g_290.pt", map_location='cpu'))
    generator.load_state_dict(torch.load(rf".\runs\202211290202\weights\g_99.pt", map_location='cpu'))
    generator.load_state_dict(torch.load(rf".\runs\202212011759\weights\g_9.pt", map_location='cpu'))
    generator.load_state_dict(torch.load(rf".\runs\202212020127\weights\g_33.pt", map_location='cpu'))
    generator.load_state_dict(torch.load(rf".\runs\202212030302\weights\g_51.pt", map_location='cpu'))
    generator.cuda()
    discriminator = build_discriminator(config)
    # print(time.time()-t)

    for i, data in enumerate(train_loader):
        if i >= 3:
            break
        img, annot = data
        img = img[0].permute(1, 2, 0).cpu()
        plt.imshow(img)
        plt.show()
        with torch.no_grad():
            for j in range(3):
                img_f = generator(annot[:1].cuda())[0]
                img_f = img_f.cpu()  # .permute(1, 2, 0)
                vutils.save_image(img_f.data, f"output/{i}_{j}.jpg", range=(-1, 1), normalize=True)
                # plt.imshow(img_f.permute(1, 2, 0))
                # plt.show()
        # print(annot)
        # print(img_f)
        # plt.imshow(img_f)
        # print(annot.min(), annot.max())
        # print(img_f.min(), img_f.max())
        # plt.show()


if __name__ == "__main__":
    test()
