import time

import yaml
import torch

from utils.build import *
from utils.configuration import Configuration
from model.trainer import Trainer


def main():
    with open(r"config/config.yaml") as stream:
        config = yaml.safe_load(stream)
    config = Configuration(**config)
    # print(config.GLOBAL.TRAIN_DIR)

    t = time.time()
    train_loader, valid_loader = build_dataloader(config)
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    # print(time.time()-t)

    trainer = Trainer(data_loader=(train_loader, valid_loader),
                      generator=generator,
                      discriminator=discriminator,
                      max_epoch=config.TRAIN.MAX_EPOCH,
                      bs=config.TRAIN.BS)
    trainer.g.load_state_dict(torch.load(rf".\runs\202212011759\weights\g_9.pt", map_location='cuda:0'))
    trainer.d.load_state_dict(torch.load(rf".\runs\202212011759\weights\d_9.pt", map_location='cuda:0'))
    trainer.g.load_state_dict(torch.load(rf".\runs\202212020127\weights\g_33.pt", map_location='cuda:0'))
    trainer.d.load_state_dict(torch.load(rf".\runs\202212020127\weights\d_33.pt", map_location='cuda:0'))
    trainer.train()


if __name__ == "__main__":
    main()
