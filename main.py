import time

import yaml

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

    trainer = Trainer(data_loader=(valid_loader, None),
                      generator=generator,
                      discriminator=discriminator,
                      max_epoch=config.TRAIN.MAX_EPOCH,
                      bs=config.TRAIN.BS)
    trainer.train()


if __name__ == "__main__":
    main()
