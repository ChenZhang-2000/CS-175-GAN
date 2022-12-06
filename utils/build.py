
import joblib
from torch.utils.data import DataLoader

from data.dataset import ImgTxtDataset
from model.networks.generator import Generator
from model.networks.discriminator import Discriminator


def build_dataloader(config, preprocessed=True):
    train = ImgTxtDataset(config.GLOBAL.TRAIN_DIR,
                          config.GLOBAL.TRAIN_TXT,
                          split="Train",
                          preprocessed=preprocessed)

    valid = ImgTxtDataset(config.GLOBAL.VALID_DIR,
                          config.GLOBAL.VALID_TXT,
                          split="Test",
                          preprocessed=preprocessed)

    print(f"Training Size: {len(train)}")
    print(f"Validate Size: {len(valid)}")

    train_loader = DataLoader(train, batch_size=config.TRAIN.BS, drop_last=True,
                              num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=config.TRAIN.BS, drop_last=True,
                              num_workers=4, shuffle=True)

    return train_loader, valid_loader


def build_encoder(config):
    pass


def build_generator(config):
    generator = Generator()
    generator.cuda()

    return generator


def build_discriminator(config):
    discriminator = Discriminator()
    discriminator.cuda()

    return discriminator


