import time
from pathlib import Path

import yaml
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from utils.build import *
from utils.configuration import Configuration
from model.trainer import Trainer


def generate():
    with open(r"config/config.yaml") as stream:
        config = yaml.safe_load(stream)
    config = Configuration(**config)

    t = time.time()

    dataset = ImgTxtDataset(config.GLOBAL.VALID_DIR,
                            config.GLOBAL.VALID_TXT,
                            split="Test",
                            preprocessed=True,
                            usr_study=True)

    data_loader = DataLoader(dataset, batch_size=1, drop_last=True,
                             num_workers=0, shuffle=False)

    generator = build_generator(config)
    generator.load_state_dict(torch.load(rf".\runs\pretrained\weights\g_290.pt", map_location='cpu'))
    generator.cuda()
    discriminator = build_discriminator(config)

    texts = []

    for i, data in enumerate(data_loader):
        if i < 20:
            continue

        if i >= 30:
            break

        img, annot, txt = data

        texts.append(txt[0])
        Path(f"usr_study/{i}").mkdir(parents=True, exist_ok=True)
        vutils.save_image(img.data, f"usr_study/{i}/real.jpg", range=(0, 1), normalize=True)

        with torch.no_grad():
            for j in range(20):
                img_f = generator(annot.cuda())[0]
                img_f = img_f.cpu()  # .permute(1, 2, 0)
                vutils.save_image(img_f.data, f"usr_study/{i}/fake_{j}.jpg", range=(-1, 1), normalize=True)

    texts = "\n".join(texts)
    with open("usr_study/captions.txt", 'w') as f:
        f.write(texts)


if __name__ == "__main__":
    generate()
