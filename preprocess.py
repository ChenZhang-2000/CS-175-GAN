import time
import pickle

import yaml
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.build import *
from utils.configuration import Configuration
from utils.tools import progress_bar
from model.trainer import Trainer
from model.networks.text_encoder import Encoder
from data.dataset import build_dictionary, padding_img


def preprocess():
    with open(r"config/config.yaml") as stream:
        config = yaml.safe_load(stream)
    config = Configuration(**config)
    config.TRAIN.BS = 1
    # print(config.GLOBAL.TRAIN_DIR)

    # train_loader, valid_loader = build_dataloader(config, preprocessed=False)
    train_filenames = pickle.load(open(r"E:\Datasets\COCO\train\filenames.pickle", 'rb'))
    valid_filenames = pickle.load(open(r"E:\Datasets\COCO\test\filenames.pickle", 'rb'))
    train_tokens, valid_tokens, ixtoword, wordtoix = pickle.load(open(r"E:\Datasets\COCO\annotations\captions_DAMSM.pickle", 'rb'))
    # print(train_tokens)

    # train_loader.dataset.preprocessed = False
    # valid_loader.dataset.preprocessed = False
    # print(time.time()-t)
    encoder = Encoder()
    encoder.load_state_dict(torch.load(fr"data/weights/text_encoder.pth"))
    encoder.cuda()

    to_tensor = ToTensor()
    resize_img = Resize((256, 256))

    for mode, filenames, l_tokens in (('train', train_filenames, train_tokens), ('valid', valid_filenames, valid_tokens)):
        for i in tqdm(range(len(filenames))):
            filename = filenames[i]
            tokens = l_tokens[i*5: (i+1)*5]
            img = Image.open(rf"E:\Datasets\COCO\images\{mode}2014\{filename}.jpg")

            # texts = open(rf"E:\Datasets\COCO\text\{filename}.txt").read().encode('utf-8').decode('utf8').split('\n')

            # plt.imshow(img)
            # plt.show()

            # sentence = ""
            # for idx in token:
            #     sentence += ixtoword[idx] +' '
            # print(sentence)
            # assert len(texts) == 5
            # for text in texts:
            #     print(text)
            # print()
            xs = []
            xs_len = []
            for token in tokens:
                sent_caption = np.asarray(token).astype('int64')
                if (sent_caption == 0).sum() > 0:
                    print('ERROR: do not need END (0) token', sent_caption)
                num_words = len(sent_caption)
                # pad with 0s (i.e., '<end>')
                x = np.zeros((18, 1), dtype='int64')
                x_len = num_words
                if num_words <= 18:
                    x[:num_words, 0] = sent_caption
                else:
                    ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
                    np.random.shuffle(ix)
                    ix = ix[:18]
                    ix = np.sort(ix)
                    x[:, 0] = sent_caption[ix]
                    x_len = 18

                xs.append(x)
                xs_len.append(x_len)

            xs = np.array(xs)
            emb = torch.from_numpy(xs)
            emb = emb.reshape(5, 18)

            sorted_cap_lens, sorted_cap_indices = xs_len = torch.sort(torch.tensor(xs_len), 0, True)
            emb = emb[sorted_cap_indices].squeeze()
            emb = Variable(emb).cuda()
            sorted_cap_lens = Variable(sorted_cap_lens)

            emb = encoder(emb, sorted_cap_lens)

            img = to_tensor(img)
            img = padding_img(img)
            img = resize_img(img)
            # print(img.shape)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            data = (img.reshape(3, 256, 256).detach().cpu(), emb.reshape(5, 256).detach().cpu())

            # if i >= 2:
            #     break
            torch.save(data, fr"E:\Datasets\COCO\preprocessed\{mode}\{str(i).zfill(12)}.pt")


if __name__ == "__main__":
    with torch.no_grad():
        preprocess()
