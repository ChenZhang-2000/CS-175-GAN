from os import listdir
import json
import time
import random
import pickle
from collections import Counter, OrderedDict, defaultdict

from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import nltk
from nltk import word_tokenize
from nltk.lm import Vocabulary
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize
from torchtext.vocab import build_vocab_from_iterator, GloVe
from sklearn.feature_extraction.text import CountVectorizer

from utils.tools import progress_bar
from model.networks.text_encoder import Encoder

'''
# Load data into dict
# Assumes that imgDir and txtDir may not be the same so enter full path to each
# Test data does not need a txtDir (probably)
# Assumes that imgDir contains single split while txtDir may contain multiple splits (as from download)
# split = train, val, or test
# year = 2014 but option to choose another year's dataset just in case, just for text stuff
# resize = use true if not generating output for display
# MAKE SURE YOU HAVE THE SAME NUMBER OF IMAGES AS VALID TEXT FILES
# ASSUMES THAT DIRECTORIES ARE SORTED SO THAT ORDER OF IMG AND TXT FILES IS THE SAME
'''


def build_dictionary(train_tokens, valid_tokens):
    """
    Moldified from https://github.com/tobran/DF-GAN
    :param train_tokens:
    :param valid_tokens:
    :return:
    """
    # word_counts = defaultdict(float)
    captions = train_tokens + valid_tokens
    # vectorizer = CountVectorizer(stop_words={'english'}, min_df=0.01, ngram_range=(1, 2))
    vocab = Vocabulary(captions, unk_cutoff=2)
    print(vocab)

    # for sent in tqdm(captions):
    #     for word in sent:
    #         word_counts[word] += 1
    # vocab = [w for w in tqdm(word_counts) if word_counts[w] >= 0]

    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in tqdm(vocab):
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    train_tokens_new = []
    for t in tqdm(train_tokens):
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        train_tokens_new.append(rev)

    valid_tokens_new = []
    for t in tqdm(valid_tokens):
        rev = []
        for w in t:
            if w in wordtoix:
                rev.append(wordtoix[w])
        # rev.append(0)  # do not need '<end>' token
        valid_tokens_new.append(rev)

    return [train_tokens_new, valid_tokens_new,
            ixtoword, wordtoix, len(ixtoword)]


def padding_img(x: torch.tensor, device='cpu'):
    c, w, h = x.shape[0], x.shape[1], x.shape[2]
    d = x.shape[1] - x.shape[2]
    if d > 0:
        return torch.cat([torch.zeros(c, w, d // 2, device=device), x, torch.zeros(c, w, d - d // 2, device=device)],
                         dim=2)
    elif d < 0:
        return torch.cat([torch.zeros(c, -d // 2, h, device=device), x, torch.zeros(c, -d + d // 2, h, device=device)],
                         dim=1)
    else:
        return x


class ImgTxtDataset(Dataset):
    def __init__(self, imgDir, json_dir=None, split='Test', year='2014', resize=True, preprocessed=False, usr_study=False):
        self.imgDir = imgDir
        self.json_dir = json_dir
        self.preprocessed_dir = rf"E:/Datasets/COCO/preprocessed/{'valid' if split=='Test' else 'train'}"
        self.split = split
        self.year = year
        self.data_list = []
        self.txt_list = []
        self.pt_list = []
        self.resize = resize
        self.usr_study = usr_study

        self.preprocessed = preprocessed

        if self.preprocessed:
            self.load_preprocessed_data()
        else:
            self.to_tensor = ToTensor()
            self.resize_img = Resize((256, 256))
            self.vec = GloVe(name='840B', dim=300)

            self.encoder = Encoder()
            self.encoder.load_state_dict(torch.load(fr"data/weights/text_encoder.pth"))
            self.encoder.cuda()

            self.load_data()
        # self.preprocess_data()

    def __len__(self):
        return len(self.data_list)

    # item must be index #
    def __getitem__(self, item):
        if self.preprocessed:
            data = torch.load(self.data_list[item], map_location='cpu')
            if self.usr_study:
                with open(self.txt_list[item]) as f:
                    return data[0], data[1][0], f.read().encode('utf-8').decode('utf8').split('\n')[0]
            else:
                return data[0], data[1][random.randint(0, 4)]
        else:
            with torch.no_grad():
                img_file_name, tokens = self.data_list[item]
                # print(tokens)
                ret = self.vec.get_vecs_by_tokens(tokens, True)
                # print(ret)

                f_path = f"{self.imgDir}/{img_file_name}.jpg"
                img = Image.open(f_path)
                if not img.mode == 'RGB':
                    img = img.convert('RGB')
                img = self.to_tensor(img)
                img = padding_img(img)
                img = self.resize_img(img)

                annot = self.encoder(ret.reshape(1, *ret.shape).cuda())
                # print(annot)
                # print()

                return img, annot

        # return torch.load(rf"{self.preprocessed_dir}\{self.pt_list[item]}")

    def load_preprocessed_data(self):
        n = len(listdir(self.preprocessed_dir))
        if self.usr_study:
            filenames = pickle.load(open(rf"E:\Datasets\COCO\{'test' if self.split == 'Test' else 'train'}\filenames.pickle", 'rb'))
            txt_dir = f"{'/'.join(self.preprocessed_dir.split('/')[:-2])}/text"
        for i in range(n):
            # if random.random() > 0.2:
            #     continue

            self.data_list.append(f"{self.preprocessed_dir}/{str(i).zfill(12)}.pt")
            if self.usr_study:
                self.txt_list.append(rf"{txt_dir}/{filenames[i]}.txt")

    def load_data(self):
        json_file = json.load(open(self.json_dir))
        n = len(json_file["annotations"])

        t = time.time()
        for i, annot_info in enumerate(tqdm(json_file["annotations"])):
            # progress_bar(i, n, progress_name=f"Loading {self.split} Dataset", t=t)

            # if random.random() > 0.15:
            #     continue

            img_id = annot_info["image_id"]
            img_file_name = f"COCO_{'train' if self.split == 'Train' else 'val'}2014_{str(img_id).zfill(12)}"
            caption = annot_info["caption"]

            # f_path = f"{self.imgDir}/{img_file_name}.jpg"
            # img = Image.open(f_path)
            # if not img.mode == 'RGB':
            #     img = img.convert('RGB')
            # img = self.to_tensor(img)
            # img = padding_img(img)
            # img = self.resize_img(img)

            tokens = word_tokenize(caption, "english")
            self.data_list.append((img_file_name, tokens))
        # progress_bar(n, n, progress_name=f"Loading {self.split} Dataset", t=t)

    def process_txt(self):
        pass

    def preprocess_data(self):
        json_file = json.load(open(self.json_dir))
        n = len(json_file["annotations"])
        vec = GloVe(name='840B', dim=300)

        t = time.time()
        for i, annot_info in enumerate(json_file["annotations"]):
            progress_bar(i, n, progress_name=f"Loading {self.split} Dataset", t=t)

            img_id = annot_info["image_id"]
            img_file_name = f"COCO_train2014_{str(img_id).zfill(12)}"
            caption = annot_info["caption"]

            f_path = f"{self.imgDir}/{img_file_name}.jpg"
            img = Image.open(f_path)
            if not img.mode == 'RGB':
                img = img.convert('RGB')
            img = self.to_tensor(img)
            img = padding_img(img)
            img = self.resize_img(img)

            tokens = word_tokenize(caption, "english")
            ret = vec.get_vecs_by_tokens(tokens, True)

            data = (img, ret)
            torch.save(data, f"{self.preprocessed_dir}/{str(i).zfill(12)}.pt")


        # unk_token = '<unk>'
        # caption_vocab = build_vocab_from_iterator(self.txt_data, specials=[unk_token])
        # # caption_vocab.set_default_index(-1)
        # print(caption_vocab.__len__())

    # def loadImg(self):
    #     file_list = listdir(self.imgDir)
    #     n = len(file_list)
    #     for i, file in enumerate(file_list):
    #         fPath = f"{self.imgDir}\{file}"
    #         progress_bar(i, n, progress_name=f"Loading {self.split} Dataset Image")
    #         img = Image.open(fPath)
    #         if not img.mode == 'RGB':
    #             img = img.convert('RGB')
    #         img = self.to_tensor(img)
    #         if self.resize:
    #             img = padding_img(img)
    #             img = self.resize_img(img)
    #         self.data.append(img)
#
    # def loadTxt(self):
    #     for file in listdir(self.txtDir):
    #         if file.startswith('COCO_' + self.split + self.year): # filters out other text files so you don't have to spend time moving shit
    #             fPath = f"{self.txtDir}\{file}"
    #             # read file
    #             with open(fPath, 'r') as f:
    #                 li = []
    #                 for line in f:
    #                     li.append(line.strip())
    #                 self.txtData.append(li)


# FOR DEBUGGING/TESTING
'''
if __name__ == '__main__':
    data = ImgTxtDataset('C:/users/K/Desktop/Blah/Hiromitsu', 'C:/users/K/Desktop/testdir', split='train', resize=True)
    for i in range(data.__len__()):
        img = Image.fromarray(data.__getitem__(i)['Img'])
        print(data.__getitem__(i)['Img'].shape)
        img.show()
        img.save('C:/users/K/Desktop/output/' + str(i) + '.jpg', 'JPEG')
'''
