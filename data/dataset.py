from torch.utils.data import Dataset


class ImgTxtDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.data_list = []
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]
