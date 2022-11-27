train = ImgTxtDataset(config.GLOBAL.TRAIN_DIR,
                      config.GLOBAL.TRAIN_TXT,
                      split="Train")
train.preprocess_data()
# joblib.dump(train, r"E:\Datasets\COCO\preprocessed\train.pt")

valid = ImgTxtDataset(config.GLOBAL.VALID_DIR,
                      config.GLOBAL.VALID_TXT,
                      split="Test")
valid.preprocess_data()