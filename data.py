import os
import numpy as np
from PIL import Image
from torch.utils.data import dataset
train_path = r"C:\Users\jj597\Desktop\vessel\DRIVE\training\images"
train_file = os.listdir(train_path)
#img = Image.open(path)

for i in train_file:
    im = cv2.imread(train_path + i)



class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        data_root = os.path.join(root, "aug" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                print(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])

        mask = Image.open(self.manual[idx]).convert('L')
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)


