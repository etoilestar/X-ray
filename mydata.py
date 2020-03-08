import os
from PIL import Image
from flyai.dataset import Dataset
from torchvision import transforms
from path import DATA_PATH
from torchtoolbox.transform import Cutout

class FlyAIDataset(Dataset):
  def __init__(self, x_dict, y_dict, train_flag=True):
      self.images = [x['image_path'] for x in x_dict]
      self.labels = [y['labels'] for y in y_dict]
      if train_flag:
          self.transform = transforms.Compose([
                  # transforms.RandomCrop(196),
                  transforms.Resize((224, 224)),
                  transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                         scale=(0.95, 1.05), fillcolor=128),
#                  Cutout(),
                  transforms.RandomHorizontalFlip(), # 随机水平翻转
                  transforms.RandomVerticalFlip(), # 随机竖直翻转
                  transforms.RandomRotation(30), #（-30，+30）之间随机旋转
                  transforms.ToTensor(), #转成tensor[0, 255] -> [0.0,1.0]
                  transforms.RandomErasing(),
                  ])#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      else:
          self.transform = transforms.Compose([
                  transforms.Resize((224, 224)),
                  transforms.ToTensor(),
                  ])#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  def __len__(self):
      return len(self.images)

  def __getitem__(self, index):
      path = os.path.join(DATA_PATH, self.images[index])
      image = Image.open(path).convert('RGB')
      img = self.transform(image)
      label = self.labels[index]
      return img, label