import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import transforms.ISTD_transforms as transforms

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(train_A_dir, train_B_dir, train_C_dir):
  path_ABC = []
  if not os.path.isdir(train_A_dir):
    raise Exception('Data directory does not exist')
  for root, _, fnames in sorted(os.walk(train_A_dir)):
    for fname in fnames:
      if is_image_file(fname):
        path_A = os.path.join(train_A_dir, fname)
        path_B = os.path.join(train_B_dir, fname)
        path_C = os.path.join(train_C_dir, fname)
        if not os.path.isfile(path_B):
          raise Exception('%s does not exist' % path_B)
        if not os.path.isfile(path_C):
          raise Exception('%s does not exist' % path_C)
        item = {'path_A': path_A, 'path_B': path_B, 'path_C': path_C}
        path_ABC.append(item)
  return path_ABC

class ISTD(data.Dataset):
  def __init__(self, dataroot, transform=None, split='train', seed=None):
    if split == 'train':
      name_A = 'train_A'
      name_B = 'train_B'
      name_C = 'train_C'
    else:
      name_A = 'test_A'
      name_B = 'test_B'
      name_C = 'test_C'
    self.A_dir = os.path.join(dataroot, name_A)
    self.B_dir = os.path.join(dataroot, name_B)
    self.C_dir = os.path.join(dataroot, name_C)
    self.path_ABC = make_dataset(self.A_dir, self.B_dir, self.C_dir)
    if len(self.path_ABC) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + dataroot + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.transform = transform

    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, _):
    index = np.random.randint(self.__len__(), size=1)[0]
    path_A = self.path_ABC[index]['path_A']
    path_B = self.path_ABC[index]['path_B']
    path_C = self.path_ABC[index]['path_C']
    img_A = Image.open(path_A).convert('RGB')
    # img_B = Image.open(path_B).convert('L')
    img_B = Image.open(path_B)
    img_C = Image.open(path_C).convert('RGB')
    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      imgs = {'imgA':img_A, 'imgB':img_B, 'imgC':img_C}
      # img_A, img_B, img_C = self.transform(img_A, img_B, img_C)
      imgs = self.transform(imgs)
    # return img_A, img_B, img_C
    return imgs

  def __len__(self):
    return len(self.path_ABC)

# def main():
#   train_A_dir = '../ISTD_Dataset/train/train_A'
#   train_B_dir = '../ISTD_Dataset/train/train_B'
#   train_C_dir = '../ISTD_Dataset/train/train_C'
#   dataroot = '../ISTD_Dataset/train'
#   img = Image.open('/Users/jiarui/git/st-cgan/ISTD_Dataset/train/train_A/1-1.png').convert('RGB')
#   w, h = img.size
#   originalSize = min(w, h)
#   imageSize = 256
#   mean = (0.5, 0.5, 0.5)
#   std = (0.5, 0.5, 0.5)
#   seed = 101
#   dataloader = ISTD(dataroot=dataroot,
#                           transform=transforms.Compose([
#                             transforms.Scale(originalSize),
#                             transforms.RandomCrop(imageSize),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.ToTensor(),
#                             transforms.Normalize(mean, std),
#                           ]),
#                           seed=seed)
#   for i, data in enumerate(dataloader, 0):
#     imgA_cpu, imgB_cpu, imgC_cpu = data
#     batch_size = imgA_cpu.size(0)
#     a = batch_size
#
# if __name__ == '__main__':
#   main()