import torch
from torch.nn import init
import os
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from skimage.color import rgb2lab
import transforms.ISTD_transforms as transforms
from datasets.data_loader import ISTD as commonDataset

def create_exp_dir(exp):
  try:
    os.makedirs(exp)
    print('Creating exp dir: %s' % exp)
  except OSError:
    pass
  return True

def getLoader(dataroot, originalSize, imageSize, batchSize=64, workers=4,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True, seed=None):

  #import pdb; pdb.set_trace()
  if split == 'train':
    dataset = commonDataset(dataroot=dataroot,
                            transform=transforms.Compose([
                              transforms.Scale(originalSize),
                              transforms.RandomCrop(imageSize),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                            ]),
                            split=split,
                            seed=seed)
  else:
    dataset = commonDataset(dataroot=dataroot,
                            transform=transforms.Compose([
                              transforms.Scale(originalSize),
                              transforms.CenterCrop(imageSize),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std),
                             ]),
                            split=split,
                            seed=seed)

  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batchSize,
                                           shuffle=shuffle,
                                           num_workers=int(workers))
  return dataloader

# def weights_init(m):
#   classname = m.__class__.__name__
#   if classname.find('Conv') != -1:
#     m.weight.data.normal_(0.0, 0.2)
#   elif classname.find('BatchNorm') != -1:
#     m.weight.data.normal_(0.0, 0.2)
#     m.bias.data.fill_(0)

def init_weights(net, init_type='normal', init_gain=0.2):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 0.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.2, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        # net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


import numpy as np
class ImagePool:
  def __init__(self, pool_size=50):
    self.pool_size = pool_size
    if pool_size > 0:
      self.num_imgs = 0
      self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image
    if self.num_imgs < self.pool_size:
      self.images.append(image.clone())
      self.num_imgs += 1
      return image
    else:
      if np.random.uniform(0,1) > 0.5:
        random_id = np.random.randint(self.pool_size, size=1)[0]
        tmp = self.images[random_id].clone()
        self.images[random_id] = image.clone()
        return tmp
      else:
        return image


def adjust_learning_rate(optimizer, init_lr, epoch, factor, every):
  #import pdb; pdb.set_trace()
  lrd = init_lr / every
  old_lr = optimizer.param_groups[0]['lr']
   # linearly decaying lr
  lr = old_lr - lrd
  if lr < 0: lr = 0
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def print_current_losses(log_dir, epoch, lr, iters, losses, t_comp, t_data):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '{epoch: %d, iters: %d, lr: %.6f, time: %.3f, data_load_time: %.3f} ' % (epoch, iters, lr, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message
    with open(log_dir, "a+") as log_file:
        log_file.write('%s\n' % message)  # save the message

def calc_BER(fake_B, real_B):
    fake_numpy = tensor2im(fake_B).reshape([-1])
    real_numpy = tensor2im(real_B).reshape([-1])
    BAC = balanced_accuracy_score(real_numpy, fake_numpy)
    BER = 1 - BAC
    return BER

def calc_RMSE(fake_C, real_C):
    real_numpy = tensor2im(real_C)
    # convert to LAB color space
    real_lab = rgb2lab(real_numpy)
    fake_numpy = tensor2im(fake_C)
    fake_lab = rgb2lab(fake_numpy)
    RMSE = 0
    for i in range(real_lab.shape[2]):
        RMSE = RMSE + mean_squared_error(real_lab[:, :, i], fake_lab[:, :, i])
    return RMSE