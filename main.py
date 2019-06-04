from __future__ import print_function
import time
import argparse
import random
import numpy as np
# from PIL import Image
# from torchsummary import summary
# import torch
# import torch.nn as nn
from ST_CGAN_model import *
from misc import *
import transforms.ISTD_transforms as transforms
from datasets.data_loader import ISTD as commonDataset
import torchvision.utils as vutils
from torch.autograd import Variable
# from collections import OrderedDict
# from models.STCGAN_model import STCGANModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=False,
                        default='ISTD_Dataset/train', help='path to train dataset')
    parser.add_argument('--testroot', required=False,
                        default='ISTD_Dataset/test', help='path to test dataset')
    parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
    parser.add_argument('--valBatchSize', type=int, default=64, help='val. input batch size')
    parser.add_argument('--originalSize', type=int,
                        default=286, help='the height / width of the original input image')
    parser.add_argument('--imageSize', type=int,
                        default=256, help='the height / width of the cropped input image to network')
    # parser.add_argument('--inputChannelSize', type=int,
    #                     default=3, help='size of the input channels')
    # parser.add_argument('--outputChannelSize', type=int,
    #                     default=3, help='size of the output channels')
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--lambda1', type=float, default=5, help='lambda1 for G2')
    parser.add_argument('--lambda2', type=float, default=0.1, help='lambda2 for D1')
    parser.add_argument('--lambda3', type=float, default=0.1, help='lambda3 for D2')
    parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
    parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
    parser.add_argument('--poolSize', type=int, default=50,
                        help='Buffer size for storing previously generated samples from G')
    parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate for adam')
    parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--netG1', default='', help="path to netG1 (to continue training)")
    parser.add_argument('--netD1', default='', help="path to netD1 (to continue training)")
    parser.add_argument('--netG2', default='', help="path to netG2 (to continue training)")
    parser.add_argument('--netD2', default='', help="path to netD2 (to continue training)")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--save_epoch_freq', type=int, default=20,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
    parser.add_argument('--evalIter', type=int, default=400,
                        help='interval for evauating(generating) images from testDataroot')
    parser.add_argument('--output_dir', default='./output', help='folder to output images and model checkpoints')
    opt = parser.parse_args()
    opt.isTrain = True
    print(opt)

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt

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

def create_exp_dir(exp):
  try:
    os.makedirs(exp)
    print('Creating exp dir: %s' % exp)
  except OSError:
    pass
  return True

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def main(opt):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:{}'.format(opt.gpu_ids[1])) if opt.gpu_ids else torch.device('cpu')
    create_exp_dir(opt.output_dir)
    create_exp_dir(os.path.join(opt.output_dir, 'val_results'))
    create_exp_dir(os.path.join(opt.output_dir, 'train_results'))
    opt.manualSeed = 101
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    print("Random Seed: ", opt.manualSeed)

    dataloader = getLoader(opt.dataroot,
                           opt.originalSize,
                           opt.imageSize,
                           opt.batchSize,
                           opt.workers,
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='train',
                           shuffle=True,
                           seed=opt.manualSeed)
    valDataloader = getLoader(opt.testroot,
                              opt.imageSize,  # opt.originalSize,
                              opt.imageSize,
                              opt.valBatchSize,
                              opt.workers,
                              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                              split='test',
                              shuffle=False,
                              seed=opt.manualSeed)
    G1 = generator(3, 1)
    G2 = generator(4, 3)
    D1 = discriminator(4, 1)
    D2 = discriminator(7, 1)

    G1 = init_net(G1, init_type='normal', init_gain=0.2, gpu_ids=opt.gpu_ids)
    G2 = init_net(G2, init_type='normal', init_gain=0.2, gpu_ids=opt.gpu_ids)
    D1 = init_net(D1, init_type='normal', init_gain=0.2, gpu_ids=opt.gpu_ids)
    D2 = init_net(D2, init_type='normal', init_gain=0.2, gpu_ids=opt.gpu_ids)

    # G1.apply(weights_init)
    if opt.netG1 != '':
        G1.load_state_dict(torch.load(opt.netG1))
    print(G1)

    # D1.apply(weights_init)
    if opt.netD1 != '':
        D1.load_state_dict(torch.load(opt.netD1))
    print(D1)

    # G2.apply(weights_init)
    if opt.netG2 != '':
        G2.load_state_dict(torch.load(opt.netG2))
    print(G2)

    # D2.apply(weights_init)
    if opt.netD2 != '':
        D2.load_state_dict(torch.load(opt.netD2))
    print(D2)

    G1.train()
    D1.train()
    G2.train()
    D2.train()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # model = D1
    # model = model.to(device)
    #
    # summary(model, input_size=(4, 256, 256))

    criterionGAN = nn.BCEWithLogitsLoss().to(opt.gpu_ids[0])
    criterionL1 = nn.L1Loss().to(opt.gpu_ids[0])

    # get randomly sampled validation images and save it
    val_iter = iter(valDataloader)
    val_cpu = val_iter.next()
    val_A = Variable(val_cpu['imgA'].type(Tensor))
    val_B = Variable(val_cpu['imgB'].type(Tensor))
    val_C = Variable(val_cpu['imgC'].type(Tensor))
    vutils.save_image(val_A, '%s/val_results/real_A.png' % opt.output_dir, normalize=True)
    vutils.save_image(val_B, '%s/val_results/real_B.png' % opt.output_dir, normalize=True)
    vutils.save_image(val_C, '%s/val_results/real_C.png' % opt.output_dir, normalize=True)

    # get optimizer
    optimizerD = torch.optim.Adam([{'params': D1.parameters()}, {'params': D2.parameters()}],
                                  lr=opt.lr,
                                  betas=(opt.beta1, 0.999))
    optimizerG = torch.optim.Adam([{'params': G1.parameters()}, {'params': G2.parameters()}],
                                  lr=opt.lr,
                                  betas=(opt.beta1, 0.999))

    total_iters = 0  # the total number of training iterations
    # training loop
    for epoch in range(opt.niter):
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        if epoch > opt.annealStart:
            adjust_learning_rate(optimizerD, opt.lr, epoch, None, opt.annealEvery)
            adjust_learning_rate(optimizerG, opt.lr, epoch, None, opt.annealEvery)

        iter_data_time = time.time()  # timer for data loading per iteration

        for i, data in enumerate(dataloader, 0):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += 1
            epoch_iter += 1

            real_A = Variable(data['imgA'].type(Tensor))
            real_B = Variable(data['imgB'].type(Tensor))
            real_C = Variable(data['imgC'].type(Tensor))

            # compute fake images: G1(A), G2(A, fake_B)
            fake_B = G1(real_A)
            # fake_B = fake_B_pool.query(fake_B)
            fake_C = G2(torch.cat((real_A, fake_B), 1))
            # fake_C = fake_C_pool.query(fake_C)

            # update D1, D2
            set_requires_grad([D1, D2], True)  # enable backprop for D1, D2
            # optimizerD1.zero_grad()  # set D1's gradients to zero
            # optimizerD2.zero_grad()  # set D2's gradients to zero
            optimizerD.zero_grad()

            """Calculate GAN loss for the discriminator"""
            # calculate gradients for D1
            ## Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((real_A, fake_B),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = D1(fake_AB.detach())

            label_d_fake = Variable(Tensor(np.zeros(pred_fake.size())), requires_grad=False)
            label_d_real = Variable(Tensor(np.ones(pred_fake.size())), requires_grad=False)

            loss_D1_fake = criterionGAN(pred_fake, label_d_fake)
            ## Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = D1(real_AB)
            loss_D1_real = criterionGAN(pred_real, label_d_real)
            ## combine loss and calculate gradients
            # loss_D1 = (loss_D1_fake + loss_D1_real) * 0.5
            # torch.autograd.set_detect_anomaly(True)
            # loss_D1.backward(retain_graph=True)

            # calculate gradients for D2
            fake_ABC = torch.cat((real_A, fake_B, fake_C),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake2 = D2(fake_ABC.detach())
            # label_d = torch.tensor(fake_label).expand_as(pred_fake2).to(device)
            loss_D2_fake = criterionGAN(pred_fake2, label_d_fake)
            ## Real
            real_ABC = torch.cat((real_A, real_B, real_C), 1)
            pred_real2 = D2(real_ABC)
            # label_d = torch.tensor(real_label).expand_as(pred_real2).to(device)
            loss_D2_real = criterionGAN(pred_real2, label_d_real)
            ## combine loss and calculate gradients
            # loss_D2 = (loss_D2_fake + loss_D2_real) * 0.5
            # loss_D2.backward(retain_graph=True)

            loss_D = opt.lambda2 * (loss_D1_fake + loss_D1_real) + opt.lambda3 * (loss_D2_fake + loss_D2_real)
            loss_D.backward()

            # update D's weights
            # optimizerD1.step()
            # optimizerD2.step()
            optimizerD.step()

            # update G
            ## D requires no gradients when optimizing G
            set_requires_grad([D1, D2], False)
            ## set G's gradients to zero
            # optimizerG1.zero_grad()
            # optimizerG2.zero_grad()
            optimizerG.zero_grad()

            """Calculate GAN and L1 loss for the generator"""
            # calculate gradients for G1
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = D1(fake_AB.detach())
            # label_d = torch.tensor(real_label).expand_as(pred_fake).to(device)
            loss_G1_GAN = criterionGAN(pred_fake, label_d_real)
            # Second, G(A) = B
            loss_G1_L1 = criterionL1(fake_B, real_B)
            # combine loss and calculate gradients
            # loss_G1 = loss_G1_GAN + loss_G1_L1
            # loss_G1.backward(retain_graph=True)

            # calculate graidents for G2
            # First, G(A) should fake the discriminator
            fake_ABC = torch.cat((real_A, fake_B, fake_C), 1)
            pred_fake = D2(fake_ABC.detach())
            # label_d = torch.tensor(real_label).expand_as(pred_fake).to(device)
            loss_G2_GAN = criterionGAN(pred_fake, label_d_real)
            # Second, G(A) = B
            loss_G2_L1 = criterionL1(fake_C, real_C)
            # combine loss and calculate gradients
            # loss_G2 = loss_G2_GAN + loss_G2_L1
            # loss_G2.backward(retain_graph=True)

            loss_G = loss_G1_GAN + loss_G1_L1 * opt.lambda2 + loss_G2_GAN * opt.lambda3 + loss_G2_L1 * opt.lambda1
            loss_G.backward()

            # udpate G's weights
            # optimizerG1.step()
            # optimizerG2.step()
            optimizerG.step()

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = {'loss_G1_GAN': loss_G1_GAN.item(), 'loss_G1_L1': loss_G1_L1.item(),
                          'loss_D1_real': loss_D1_real.item(), 'loss_D1_fake': loss_D1_fake.item(),
                          'loss_G2_GAN': loss_G2_GAN.item(), 'loss_G2_L1': loss_G2_L1.item(),
                          'loss_D2_real': loss_D2_real.item(), 'loss_D2_fake': loss_D2_fake.item()}
                t_comp = (time.time() - iter_start_time) / opt.batchSize
                lr = optimizerG.param_groups[0]['lr']
                print_current_losses(os.path.join(opt.output_dir, 'train.log'), epoch, lr, epoch_iter, losses, t_comp, t_data)

            if total_iters % opt.display_freq == 0:  # save images
                visuals = {'real_A': real_A,
                            'fake_B': fake_B, 'real_B': real_B,
                            'fake_C':fake_C, 'real_C':real_C}
                # batch_vis = torch.FloatTensor(len(visuals.items()) * opt.batchSize, *real_A.size()[1:4]).fill_(0)
                batch_vis = Tensor(np.zeros((len(visuals.items()) * opt.batchSize, *real_A.size()[1:4])))
                idx = 0
                for i in range(opt.batchSize):
                    for label, image in visuals.items():
                        # if label == 'fake_B':
                        #     image.data[i][image.data[i] > 0] = 1.0
                        #     image.data[i][image.data[i] <= 0] = -1.0
                        batch_vis[idx, :, :, :].copy_(image.data[i])
                        idx = idx + 1
                vutils.save_image(batch_vis, '%s/train_results/generated_epoch_%06d_iter%06d_B.png' % \
                                      (opt.output_dir, epoch, total_iters), nrow=len(visuals.items()), normalize=True)
                    # image_numpy = tensor2im(image)
                    # image_pil = Image.fromarray(image_numpy)
                    # image_pil.save(os.path.join(opt.output_dir, 'train_results', 'epoch%.3d_%s.png' % (epoch, label)))

            if total_iters % opt.evalIter == 0:   # evaluation
                # val_batch_output_B = torch.FloatTensor(val_A.size(0), 1, *val_A.size()[2:4]).fill_(0)
                val_batch_output_B = Tensor(np.zeros((val_A.size(0), 1, *val_A.size()[2:4])))
                # val_batch_output_C = torch.FloatTensor(val_A.size()).fill_(0)
                val_batch_output_C = Tensor(np.zeros(val_A.size()))
                BER = []
                RMSE = []
                for idx in range(val_A.size(0)):
                    single_img = val_A[idx, :, :, :].unsqueeze(0)
                    val_B_single_img = val_B[idx, :, :, :].unsqueeze(0)
                    val_C_single_img = val_C[idx, :, :, :].unsqueeze(0)
                    with torch.no_grad():
                        val_inputv = Variable(single_img)
                    fake_B = G1(val_inputv)
                    fake_C = G2(torch.cat([single_img, fake_B], 1))
                    # Balance error rate
                    fake_B_label = Tensor(np.zeros(fake_B.size())).copy_(fake_B)
                    fake_B_label[fake_B_label > 0] = 1.0
                    fake_B_label[fake_B_label <= 0] = -1.0
                    BER.append(calc_BER(fake_B_label.data, val_B_single_img))
                    # RMSE
                    RMSE.append(calc_RMSE(fake_C.data, val_C_single_img))
                    val_batch_output_B[idx, :, :, :].copy_(fake_B.data.squeeze(0))
                    val_batch_output_C[idx, :, :, :].copy_(fake_C.data.squeeze(0))
                vutils.save_image(val_batch_output_B, '%s/val_results/generated_epoch_%06d_iter%06d_B.png' % \
                                  (opt.output_dir, epoch, total_iters), normalize=True)
                vutils.save_image(val_batch_output_C, '%s/val_results/generated_epoch_%06d_iter%06d_C.png' % \
                                  (opt.output_dir, epoch, total_iters), normalize=True)
                print('BER: %.3f \t RMSE: %.3f' % (sum(BER), sum(RMSE)))

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(G1.module.cpu().state_dict(), '%s/G1_epoch_%d.pth' % (opt.output_dir, epoch))
                torch.save(D1.module.cpu().state_dict(), '%s/D1_epoch_%d.pth' % (opt.output_dir, epoch))
                torch.save(G2.module.cpu().state_dict(), '%s/G2_epoch_%d.pth' % (opt.output_dir, epoch))
                torch.save(D2.module.cpu().state_dict(), '%s/D2_epoch_%d.pth' % (opt.output_dir, epoch))
                G1.cuda(opt.gpu_ids[0])
                G2.cuda(opt.gpu_ids[0])
                D1.cuda(opt.gpu_ids[0])
                D2.cuda(opt.gpu_ids[0])
            else:
                torch.save(G1.cpu().state_dict(), '%s/G1_epoch_%d.pth' % (opt.output_dir, epoch))
                torch.save(D1.cpu().state_dict(), '%s/D1_epoch_%d.pth' % (opt.output_dir, epoch))
                torch.save(G2.cpu().state_dict(), '%s/G2_epoch_%d.pth' % (opt.output_dir, epoch))
                torch.save(D2.cpu().state_dict(), '%s/D2_epoch_%d.pth' % (opt.output_dir, epoch))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter, time.time() - epoch_start_time))


if __name__ == '__main__':
    args = parse_args()
    main(args)