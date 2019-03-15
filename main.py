import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import misc
import argparse
import os

import model

def train(args):
    if args.dataset.lower() == 'celeba':
        train_loader, _, _ = misc.load_celebA(args.batch_s, args.img_s)
        img_c = 3
    elif args.dataset.lower() == 'lsun':
        train_loader, val_loader, _ = misc.load_LSUN(args.batch_s, args.img_s)
        img_c = 3
    elif args.dataset.lower() == 'imagenet':
        train_loader, val_loader, _ = misc.load_imagenet(args.batch_s, args.img_s)
        img_c = 3
    elif args.dataset.lower() == 'mnist':
        train_loader, val_loader, _ = misc.load_mnist(args.batch_s, args.img_s)
        img_c = 1
    else:
        raise NotImplementedError

    fm_gen = [args.base_fm_n*pow(2,args.layer_n-1-l) for l in range(args.layer_n)]
    fm_disc = [args.base_fm_n*pow(2,l) for l in range(args.layer_n)]

    gen = model.Generator(args.z_dim, img_c, fm_gen).cuda()
    gen.apply(model.init_weights)
    disc = model.Discriminator(img_c, fm_disc).cuda()
    disc.apply(model.init_weights)

    criterion = nn.BCELoss()
    label_real = 1
    label_fake = 0

    optim_gen = optim.Adam(gen.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    optim_disc = optim.Adam(disc.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))

    if args.resume:
        filename = args.ckpt_dir + args.resume
        if os.path.isfile(filename):
            print("==> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch'] + 1
            gen.load_state_dict(checkpoint['state_dict_gen'])
            disc.load_state_dict(checkpoint['state_dict_disc'])
            optim_gen.load_state_dict(checkpoint['optimizer_gen'])
            optim_disc.load_state_dict(checkpoint['optimizer_disc'])
            print("==> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            print("==> no checkpoint found at '{}'".format(filename))
    else:
        start_epoch = 0

    if not os.path.isdir(args.img_dir):
        os.system('mkdir ' + args.img_dir)
    if not os.path.isdir(args.ckpt_dir):
        os.system('mkdir ' + args.ckpt_dir)

    #########################################
    #### Train
    ## 1. Update Discriminator: maximize log(D(x)) + log(1-D(G(z)))
    # 1-1. with real image x
    # 1-2. with fake image G(z)
    ## 2. Update Generator: maximize log(D(G(z)))
    for e in range(args.epochs):
        epoch = start_epoch + e
        loss_meter_gen = AverageMeter()
        loss_meter_disc = AverageMeter()
        out_meter_disc_f = AverageMeter()
        out_meter_disc_r = AverageMeter()
        out_meter_disc_g = AverageMeter()
        for i, data in enumerate(train_loader):
            img_real, _ = data
            img_real = img_real.cuda()
            batch_s = img_real.size(0)
            
            optim_disc.zero_grad()

            # 1-1. with real image x            
            label_r = torch.full((batch_s, 1), label_real).cuda()
            out_disc_r = disc(img_real).view(batch_s, -1)
            error_disc_r = criterion(out_disc_r, label_r)
            error_disc_r.backward()

            # 1-2. with fake image G(z)
            img_fake = gen(torch.randn(batch_s, args.z_dim, 1, 1).cuda())
            label_f = torch.full((batch_s, 1), label_fake).cuda()
            out_disc_f = disc(img_fake.detach()).view(batch_s, -1)
            error_disc_f = criterion(out_disc_f, label_f)
            
            error_disc = error_disc_r + error_disc_f
            error_disc_f.backward()
            optim_disc.step()

            # 2. Update Generator
            for g_iter in range(3):
                img_fake = gen(torch.randn(batch_s, args.z_dim, 1, 1).cuda())
                out_disc_g = disc(img_fake).view(batch_s, -1)
                error_gen = criterion(out_disc_g, label_r)
                optim_gen.zero_grad()
                error_gen.backward()
                optim_gen.step()

            loss_meter_gen.update(error_gen.item(), batch_s)
            loss_meter_disc.update(error_disc.item(), batch_s)
            out_meter_disc_f.update(torch.sum(out_disc_f).item(), batch_s)
            out_meter_disc_r.update(torch.sum(out_disc_r).item(), batch_s)
            out_meter_disc_g.update(torch.sum(out_disc_g).item(), batch_s)

            if i % args.log_term == 0:
                print('epoch: %d, batch: %d \t Loss(D/G): %.4f / %.4f \t D(R/F/G): %.4f / %.4f / %.4f'
                        % (epoch, i, loss_meter_disc.avg, loss_meter_gen.avg, 
                            out_meter_disc_r.avg/batch_s, out_meter_disc_f.avg/batch_s, out_meter_disc_g.avg/batch_s))
                fd = open('save_log.txt', 'a')
                fd.write('epoch: %d, batch: %d \t Loss(D/G): /%.4f / %.4f/ || D(R/F/G): /%.4f / %.4f / %.4f/ \n'
                        % (epoch, i, loss_meter_disc.avg, loss_meter_gen.avg, 
                            out_meter_disc_r.avg, out_meter_disc_f.avg, out_meter_disc_g.avg))
                fd.close()
                misc.plot_samples_from_images(img_fake, batch_s, args.img_dir, 'img_e{}b{}.jpg'.format(epoch, i))

                torch.save({
                    'epoch': epoch, 
                    'state_dict_gen': gen.state_dict(), 
                    'state_dict_disc': disc.state_dict(),
                    'optimizer_gen': optim_gen.state_dict(), 
                    'optimizer_disc': optim_disc.state_dict()
                    }, 
                    args.ckpt_dir + 'checkpoint_e{}b{}.pt'.format(epoch, i))
                
                loss_meter_gen = AverageMeter()
                loss_meter_disc = AverageMeter()
                out_meter_disc_f = AverageMeter()
                out_meter_disc_r = AverageMeter()
                out_meter_disc_g = AverageMeter()
                

def test(args):
    raise NotImplementedError

class AverageMeter(object):
    '''
    from https://github.com/pytorch/examples/blob/master/imagenet/main.py.
    Computes and stores the average and current values
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main(args):
    if args.train:
        train(args)
    else:
        test(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch implementation of DCGAN')

    parser.add_argument('--base_fm_n', default=64, type=int, help='The number of base FM')
    parser.add_argument('--learning_rate', '-lr', default=0.0002, type=float, help='Learning rate for Adam')
    parser.add_argument('--beta1', default=0.5, type=float, help='Beta1 for Adam')
    parser.add_argument('--epochs', default=1000, type=int, help='Total epoch')
    parser.add_argument('--dataset', default='celeba', type=str, help='Dataset name: MNIST, CelebA, LSUN or imagenet')
    parser.add_argument('--z_dim', default=100, type=int, help='Dimension of z')
    parser.add_argument('--resume', default='', type=str, help='Name of previouse checkpoint file (defalut: None)')
    parser.add_argument('--img_dir', default='/export/scratch/a/choi574/saved_model/gan_face/plots/', type=str, help='Directory to save test plots')
    parser.add_argument('--ckpt_dir', default='/export/scratch/a/choi574/saved_model/gan_face/', type=str, help='Name of previouse checkpoint dir')
    parser.add_argument('--batch_s', default=128, type=int, help='Size of batch')
    parser.add_argument('--img_s', default=64, type=int, help='Size of Image')
    parser.add_argument('--layer_n', default=4, type=int, help='The number of layers')
    parser.add_argument('--train', default=True, type=misc.str2bool, help='Train or generate')
    parser.add_argument('--log_term', default=10, type=int, help='log recording term (save every N batch)')
    args = parser.parse_args()
    main(args)


