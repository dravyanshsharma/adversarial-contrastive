"""
Code for SimCLR pre-training

"""
import argparse
import os
import time
import json

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from torch import nn

from contrast.NCE import IntraBatchContrast, NCESoftmaxLoss
from contrast.dataset import STL10Instance
from contrast.transform import RandomGaussianBlur
from contrast.logger import setup_logger
from contrast.models.resnet import MyModel
from contrast.util import AverageMeter, MyHelpFormatter, DistributedShufle, set_bn_train, moment_update, accuracy
from contrast.lr_scheduler import get_scheduler
from torch.autograd import Variable

# try:
#     # noinspection PyUnresolvedReferences
#     from apex import amp
# except ImportError:
amp = None


def parse_option():
    parser = argparse.ArgumentParser('self-supervised training', formatter_class=MyHelpFormatter)

    # dataset
    parser.add_argument('--data-dir', type=str, required=True, help='root director of dataset')
    parser.add_argument('--dataset', type=str, default='stl10', choices=['stl10', 'imagenet'],
                        help='dataset to training')
    parser.add_argument('--crop', type=float, default=0.08, help='minimum crop')
    parser.add_argument('--batch-size', type=int, default=512, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')

    # model and loss function
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help="backbone model")
    parser.add_argument('--model-width', type=int, default=1, help='width of resnet, eg, 1, 2, 4')
    parser.add_argument('--nce-t', type=float, default=0.1, help='NCE temperature')

    # optimization
    parser.add_argument('--base-learning-rate', '--base-lr', type=float, default=0.1,
                        help='Base learning rate for batch 128. The actual lr follows the linear scaling rule')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')

    # io
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')

    # misc
    parser.add_argument('--distributed', action='store_true', default=False, help='use distributed training')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')

    args = parser.parse_args()

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)

    return args

def get_loader(args):
    # set the data loader
    image_size = 96
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        RandomGaussianBlur(int(0.1 * image_size), p=0.5),
        transforms.ToTensor(),
    ])

    train_dataset = STL10Instance(args.data_dir, split='train+unlabeled', transform=train_transform, two_crop=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            sampler=train_sampler, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return train_loader


def load_checkpoint(args, model, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(args.resume))

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model_to_loaded = model.module if hasattr(model, 'module') else model
    model_to_loaded.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if args.amp_opt_level != "O0" and checkpoint['opt'].amp_opt_level != "O0":
        amp.load_state_dict(checkpoint['amp'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler):

    model_to_saved = model.module if hasattr(model, 'module') else model
    state = {
        'opt': args,
        'model': model_to_saved.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if args.amp_opt_level != "O0":
        state['amp'] = amp.state_dict()
    if epoch % args.save_freq == 0:
        logger.info('==> Saving...')
        torch.save(state, os.path.join(args.output_dir, 'current_sparse5.pth'))



def main(args):
    train_loader = get_loader(args)
    n_data = len(train_loader.dataset)
    logger.info(f"length of training dataset: {n_data}")

    model = MyModel(width=args.model_width)

    if not args.distributed and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of trainable parameters: {total_params}")
    lr_mult = dist.get_world_size() if args.distributed else 1
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr_mult * args.batch_size / 128 * args.base_learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = get_scheduler(optimizer, len(train_loader), args)

    criterion = NCESoftmaxLoss().cuda()


    if args.amp_opt_level != "O0":
        if amp is None:
            logger.warning(f"apex is not installed but amp_opt_level is set to {args.amp_opt_level}, ignoring.\n"
                           "you should install apex from https://github.com/NVIDIA/apex#quick-start first")
            args.amp_opt_level = "O0"
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)
        
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, optimizer, scheduler)

    # tensorboard
    if not args.distributed or dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        tic = time.time()
        loss = train_once(epoch, train_loader, model, criterion, optimizer, scheduler, args)

        logger.info('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))

        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('ins_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if not args.distributed or dist.get_rank() == 0:
            # save model
            save_checkpoint(args, epoch, model, scheduler, optimizer)

def train_once(epoch, train_loader, model, criterion, optimizer, scheduler, args):
    """
    one epoch training for simclr
    """
    model.train()

    loss_meter = AverageMeter()
    prob_meter = AverageMeter()


    for idx, (inputs, _) in enumerate(train_loader):

        # forward
        bsz = inputs.size(0)
        x1, x2 = torch.split(inputs, [3, 3], dim=1)
        inputs = torch.cat([x1, x2], dim=0).cuda(non_blocking=True)
        repres = model(inputs)
        
        # contrast
        contrast = IntraBatchContrast(args.nce_t, bsz).cuda()
        out = contrast(repres)

        # loss
        loss = criterion(out)

        # backward
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        # update meters
        prob = F.softmax(out.detach(), dim=1)[:, 0].mean()
        prob_meter.update(prob.item(), bsz)
        loss_meter.update(loss.item(), bsz)

        # print info
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Train: [{epoch}][{idx:3d}/{len(train_loader)}]\t'
                        f'LR {lr:.6f}\t'
                        f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                        f'prob {prob_meter.val:.3f} ({prob_meter.avg:.3f})')


    return loss_meter.avg


if __name__ == '__main__':
    opt = parse_option()

    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    if opt.distributed:
        logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="SimCLR")
    else:
        logger = setup_logger(output=opt.output_dir, distributed_rank=0, name="SimCLR")
    if opt.distributed and dist.get_rank() == 0:
        logger.info("Config: {}".format(opt))
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    main(opt)
