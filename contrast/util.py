import argparse

import torch
import torch.distributed as dist


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class DistributedShufle:
    @staticmethod
    def forward_shuffle(x, epoch):
        """ forward shuffle, return shuffled batch of x from all processes.
        epoch is used as manual seed to make sure the shuffle id in all process is same.
        """
        x_all = dist_collect(x)
        forward_inds, backward_inds = DistributedShufle.get_shuffle_ids(x_all.shape[0], epoch)

        forward_inds_local = DistributedShufle.get_local_id(forward_inds)

        return x_all[forward_inds_local], backward_inds

    @staticmethod
    def backward_shuffle(x, backward_inds, return_local=True):
        """ backward shuffle, return data which have been shuffled back
        x is the shared data, should be local data
        if return_local, only return the local batch data of x.
            otherwise, return collected all data on all process.
        """
        x_all = dist_collect(x)
        if return_local:
            backward_inds_local = DistributedShufle.get_local_id(backward_inds)
            return x_all[backward_inds], x_all[backward_inds_local]
        else:
            return x_all[backward_inds]

    @staticmethod
    def get_local_id(ids):
        return ids.chunk(dist.get_world_size())[dist.get_rank()]

    @staticmethod
    def get_shuffle_ids(bsz, epoch):
        """generate shuffle ids for ShuffleBN"""
        torch.manual_seed(epoch)
        # global forward shuffle id  for all process
        forward_inds = torch.randperm(bsz).long().cuda()

        # global backward shuffle id
        backward_inds = torch.zeros(forward_inds.shape[0]).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)

        return forward_inds, backward_inds


def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model.eval()
    model.apply(set_bn_train_helper)


class BatchNormWrapper(torch.nn.Module):
    def __init__(self, m):
        super(BatchNormWrapper, self).__init__()
        self.m = m
        self.m.eval()  # Set the batch norm to eval mode

    def forward(self, x):
        input_type = x.dtype
        x = self.m(x.float())
        return x.to(input_type)


def wrap_bn_fp16(model):
    for child_name, child in model.named_children():
        # if isinstance(child, torch.nn.BatchNorm2d):
        classname = child.__class__.__name__
        if classname.find('BatchNorm') != -1:
            setattr(model, child_name, BatchNormWrapper(child))
        else:
            wrap_bn_fp16(child)


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


class MyHelpFormatter(argparse.MetavarTypeHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
