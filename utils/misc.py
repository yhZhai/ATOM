import time
import datetime
from collections import defaultdict

import torch
import prettytable as pt


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def cleanexit():
    import sys
    import os
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def freeze_joints(x, joints_to_freeze):
    # Freezes selected joint *rotations* as they appear in the first frame
    # x [bs, [root+n_joints], joint_dim(6), seqlen]
    frozen = x.detach().clone()
    frozen[:, joints_to_freeze, :, :] = frozen[:, joints_to_freeze, :, :1]
    return frozen


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.avg: .5f}'


class MetricLogger(object):
    def __init__(self, delimiter=" ", writer=None, suffix=None):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter
        self.writer = writer
        self.suffix = suffix

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f'Unsupport type {type(v)}.'
            self.meters[k].update(v)

    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def get_meters(self, add_suffix: bool = False):
        result = {}
        for k, v in self.meters.items():
            result[k if not add_suffix else '_'.join([k, self.suffix])] = v.avg
        return result
    
    def prepend_subprefix(self, subprefix: str):
        old_keys = list(self.meters.keys())
        for k in old_keys:
            self.meters[k.replace('/', f'/{subprefix}')] = self.meters[k]
        for k in old_keys:
            del self.meters[k]

    def log_every(self, iterable, print_freq=10, header=''):
        i = 0
        start_time = time.time()
        end = time.time()
        iter_time = AverageMeter()
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'iter time: {time}s'
        ])
        for obj in iterable:
            yield i, obj
            iter_time.update(time.time() - end)
            if (i + 1) % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i + 1, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time)).replace('  ', ' '))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f}s / it)'.format(header, total_time_str,
                                                        total_time / len(iterable)))

    def write_tensorboard(self, step):
        if self.writer is not None:
            for k, v in self.meters.items():
                # if self.suffix:
                #     self.writer.add_scalar(
                #         '{}/{}'.format(k, self.suffix), v.avg, step)
                # else:
                self.writer.add_scalar(k, v.avg, step)

    def stat_table(self):
        tb = pt.PrettyTable(field_names=['Metrics', 'Values'])
        for name, meter in self.meters.items():
            tb.add_row([name, str(meter)])
        return tb.get_string()

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str).replace('  ', ' ')
