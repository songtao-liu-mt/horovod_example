import torch
import argparse
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd
import os
import math
from tqdm import tqdm
import torchvision
from torchvision import datasets, transforms
from models.resnet import resnet50
from torch.profiler import profile, record_function, ProfilerActivity

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training')
parser.add_argument('--checkpoint-format', default='./save/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=128,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.1,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=1,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.0005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        if hvd.rank() == 0:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                for batch_idx, (data, target) in enumerate(train_loader):

                    if args.cuda:
                        #data, target = data.cuda(), target.cuda()
                        data, target = data.to("mtgpu"), target.to("mtgpu")
                    optimizer.zero_grad()
                    # Split data into sub-batches of size batch_size
                    for i in range(0, len(data), args.batch_size):
                        data_batch = data[i:i + args.batch_size]
                        target_batch = target[i:i + args.batch_size]
                        output = model(data_batch)
                        train_accuracy.update(accuracy(output, target_batch))
                        loss = F.cross_entropy(output, target_batch)
                        train_loss.update(loss)
                        # Average gradients among sub-batches
                        loss.div_(math.ceil(float(len(data)) / args.batch_size))
                        loss.backward()
                    # Gradient is applied across all ranks
                    optimizer.step()
                    t.set_postfix({'loss': train_loss.avg.item(),
                                'lr': optimizer.param_groups[0]['lr'],
                                'accuracy': 100. * train_accuracy.avg.item()})
                    t.update(1)
            prof.export_chrome_trace("trace-nonblocking.json")
        else:
            for batch_idx, (data, target) in enumerate(train_loader):

                if args.cuda:
                    #data, target = data.cuda(), target.cuda()
                    data, target = data.to("mtgpu"), target.to("mtgpu")
                optimizer.zero_grad()
                # Split data into sub-batches of size batch_size
                for i in range(0, len(data), args.batch_size):
                    data_batch = data[i:i + args.batch_size]
                    target_batch = target[i:i + args.batch_size]
                    output = model(data_batch)
                    train_accuracy.update(accuracy(output, target_batch))
                    loss = F.cross_entropy(output, target_batch)
                    train_loss.update(loss)
                    # Average gradients among sub-batches
                    loss.div_(math.ceil(float(len(data)) / args.batch_size))
                    loss.backward()
                # Gradient is applied across all ranks
                optimizer.step()
                t.set_postfix({'loss': train_loss.avg.item(),
                            'lr': optimizer.param_groups[0]['lr'],
                            'accuracy': 100. * train_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    #data, target = data.cuda(), target.cuda()
                    data, target = data.to("mtgpu"), target.to("mtgpu")
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0 and ((epoch +1) % 20 == 0):
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': train_scheduler.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    args = parser.parse_args()
    #args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.cuda = not args.no_cuda

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        os.environ["PVR_GPUIDX"] = str(hvd.local_rank())
        os.environ["MTGPU_MAX_MEM_USAGE_GB"] = "14"
        import musa_torch_extension
        #torch.cuda.set_device(hvd.local_rank())
        #torch.cuda.manual_seed(args.seed)

    #cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    if args.resume:
        for try_epoch in range(args.epochs, 0, -1):
            if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
                resume_from_epoch = try_epoch
                break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(8)

    #kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    kwargs = {'num_workers': 8} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, 
                             std=CIFAR100_TRAIN_STD)
    ])

    train_dataset = \
        datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
                             std=CIFAR100_TRAIN_STD)
    ])

    val_dataset = \
        datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)


    # Set up standard ResNet-50 model.
    #model = models.resnet.resnet50(num_classes=100)
    model = resnet50()

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    #lr_scaler = (args.batches_per_allreduce * hvd.size()) if not args.use_adasum else 1
    lr_scaler = (allreduce_batch_size * hvd.size()) // 128 if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        #model.cuda()
        model.to("mtgpu")
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_scheduler.load_state_dict(checkpoint['scheduler'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
        validate(epoch)
        save_checkpoint(epoch)
