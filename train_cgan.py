import logging
import time
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.data.distributed
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from RadioAudio_meldataset import Meldataset
from arguments import ArgParser

from loss_cgan import adversarial_loss
from mel_utils import AverageMeter, dist_average
from eval import evaluate, evaluate_visual
from cgan_model import cGAN

def checkpoint(net, history, epoch, optimizer, count_iter, args):
    package = dict()

    package['net'] = net.state_dict()
    package['optimizer'] = optimizer.state_dict()
    package['epoch'] = epoch
    package['history'] = history
    package['iter'] = count_iter

    logging.info(f'Saving checkpoints at {epoch} epochs.')
    suffix_first = 'first.pth'
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    if not os.path.exists(args.save_ckpt):
        os.makedirs(args.save_ckpt)
        torch.save(package, '{}/package_{}'.format(args.save_ckpt, suffix_latest))

    if epoch == 0:
        torch.save(net.state_dict(), '{}/package_{}'.format(args.save_ckpt, suffix_first))

    cur_err = package['history']['eval_loss'][-1]
    if cur_err < args.best_loss:
        args.best_loss = cur_err
        torch.save(net.state_dict(), '{}/net_{}'.format(args.save_ckpt, suffix_best))

def train_one_epoch(net, train_loader, optimizer, criterion, epoch, accumulated_iter, tb_log, args):
    batch_time = AverageMeter()
    dataload_time = AverageMeter()
    total_loss = 0

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='batch') as pbar:
        torch.cuda.synchronize()
        tic = time.perf_counter()

        for i, batch_data in enumerate(train_loader):
            mmwave, real_speech, _ = batch_data
            mmwave = mmwave.cuda()
            real_speech = real_speech.cuda()
            # print(mmwave.shape, real_speech.shape)

            # measure data time
            torch.cuda.synchronize()
            dataload_time.update(time.perf_counter() - tic)

            enhanced_speech, real_output, fake_output = net(mmwave, real_speech)

            # with torch.autograd.set_detect_anomaly(True):
            loss = criterion(real_speech, enhanced_speech, real_output, fake_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure total time
            torch.cuda.synchronize()
            batch_time.update(time.perf_counter() - tic)
            tic = time.perf_counter()

            accumulated_iter += 1
            total_loss += loss.item()

            # display
            if i % args.disp_iter == 0:
                print('Epoch: [{}][{}/{}], batch time: {:.3f}, Data time: {:.3f}, loss: {:.4f}'
                  .format(epoch, i, len(train_loader),
                          batch_time.average(), dataload_time.average(), loss.item()))

            # add tensorboard
            tb_log.add_scalar('train/loss', loss.item(), accumulated_iter)
            tb_log.add_scalar('train/epoch', epoch, accumulated_iter)
            tb_log.add_scalar('train/learning_rate_generator', optimizer.param_groups[0]['lr'], accumulated_iter)
            tb_log.add_scalar('train/learning_rate_discriminator', optimizer.param_groups[1]['lr'], accumulated_iter)

            # Manually update the progress bar, useful for streams such as reading files.
            pbar.update(1)
            if args.distributed:
                pbar.set_postfix(**{'loss (batch)': dist_average([total_loss / (i + 1)], i + 1)[0]})
            else:
                pbar.set_postfix(**{'loss (batch)': total_loss / (i+1)})

    return accumulated_iter

def train_net(args):
    # 0. initialize distribute and tensorboard
    args.gpu = 2
    args.world_size = 3
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.local_rank)
        args.world_size = torch.distributed.get_world_size()

    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    tb_log = SummaryWriter(log_dir=args.tensorboard_dir)

    # 1. Create dataset
    train_sampler = None
    val_sampler = None
    train_set = Meldataset(args.list_train)
    val_set = Meldataset(args.list_val)
    # train_set = Meldataset2(args.list_train, tile=True, expmel_len=128, stride=64)
    # val_set = Meldataset2(args.list_val, tile=False, expmel_len=128, stride=128)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    # 2. Create data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                              shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, num_workers=4,
                            pin_memory=True, shuffle=False, sampler=val_sampler) #drop_last=False

    # 3. Build the network
    ###############################
    net = cGAN(in_channels_d=2, in_channels_g=1, out_channels_g=1, bilinear=False)

    # warp the network
    if args.distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.cuda()
    # distribute
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net,
                            device_ids=[args.local_rank], output_device=args.local_rank)

    # 4. Set up the optimizer, the criterion, and the learning rate scheduler.
    # optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    optimizer = optim.Adam([
        {'params': net.module.generator.parameters(), 'lr': args.lr_g, 'weight_decay': 1e-4},
        {'params': net.module.discriminator.parameters(), 'lr': args.lr_d, 'weight_decay': 1e-4},
    ])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.4)
    criterion = adversarial_loss()


    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate of generator:   {args.lr_g}
        Learning rate of discriminator:   {args.lr_d}
    ''')

    # 5. load  checkpoint
    start_epoch = count_iter = 0
    if args.load_checkpoint:
        dist.barrier()
        logging.info(f'Loading checkpoint net from: {args.load_checkpoint}')
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        package = torch.load(args.load_checkpoint, map_location='cpu')
        net.load_state_dict(package['net'])
        optimizer.load_state_dict(package['optimizer'])
        start_epoch = package['epoch']
        count_iter = package['iter']

    # initialize checkpoint package
    history = {'eval_loss': []}

    # 6. Begin training
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch) # syn different node in one epoch
        net.train()
        start = time.time()

        logging.info('-' * 70)
        logging.info('Training...')
        # train one epoch
        count_iter = train_one_epoch(net, train_loader, optimizer, criterion, epoch, count_iter, tb_log, args)
        logging.info(f'Train Summary | End of Epoch {epoch} | Time {time.time() - start:.2f}s')

        scheduler.step()

        # # Evaluation round
        logging.info('-' * 70)
        logging.info('Evaluating...')
        evaluate(net, val_loader, criterion, epoch, history, tb_log, count_iter, args)

        # evaluate samples every 'eval_every' argument number of epochs also evaluate on last epoch
        if (epoch + 1) % args.metrics_every == 0 or epoch == 0 or epoch == args.epochs - 1 :
            # Evaluate on the testset
            logging.info('-' * 70)
            logging.info('Calculating metrics...')
            evaluate_visual(net, val_loader, epoch, tb_log, count_iter, args)

        if args.local_rank == 0:
            #checkpointing
            checkpoint(net, history, epoch, optimizer, count_iter, args)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'

    parser = ArgParser()
    args = parser.parse_train_arguments()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    train_net(args)
