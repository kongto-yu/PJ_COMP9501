import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import librosa
import librosa.display
from mel_utils import AverageMeter
import torch
from mel_utils import dist_average
from utils import *
import pysepm


def melamp_visual(audio_amp, pred_amp, radio_amp):
    # record amplitude in tensorboard
    # audio_amp = 10**audio_amp
    # pred_amp = 10**pred_amp
    B = pred_amp.size(0)
    b = np.random.randint(0, B-1)
    amp_list = [audio_amp[b, 0, ...].cpu().numpy().transpose(1,0), radio_amp[b, 0, ...].cpu().numpy().transpose(1,0),
                pred_amp[b, 0, ...].cpu().numpy().transpose(1,0)]
    return amp_list

def evaluate(model, val_loader, criterion, epoch, history, tb_log, count_iter, args):
    logging.info(f'Evaluating at {epoch} epochs...')

    # initialize metrics
    # distribute training
    total_loss = 0
    total_count = 0

    # without distribute
    loss_metrics = AverageMeter()

    model.eval()
    with torch.no_grad():
        # num_iters = len(val_loader)
        # prefetcher = data_prefetcher(val_loader)
        # audio_amp, _, radio_amp = prefetcher.next()
        # i = 0
        with tqdm(total=len(val_loader), desc='validation', unit='batch', leave=False) as pbar:
            for i, batch_data in enumerate(val_loader):
            # while radio_amp is not None:
            #
            #     i += 1
            #     if i > num_iters:
            #         break
                mmwave, real_speech, _ = batch_data
                mmwave = mmwave.cuda()
                real_speech = real_speech.cuda()
                enhanced_speech, real_output, fake_output = model(mmwave, real_speech)
                eval_loss = criterion(real_speech, enhanced_speech, real_output, fake_output)

                # distribute
                # total_loss += eval_loss.item()
                # total_count += real_speech.size(0)
                # without distribute
                loss_metrics.update(eval_loss.item())

                pbar.update(1)

                # audio_amp, _,radio_amp = prefetcher.next()

                # del eval_loss

    # distribute
    # evaluation_loss = dist_average([total_loss / (i + 1)], i + 1)[0]
    # history['eval_loss'].append(evaluation_loss)
    # tb_log.add_scalar('eval/loss', evaluation_loss, count_iter)
    # print('Evaluation Summary: Epoch: {}, Loss: {:.4f}'.format(epoch, evaluation_loss))

    # without distribute
    history['eval_loss'].append(loss_metrics.average())
    tb_log.add_scalar('eval/loss', eval_loss, count_iter)
    print('Evaluation Summary: Epoch: {}, Loss: {:.4f}'.format(epoch, loss_metrics.average()))


def evaluate_visual(model, val_loader, epoch, tb_log, count_iter, args):
    logging.info(f'Upload the mel spectrogram at {epoch} /epochs during evaluation')

    # initialize metrics
    tb_amp_list = []

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc='metric for val', unit='batch', leave=False) as pbar:

            for i, batch_data in enumerate(val_loader):
                mmwave, real_speech, filename = batch_data
                mmwave = mmwave.cuda()
                real_speech = real_speech.cuda()
                

                enhanced_speech, _, _ = model(mmwave, real_speech)

                # calculate the metrics and visualize
                amp_list = melamp_visual(real_speech, mmwave, enhanced_speech)

                # distribute
                tb_amp_list.append(amp_list)

                pbar.update(1)
                # audio_amp, audio_raw, radio_amp = prefetcher.next()

    # add spectrogram in tensorboard
    b = np.random.randint(0, len(tb_amp_list) - 1)
    fig, axes = plt.subplots(3, 1, figsize=(10,10))
    for k, mag in enumerate(tb_amp_list[b]):
        axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                          f"std: {np.std(mag):.3f}, "
                          f"max: {np.max(mag):.3f}, "
                          f"min: {np.min(mag):.3f}")
        librosa.display.specshow(mag, x_axis='s', y_axis="mel", ax=axes[k], sr=args.audRate, hop_length=args.audio_hop_len)
    plt.tight_layout()
    tb_log.add_figure(f'eval/{epoch}_melspect', fig, count_iter)


def cal_metrics(ref_amp, pred_amp, ref_phase, pred_phase, args):
    # initialize metrics
    SNRseg_metric = AverageMeter()
    stoi_metric = AverageMeter()
    llr_metric = AverageMeter()

    #unwrap log scale
    pred_amp = torch.exp(pred_amp).detach()
    B = pred_amp.size(0)
    H = args.audio_nfft // 2 + 1
    T = pred_amp.size(3)
    grid_unwarp = torch.from_numpy(warpgrid(B, H, T, warp=False)).cuda()
    pred_amp_unwrap = F.grid_sample(pred_amp, grid_unwarp)

    assert pred_amp_unwrap.size() == ref_amp.size(), f'the size of unwrapped pred amp should the same as the ref amp'

    # convert into numpy
    pred_amp_unwrap = pred_amp_unwrap.cpu().numpy()
    pred_phase = pred_phase.cpu().numpy()
    ref_amp = ref_amp.cpu( ).numpy()
    ref_phase = ref_phase.cpu( ).numpy()

    for i in range(B):
        #iSTFT construction
        audio_signal = istft_reconstruction(ref_amp[i, 0], ref_phase[i, 0])
        radio_signal = istft_reconstruction(pred_amp_unwrap[i, 0], pred_phase[i, 0])

        # calculate metrics
        SNRseg = pysepm.SNRseg(audio_signal, radio_signal, args.audRate)
        stoi = pysepm.stoi(audio_signal, radio_signal, args.audRate)
        llr = pysepm.llr(audio_signal, radio_signal, args.audRate)

        SNRseg_metric.update(SNRseg)
        stoi_metric.update(stoi)
        llr_metric.update(llr)

    return SNRseg_metric.sum, stoi_metric.sum, llr_metric.sum

