import argparse

class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train a Unet')
        # vit
        parser.add_argument('--img_size', type=int, default = 80,
                            help = 'input patch size of network input')
        parser.add_argument('--num_classes', type=int,
                            default=1, help='output channel of network')
        parser.add_argument('--n_skip', type=int, default=3,
                            help='using number of skip-connect, default is num')
        parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                            help='select one vit model')
        parser.add_argument('--vit_patches_size', type=int, default=16,
                            help='vit_patches_size, default is 16')

        # Model related arguments
        parser.add_argument('--ar', dest='audRate', help='audio sample rate', default=8000, type=int)
        parser.add_argument('--rr', dest='radRate', help='radio sample rate', default=8000, type=int)
        parser.add_argument('--as', dest='audSec', help='expected audio second', default=3, type=int)
        parser.add_argument('--std', dest='stride', help='stride for sliding signal segment', default=1)

        # stft parameter
        parser.add_argument('--audio_nfft', dest='audio_nfft', help='size of Fourier transform', default=512, type=int)
        parser.add_argument('--audio_hop', dest='audio_hop_len', help='the distance between neighboring sliding window frames',
                            default=128)
        parser.add_argument('--radio_nfft', dest='radio_nfft', help='size of Fourier transform', default=512, type=int)
        parser.add_argument('--radio_hop', dest='radio_hop_len', help='the distance between neighboring sliding window frames',
                            default=128)

        # mel parameter
        parser.add_argument('--n_mel', dest='n_mel', help='Number of mel basis.', default=80, type=int)
        parser.add_argument('--f_min', dest='f_min', help='Minimum frequency in mel basis calculation.', default=60, type=int)
        parser.add_argument('--f_max', dest='f_max', help='Maximum frequency in mel basis calculation.', default=4000, type=int)

        # visualize
        parser.add_argument('--disp_iter', dest='disp_iter',
                            help='frequency to display the training information', default=50) #40

        #training
        # learning rate
        parser.add_argument('--bs', dest='batch_size',
                            help='train batch size', default=512) #LJSpeech:256 TIMIT:128
        parser.add_argument('--val_bs', dest='val_batch_size',
                            help='val batch size', default=50)
        parser.add_argument('--lr', dest='learning_rate',
                            help='learning rate for training', default=0.004)  #0.004, vit: 0.01
        parser.add_argument('--lr_g', dest='lr_g',
                            help='learning rate for generator training', default=0.0004)
        parser.add_argument('--lr_d', dest='lr_d',
                            help='learning rate for discriminator training', default=0.0002)
        parser.add_argument('--epochs', dest='epochs',
                            help='the number of training epochs', default=1000) # 900
        parser.add_argument('--step_size', dest='step_size',
                            help='step size epoch to adjust the learning rate', default=200) #350 #35

        # loss setting and parameter
        parser.add_argument('--amp_wave_loss', dest='amp_wave_loss',
                            help='combine wave loss and wrapped stft amplitude loss', default=False)
        parser.add_argument('--adjust_loss', dest='adjust_loss',
                            help='adjust the loss function at which epoch', default=45)
        parser.add_argument('--factor_wave', dest='factor_wave', help='wave loss factor', default=0.5)

        # distributed
        parser.add_argument('--dist', dest='distributed',
                            help='distributed training', default=False)
        parser.add_argument('--local_rank', dest='local_rank',
                            help='distributed training for local_rank', default=0, type=int)

        # directory
        parser.add_argument('--list_train', dest='list_train', help='list of training data',
                            default='/home/yujt/radio2text/dataset/glass/'
                                    'train_LJSpeech_mel_glass_norm.csv')
        parser.add_argument('--list_val', dest='list_val', help='list of evaluation data',
                            default='/home/yujt/radio2text/dataset/glass/'
                                    'eval_LJSpeech_mel_glass_norm.csv')

        # tensorboard
        parser.add_argument('--tb_log', dest='tensorboard_dir',
                            help='output dir to save tensorboard file',
                            default='/home/yujt/radio2text/checkpoint_glass/tb_log_norm/cgan_ablation/')

        # checkpoint
        parser.add_argument('--load_ckpt', dest='load_checkpoint',
                            help='load from pre-trained checkpoint', default='')
        parser.add_argument('--load_best', dest='load_best_model',
                            help='load from best model', default='')
        parser.add_argument('--best_loss', dest='best_loss',
                            help='best loss for evaluation', default=float("inf"))
        parser.add_argument('--save_ckpt', dest='save_ckpt',
                            help='save model to checkpoint path',
                            default='/home/yujt/radio2text/checkpoint_glass/radioaudio_ckpt_norm/cgan_ablation/')

        # vocoder configuration
        parser.add_argument('--vocoder_ckpt', dest='vocoder_ckpt',
                            help='checkpoint path to recover vocoder', default='')
        parser.add_argument('--vocoder_config', dest='vocoder_config',
                            help='vocoder configuration', default='')
        parser.add_argument('--dumpdir', dest='dumpdir', help='directory to dump feature files.',
                            default='')

        #evaluate
        parser.add_argument('--freq_eval', dest='metrics_every',
                            help='frequency to evaluate', default=10) #30
        parser.add_argument('--pth_log', dest='pth_log',
                            help='output dir to save pth file', default='/home/yujt/radio2text/checkpoint_glass/_exp/istft/'
                                        'metrics.log')
        self.parser = parser

    def parse_train_arguments(self):
        args = self.parser.parse_args()
        return args