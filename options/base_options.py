import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # ---------------- runtime ----------------
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids: e.g. "0" or "0,1". use -1 for CPU')
        parser.add_argument('--name', type=str, default='HSIC_grid',
                            help='experiment name; training code may append suffixes')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='models are saved here')

        # ------------- data / sequence -----------
        parser.add_argument('--real_datasets', type=str, default='',
                            help='comma-separated list of real dataset feature folders (no 0_real/1_fake)')
        parser.add_argument('--fake_datasets', type=str, default='',
                            help='comma-separated list of fake dataset feature folders (no 0_real/1_fake)')
        parser.add_argument('--train_ids', type=str, default='train_ids.txt',
                            help='path to train ids file (cat<TAB>idx per line)')
        parser.add_argument('--val_ids', type=str, default='val_ids.txt',
                            help='path to val ids file (cat<TAB>idx per line)')
        parser.add_argument('--test_ids', type=str, default='test_ids.txt',
                            help='path to test ids file (cat<TAB>idx per line)')
        parser.add_argument('--pc_root', type=str, default='datasets/PCs',
                            help='root folder for PC datasets')
        parser.add_argument('--pc_variant', type=str, default='ULIP-1',
                            help='PC model variant (e.g., ULIP-1, ULIP-2)')
        parser.add_argument('--pc_backbone', type=str, default='pointbert',
                            help='PC backbone (e.g., pointbert, pointnext, pointmlp, pointnet2_ssg, pointbert_xyz)')
        parser.add_argument('--unpaired_datasets', action='store_true',
                            help='do not require matching ids between real and fake datasets; sample independently')
        parser.add_argument('--train_real_datasets', type=str, default='',
                            help='comma-separated real dataset names (or full paths) for training')
        parser.add_argument('--train_fake_datasets', type=str, default='',
                            help='comma-separated fake dataset names (or full paths) for training')
        parser.add_argument('--test_real_datasets', type=str, default='',
                            help='comma-separated real dataset names (or full paths) for testing (optional)')
        parser.add_argument('--test_fake_datasets', type=str, default='',
                            help='comma-separated fake dataset names (or full paths) for testing (optional)')

        # ------------- grid / single-run (legacy) -
        parser.add_argument('--lambda_x_list', type=str,
                            default='0,100,200,300,400,500,600,700,800,900,1000',
                            help='comma-separated list for lambda_x grid (legacy)')
        parser.add_argument('--lambda_y_list', type=str,
                            default='0,100,200,300,400,500,600,700,800,900,1000',
                            help='comma-separated list for lambda_y grid (legacy)')

        # Single-run values used by CL trainer as GLOBAL defaults
        parser.add_argument('--lambda_x', type=int, default=500,
                            help='HSIC-Bottleneck λ_x for current task')
        parser.add_argument('--lambda_y', type=int, default=300,
                            help='HSIC-Bottleneck λ_y for current task')

        # ------------- run mode -----------------
        parser.add_argument('--eval_only', action='store_true',
                            help='skip training and only run evaluation on existing checkpoints')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = '' if v == default else f'\t[default: {default}]'
            message += f'{k:>25}: {str(v):<30}{comment}\n'
        message += '----------------- End -------------------'
        print(message)

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        with open(os.path.join(expr_dir, 'opt.txt'), 'wt') as f:
            f.write(message + '\n')

    def parse(self, print_options=True):
        opt = self.gather_options()

        # normalize sequence into a list when present
        if hasattr(opt, "sequence") and isinstance(opt.sequence, str):
            opt.sequence = [s for s in opt.sequence.split(',') if s]

        if print_options:
            self.print_options(opt)

        # set gpu ids
        ids = [s.strip() for s in opt.gpu_ids.split(',') if s.strip() != '']
        opt.gpu_ids = []
        for sid in ids:
            i = int(sid)
            if i >= 0:
                opt.gpu_ids.append(i)
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
