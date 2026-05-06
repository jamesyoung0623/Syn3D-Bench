from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--feature_dim', type=int, default=512,
                            help='input feature dimension for encoder')
        return parser
