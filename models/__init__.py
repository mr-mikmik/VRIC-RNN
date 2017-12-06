import os
import copy

import numpy as np
import torch

from FC_models import *
from Conv_models import *
from LSTM_models import *

def setup(args):
    if args.model == 'fc':
        model = CoreFC(args.coded_size, args.patch_size)
    elif args.model == 'fc_rec':
        model = RecursiveCoreFC(args.coded_size, args.patch_size, args.num_passes)
    elif args.model == 'conv':
        model = ConvolutionalCore(args.coded_size, args.patch_size)
    elif args.model == 'conv_rec':
        model = ConvolutionalRecursiveCore(args.coded_size, args.patch_size, args.num_passes)
    elif args.model == 'lstm':
        model = LSTMCore(args.coded_size, args.patch_size, args.batch_size, args.num_passes)

    else:
        raise Exception("Caption model not supported: {}".format(args.model))

    # TODO: START FROM:
    # check compatibility if training is continued from previously saved model
    #if vars(args).get('start_from', None) is not None:
    #    # check if all necessary files exist
    #    assert os.path.isdir(args.start_from), " %s must be a a path" % opt.start_from
    #    assert os.path.isfile(os.path.join(args.start_from,
    #                                       "infos_" + opt.id + ".pkl")), "infos.pkl file does not exist in path %s" % opt.start_from
    #    model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model