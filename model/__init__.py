from .crog import CROG
from .ssg import SSG
from loguru import logger


def build_crog(args):
    model = CROG(args)
    backbone = []
    head = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        else:
            head.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]
    return model, param_list


def build_ssg(args):
    model = SSG(args)

    return model, model.parameters()