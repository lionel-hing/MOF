import argparse
from contextlib import contextmanager
import pandas as pd
import torch
import transformers
from sacred import Experiment
from tqdm import tqdm

import data_loader.data_loader as module_data
import model.metric as module_metric
import model.model as module_arch
from model.model import sim_matrix
from parse_config import ConfigParser
from trainer.trainer import verbose
from utils.util import state_dict_data_parallel_fix
import numpy as np
import os

ex = Experiment('test')


def get_flat_w(model):
    # collect weight (module, name) pairs
    # flatten weights
    w_modules_names = []

    for m in model.modules():
        for n, p in m.named_parameters(recurse=False):
            if p is not None:
                w_modules_names.append((m, n))
        for n, b in m.named_buffers(recurse=False):
            if b is not None:
                logging.warning((
                                    '{} contains buffer {}. The buffer will be treated as '
                                    'a constant and assumed not to change during gradient '
                                    'steps. If this assumption is violated (e.g., '
                                    'BatchNorm*d\'s running_mean/var), the computation will '
                                    'be incorrect.').format(m.__class__.__name__, n))
    model._weights_module_names = tuple(w_modules_names)

    ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)

    assert len(set(w.dtype for w in ws)) == 1

    # reparam to a single flat parameter
    model._weights_numels = tuple(w.numel() for w in ws)
    model._weights_shapes = tuple(w.shape for w in ws)
    with torch.no_grad():
        flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

    # remove old parameters, assign the names as buffers
    for m, n in model._weights_module_names:
        delattr(m, n)
        m.register_buffer(n, None)

    # remove old parameters, assign the names as buffers
    # for m, n in self.model._weights_module_names:
    #     delattr(m, n)
    #     m.register_buffer(n, None)

    # register the flat one
    model.register_parameter('flat_w', torch.nn.Parameter(flat_w, requires_grad=True))
    return flat_w


@contextmanager
def unflatten_weight(model, flat_w):
    ws = (t.view(s) for (t, s) in zip(flat_w.split(model.module._weights_numels), model.module._weights_shapes))
    for (m, n), w in zip(model.module._weights_module_names, ws):
        setattr(m, n, w)
    yield
    for m, n in model.module._weights_module_names:
        setattr(m, n, None)


@ex.main
def run():
    # setup data_loader instances
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride

    data_loader = config.initialize('data_loader', module_data)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # build model architecture
    model = config.initialize('arch', module_arch)

    # register flat_w
    w_modules_names = []

    for m in model.modules():
        for n, p in m.named_parameters(recurse=False):
            if p is not None:
                w_modules_names.append((m, n))
        for n, b in m.named_buffers(recurse=False):
            if b is not None:
                logging.warning((
                                    '{} contains buffer {}. The buffer will be treated as '
                                    'a constant and assumed not to change during gradient '
                                    'steps. If this assumption is violated (e.g., '
                                    'BatchNorm*d\'s running_mean/var), the computation will '
                                    'be incorrect.').format(m.__class__.__name__, n))

    model._weights_module_names = tuple(w_modules_names)

    ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)

    assert len(set(w.dtype for w in ws)) == 1

    # reparam to a single flat parameter

    model._weights_numels = tuple(w.numel() for w in ws)
    model._weights_shapes = tuple(w.shape for w in ws)
    with torch.no_grad():
        flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

    # remove old parameters, assign the names as buffers
    for m, n in model._weights_module_names:
        delattr(m, n)
        m.register_buffer(n, None)

    # register the flat one
    model.register_parameter('flat_w', torch.nn.Parameter(flat_w, requires_grad=True))

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))

    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print('Using random weights')

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    # flat_w = get_flat_w(model)

    meta_arr = []
    text_embed_arr = []
    vid_embed_arr = []
    print(len(data_loader))
    with torch.no_grad():
        for i, data in tqdm(tqdm(enumerate(data_loader))):
            # leave this for now since not doing anything on the gpu
            meta_arr.append(data['meta'])
            if tokenizer is not None:
                data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            data['text'] = {key: val.cuda() for key, val in data['text'].items()}
            if isinstance(data['video'], list):
                data['video'] = [x.to(device) for x in data['video']]
            else:
                data['video'] = data['video'].to(device)
            with unflatten_weight(model, model.module.flat_w):
                text_embed, vid_embed = model(data['text'], data['video'])
            text_embed_arr.append(text_embed.cpu().detach())
            vid_embed_arr.append(vid_embed.cpu().detach())

    text_embeds = torch.cat(text_embed_arr)
    vid_embeds = torch.cat(vid_embed_arr)

    mask = None
    if data_loader.dataset.sliding_window_stride != -1:
        cpu_vid_embeds = vid_embeds
        cpu_text_embeds = text_embeds

        li_vid_embeds = [x for x in cpu_vid_embeds]
        li_txt_embeds = [x for x in cpu_text_embeds]
        videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
        raw_caps = pd.Series([x['raw_captions']] for x in meta_arr).explode().explode()
        vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                               'captions': raw_caps})
        new_vid_embeds = []
        new_txt_embeds = []
        for vid in vid_df['videoid'].unique():
            tdf = vid_df[vid_df['videoid'] == vid]
            tvembeds = torch.stack(tdf['vid_embed'].values.tolist())
            tvembeds = tvembeds.mean(dim=0)
            new_vid_embeds.append(tvembeds)

            for cap in tdf['captions'].unique():
                cdf = vid_df[vid_df['captions'] == cap]
                ttembeds = torch.stack(cdf['txt_embed'].values.tolist())
                new_txt_embeds.append(ttembeds[0])

        vid_embeds = torch.stack(new_vid_embeds)
        text_embeds = torch.stack(new_txt_embeds)

    if args.split != 'train':  # because train is usually too big
        sims = sim_matrix(text_embeds, vid_embeds)
        sims = sims.numpy()

        nested_metrics = {}
        for metric in metric_fns:
            metric_name = metric.__name__
            res = metric(sims, query_masks=mask)
            verbose(epoch=0, metrics=res, name="", mode=metric_name)
            nested_metrics[metric_name] = res

    # if config.config['visualizer']:
    #    raise NotImplementedError
    if args.save_feats is not None:
        vid_embeds = vid_embeds.cpu().detach().numpy()
        text_embeds = text_embeds.cpu().detach().numpy()
        vid_embeds_save_fp = os.path.join(args.save_feats, f'vid_embeds_{data_loader.dataset.split}.npy')
        txt_embeds_save_fp = os.path.join(args.save_feats, f'txt_embeds_{data_loader.dataset.split}.npy')

        np.save(vid_embeds_save_fp, vid_embeds)
        np.save(txt_embeds_save_fp, text_embeds)

        videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
        videoids.to_csv(os.path.join(args.save_feats, f'ids_{data_loader.dataset.split}.csv'), index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default=None,
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=16, type=int,
                      help='size of batch')
    config = ConfigParser(args, test=True)
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    ex.run()
