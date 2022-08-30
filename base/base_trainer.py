from abc import abstractmethod
from data_loader.transforms import init_transform_dict
from base.base_dataset import read_frames_decord
import torch
import logging
from numpy import inf
import torch.optim as optim
import pandas as pd
import os
import transformers


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, config, writer=None, init_val=False):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.init_val = init_val
        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        self.model.device = self.device
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # flat_w
        _ = self.register_flat_w()

        self.params = self.init_params()

        for p in self.params:
            p.grad = torch.zeros_like(p)

        loss = loss.to(self.device)
        self.loss = loss
        self.metrics = metrics
        
        # self.optimizer = optimizer
        self.optimizer = config.initialize('optimizer', transformers, self.model.parameters())


        self.optimizer0 = optim.Adam(self.params, lr=0.0008, betas=(0.5, 0.999))

        self.scheduler0 = optim.lr_scheduler.StepLR(self.optimizer0, step_size=30,
                                                    gamma=0.5)

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.init_val = cfg_trainer.get('init_val', False)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        #self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.writer = writer

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    # register flat_w
    def register_flat_w(self):
        # collect weight (module, name) pairs
        # flatten weights
        w_modules_names = []

        for m in self.model.modules():
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

        self.model._weights_module_names = tuple(w_modules_names)

        ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)

        assert len(set(w.dtype for w in ws)) == 1

        # reparam to a single flat parameter

        self.model._weights_numels = tuple(w.numel() for w in ws)
        self.model._weights_shapes = tuple(w.shape for w in ws)
        with torch.no_grad():
            flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

        # remove old parameters, assign the names as buffers
        for m, n in self.model._weights_module_names:
            delattr(m, n)
            m.register_buffer(n, None)

        # register the flat one
        self.model.register_parameter('flat_w', torch.nn.Parameter(flat_w, requires_grad=True))
        return torch.nn.Parameter(flat_w, requires_grad=True)

    def get_param(self):
        return self.model.flat_w

    def init_params(self):

        dir_path = "/data/MSRVTT/"
        train_list_path = "structured-symlinks/train_list_jsfusion.txt"
        train_video_path = "videos/all"

        params = [torch.zeros(8, 3, 224, 224, requires_grad=True) for _ in range(10000)]

        # params = nn.ParameterDict()

        tsfm_dict = init_transform_dict()
        tsfm_split = "train"
        tsfm = tsfm_dict[tsfm_split]

        train_df = pd.read_csv(os.path.join(dir_path, train_list_path), names=['videoid'])
        video_ids = train_df['videoid']

        for video_id in video_ids:
            id = int(video_id[5:])
            if self.config["data_loader"]["args"]["dataset_name"] == "MSRVTT":
                video_path = os.path.join(dir_path, train_video_path, video_id + ".mp4")
            elif self.config["data_loader"]["args"]["dataset_name"] == "MSVD":
                video_path = os.path.join(dir_path, train_video_path, video_id + ".avi")
            
            frames, _ = read_frames_decord(video_path=video_path, num_frames=8, sample='uniform', fix_start=None)

            frames = tsfm(frames)

            frames_torch = torch.ones([8, 3, 224, 224])
            frames_torch[:frames.shape[0]] = frames

            frames_param = frames_torch
            params[id] = frames_param

        return params

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError


    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0

        if self.init_val:
            _ = self._valid_epoch(-1)

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'nested_val_metrics':
                    # NOTE: currently only supports two layers of nesting
                    for subkey, subval in value.items():
                        for subsubkey, subsubval in subval.items():
                            for subsubsubkey, subsubsubval in subsubval.items():
                                log[f"val_{subkey}_{subsubkey}_{subsubsubkey}"] = subsubsubval
                else:
                    log[key] = value

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0 or best:
            #if best:
                self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        #list_ids = list(range(n_gpu_use))
        list_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")

        state_dict = checkpoint['state_dict']

        load_state_dict_keys = list(state_dict.keys())
        curr_state_dict_keys = list(self.model.state_dict().keys())
        redo_dp = False
        if not curr_state_dict_keys[0].startswith('module.') and load_state_dict_keys[0].startswith('module.'):
            undo_dp = True
        elif curr_state_dict_keys[0].startswith('module.') and not load_state_dict_keys[0].startswith('module.'):
            redo_dp = True
            undo_dp = False
        else:
            undo_dp = False

        if undo_dp:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
        elif redo_dp:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k  # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict

        self.model.load_state_dict(new_state_dict)

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
