import os
import random
from abc import abstractmethod

from decimal import Decimal, getcontext

import av
import cv2
import decord
import numpy as np
import torch
import math
import operator
from scipy.signal import argrelextrema
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms

getcontext().prec = 50

class TextVideoDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='decord'
                 ):
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.video_params = video_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.label_type = 'caption'
        self._load_metadata()
        if self.sliding_window_stride != -1:
            if self.split != 'test':
                raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            self._fix_temporal_samples()

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError("Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError("Get video path function must be implemented by subclass")

    def _get_caption(self, sample):
        raise NotImplementedError("Get caption function must be implemented by subclass")

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata['vlen'] = self._get_video_lens()
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0, stop=x, num=min(x, self.video_params['num_frames']) + 1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode('fix_start')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_id = sample.name
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get('loading', 'strict')
        frame_sample = 'rand'
        num_frames = self.video_params['num_frames']
        fix_start = None
        if self.split == 'val':
            num_frames = 8
        if self.split == 'test':
            frame_sample = 'uniform'
            num_frames = 8
        if self.sliding_window_stride != -1:
            fix_start = sample['fix_start']

        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_fp, num_frames, frame_sample, fix_start=fix_start)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        final = torch.zeros([num_frames, 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'video_id': video_id, 'text': caption, 'meta': meta_arr}
        return data


class TextImageDataset(TextVideoDataset):

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get('loading', 'strict')

        try:
            img = Image.open(video_fp).convert("RGB")
        except:
            if video_loading == 'strict':
                raise ValueError(f'Image loading failed for {video_fp}, image loading for this dataset is strict.')
            else:
                img = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))

        # convert to tensor because video transforms don't, expand such that its a 1-frame video.
        img = transforms.ToTensor()(img).unsqueeze(0)
        if self.transforms is not None:
            img = self.transforms(img)
        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': img, 'text': caption, 'meta': meta_arr}
        return data


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen-1, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1]))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs

def information_entropy(img):

    prob = np.zeros(256, )

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ind = img[i][j]
            prob[ind] += 1
    prob = prob / (img.shape[0] * img.shape[1])


    res = 0
    for i in range(prob.shape[0]):
        if prob[i] != 0:
            res -= prob[i] * math.log2(prob[i])
    return res


def rel_change(a, b):
    return (b - a) / max(a, b)


class Frame:
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def smooth(data, window_len=13, window='hanning'):
    s = np.r_[2 * data[0] - data[window_len:1:-1], data, 2 * data[-1] - data[-1:-window_len:-1]]

    if window == 'flat':
        win = np.ones(window_len, 'd')
    elif window == 'hanning':
        win = getattr(np, window)(window_len)

    y = np.convolve(win / win.sum(), s, mode='same')
    return y[window_len - 1: -window_len + 1]


def hist_extraction(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())

    ind = 0
    curr_frame, prev_frame = None, None
    frame_hist = []
    frames = []
    success, frame = cap.read()
    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_frame = gray

        if curr_frame is not None and prev_frame is not None:
            H1 = cv2.calcHist([curr_frame], [0], None, [256], [0, 255])
            H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)

            H2 = cv2.calcHist([prev_frame], [0], None, [256], [0, 255])
            H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)

            # KL散度比较
            similarity = cv2.compareHist(H1, H2, 5)

            frame_hist.append(similarity)
            frame = Frame(ind, similarity)
            frames.append(frame)

        prev_frame = curr_frame
        ind += 1
        success, frame = cap.read()

    cap.release()
    frames.sort(key=operator.attrgetter("diff"), reverse=True)
    keyframe_id_list = []
    for i in range(num_frames):
        keyframe_id_list.append(frames[i].id)

    return keyframe_id_list



def diff_exaction(video_path, num_frames, use_thresh=True, thresh=0.7):

    cap = cv2.VideoCapture(video_path)

    ind = 0
    curr_frame, prev_frame = None, None
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    while (success):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv

        if curr_frame is not None and prev_frame is not None:
            diff = cv2.absdiff(curr_frame, prev_frame)  
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(ind, diff_sum_mean)
            frames.append(frame)

        prev_frame = curr_frame
        ind += 1
        success, frame = cap.read()

    cap.release()
    #vlen = len(frames)
    #frame_idxs = sample_frames(num_frames, vlen, sample='rand', fix_start=None)
    keyframe_id_set = set()

    #if use_top_order:
        # sort the list in descending order
    #    frames.sort(key=operator.attrgetter("value"), reverse=True)
    #    frames = frames[:num_frames]
    if use_thresh:
        for i in range(1, len(frames)):
            a1 = np.float(frames[i - 1].diff)
            a2 = np.float(frames[i].diff)
            if rel_change(Decimal(a1), Decimal(a2)) >= thresh:
                keyframe_id_set.add(frames[i].id)
    keyframe_id_list = list(keyframe_id_set)
    keyframe_id_list.sort()

    frames_ids = random.sample(keyframe_id_list, num_frames)

    #if use_local_maximal:
    #    diff_array = np.array(frame_diffs)
    #    sm_diff_array = smooth(diff_array, len_window)
    #    frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]

    #    for i in frame_indexes:
    #        keyframe_id_set.add(frames[i - 1].id)

    return frames_ids

def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    #vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    #frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frame_idxs = hist_extraction(video_path, num_frames)
    #frame_idxs = diff_exaction(video_path, num_frames, use_thresh=True, thresh=0.5)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
    #random.shuffle(frames)
    frames = torch.stack(frames).float() / 255
    #idss = torch.randperm(frames.shape[0])
    # frames = frames[idss, :, :, :].view(frames.size())
    #random.shuffle(success_idxs)
    cap.release()
    return frames, success_idxs


def read_frames_av(video_path, num_frames, sample='rand', fix_start=None):
    reader = av.open(video_path)
    try:
        frames = []
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    except (RuntimeError, ZeroDivisionError) as exception:
        print('{}: WEBM reader cannot open {}. Empty '
              'list returned.'.format(type(exception).__name__, video_path))
    vlen = len(frames)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


decord.bridge.set_bridge("torch")


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen


video_reader = {
    'av': read_frames_av,
    'cv2': read_frames_cv2,
    'decord': read_frames_decord
}
