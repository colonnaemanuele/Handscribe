import os
import sys
import warnings
import numpy as np
import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import cv2
import pickle
import glob
import six
import time
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import video_augmentation


sys.path.append("..")
global kernel_sizes 

class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_free=True, gloss_dict=None, dataset='phoenix2014-T', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.gloss_free = gloss_free
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes 
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"handscribe/preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        print(mode, len(self))
        self.data_aug = self.transform()
        if self.gloss_free:
            vocab_file = f"handscribe/preprocess/{dataset}/vocab.pkl"
            if os.path.exists(vocab_file):
                with open(vocab_file, 'rb') as f:
                    self.word2idx = pickle.load(f)
            else:
                all_sentences = [value['label'] for value in self.inputs_list.values() if isinstance(value, dict) and 'label' in value]
                self.word2idx = self.build_vocab(all_sentences)
                with open(vocab_file, 'wb') as f:
                    pickle.dump(self.word2idx, f)
        print("")

    def __getitem__(self, idx):
        if self.data_type == "video":
            input_data, label, fi = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), fi['original_info']
        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info']
        
    def build_vocab(self, sentences):
        vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        idx = 4
        for sentence in sentences:
            for word in sentence.strip().split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab

    def tokenize(self, sentece):
        tokens = sentece.strip().split()
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in tokens]
        indices = [self.word2idx['<SOS>']] + indices + [self.word2idx['<EOS>']]
        return indices
        
    
    def read_video(self, index):
        # load file info
        fi = self.inputs_list[index]
        if 'phoenix' in self.dataset:
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])  
        elif self.dataset == 'CSL':
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'] + "/*.jpg")
        elif self.dataset == 'CSL-Daily':
            img_folder = os.path.join(self.prefix, fi['folder'])
        elif 'lis' in self.dataset:
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'] + "/*.png")
        img_list = sorted(glob.glob(img_folder))
        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        label_list = []
        if not self.gloss_free:
            for phase in fi['label'].split(" "):
                if phase == '':
                    continue
                if phase in self.dict.keys():
                    label_list.append(self.dict[phase][0])
        else:
            label_list = self.tokenize(fi['label']) # sentence tokenization
        if self.dataset != 'CSL-Daily':
            return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi
        else:
            return [cv2.cvtColor(cv2.resize(cv2.imread(img_path)[40:, ...], (256, 256)), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        # video = video.float() / 127.5 - 1
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        video = ((video.float() / 255.) - 0.45) / 0.225
        
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        
        left_pad, last_stride, total_stride = 0, 1, 1
        global kernel_sizes 
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride 
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2*left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            max_label_len = max(label_length)
            batch_size = len(label)
            
            # Convert all labels to LongTensor if they're not already
            label = [torch.LongTensor(lab) if not isinstance(lab, torch.Tensor) else lab.long() for lab in label]
            
            # Create a tensor filled with padding value (0)
            padded_labels = torch.zeros(batch_size, max_label_len, dtype=torch.long)
            
            # Fill in the actual values
            for i, lab in enumerate(label):
                padded_labels[i, :len(lab)] = lab
            return padded_video, video_length, padded_labels, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder(
        prefix='handscribe/dataset/phoenix-2014/phoenix2014-release/phoenix-2014-multisigner',
        dataset='phoenix2014T',
        datatype='video',
        kernel_size = ['K5','P2','K5','P2']
        )
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=feeder.collate_fn,
    )
    for data in dataloader:
        video, label, length = data
        print(video.shape, label.shape)
        break
