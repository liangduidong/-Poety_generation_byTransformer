from transformer import get_padding_mask, get_subsequent_mask
from torch.nn.utils.rnn import pad_sequence
from config import *
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, train_data, vocab):
        self.data = train_data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]

        # id化并添加特殊符号
        src_id = [sos_id] + [self.vocab.get(i, unk_id) for i in text] + [eos_id]
        return src_id

    def collate_fn(self, batch):
        srcs = batch
        data = pad_sequence([torch.LongTensor(src) for src in srcs], True, pad_id)  # 编码器输入

        # 获得原输入与掩码
        src_x = data[:, :-1]   # 编码器输入
        src_pad_mask = get_padding_mask(src_x, pad_id)
        src_sub_mask = get_subsequent_mask(src_x.size(1))
        src_mask = src_sub_mask | src_pad_mask

        # 获得目的输入与其掩码
        tgt_y = data[:, 1:]   # 输出目标

        return src_x, src_mask, tgt_y


