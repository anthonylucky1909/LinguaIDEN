import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, src_ids, tgt_in_ids, tgt_out_ids):
        self.src_ids = src_ids
        self.tgt_in_ids = tgt_in_ids
        self.tgt_out_ids = tgt_out_ids

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return {
            'src': torch.tensor(self.src_ids[idx], dtype=torch.long),
            'tgt_in': torch.tensor(self.tgt_in_ids[idx], dtype=torch.long),
            'tgt_out': torch.tensor(self.tgt_out_ids[idx], dtype=torch.long),
        }

def collate_fn(batch):
    src_batch = [item['src'] for item in batch]
    tgt_in_batch = [item['tgt_in'] for item in batch]
    tgt_out_batch = [item['tgt_out'] for item in batch]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_in_padded = pad_sequence(tgt_in_batch, batch_first=True, padding_value=0)
    tgt_out_padded = pad_sequence(tgt_out_batch, batch_first=True, padding_value=0)

    return {
        'src': src_padded,
        'tgt_in': tgt_in_padded,
        'tgt_out': tgt_out_padded
    }