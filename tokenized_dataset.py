import torch
from torch.utils.data import DataLoader, Dataset

# ====================================================
# Dataset
# ====================================================
"""
maskの長さだけ取ってる。正直よくわからん。
"""
def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]  # バッチ方向には全部取っている。
    return inputs

"""
encode: 文を、token_idのシーケンスに変換
encode_plus: encodeに加えて, token type id（どちらの文章にぞくしているか）と attentikon maskも返す
しかも,max_lenを指定することで、空白部分をパッディングしてくれる。
"""
def prepare_input(cfg, text, tokenizer):
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TokenizedDataset(Dataset):
    def __init__(self, cfg, df, tokenizer):
        self.cfg = cfg      
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = self.prepare_input(self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label

    def prepare_input(self, text):
        inputs = self.tokenizer.encode_plus(
            text, 
            return_tensors=None, 
            add_special_tokens=True, 
            max_length=self.cfg.max_len,
            pad_to_max_length=True,
            truncation=True
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs
    


if __name__=="__main__":        
    df = ...
    cfg = ...
    dataset = TokenizedDataset()
