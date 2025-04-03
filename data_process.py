from datasets import load_from_disk
from config import *
import json

def make_vocab(data_path, vocab_path):
    # 打开文件并读取数据
    id2char = ['<pad>', '<unk>', '<sos>', '<eos>']
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 制作词汇表
            for char in line.strip():  # 去掉每行的换行符后逐字符处理
                if char not in id2char:  # 避免重复添加字符
                    id2char.append(char)

    # 构建字符到 ID 的映射
    char2id = {char: idx for idx, char in enumerate(id2char)}
    vocab = {'char_to_id': char2id, 'id_to_char': id2char}
    # 保存为 JSON 文件
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)  # 确保非 ASCII 字符正确保存

def get_train_data(data_path):
    poetry = []
    # 打开文件并读取数据
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            poetry.append(line.strip())  # 去掉换行符并存储每一行
    return poetry

def get_vocab(vocab_path):
    # 加载词典
    with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
    return vocab

if __name__ == '__main__':
    make_vocab(train_path, vocab_path)
    id_to_char = get_vocab(vocab_path)['id_to_char']
    print(id_to_char[0])
