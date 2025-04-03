from config import *
from data_process import get_vocab
from transformer import make_model
from torch.nn.utils.rnn import pad_sequence
from predict import model_predict
from dataset import get_padding_mask

def Poety_predict(model, src_x, src_mask, max_len=50):
    model = model.module if MULTI_GPU else model
    id_to_char = get_vocab(vocab_path)['id_to_char']

    # 初始化目标值
    prob_x = torch.tensor([[sos_id]] * src_x.size(0))
    prob_x = prob_x.to(device)

    for _ in range(max_len):
        prob_mask = get_padding_mask(prob_x, pad_id)
        output = model.predict(src_x, src_mask, prob_x, prob_mask)   # [32, seq, d_model]
        predict = torch.argmax(output, dim=-1, keepdim=True)  # 贪婪搜索
        prob_x = torch.concat([prob_x, predict], dim=-1)
        # 全部预测结束，结束循环
        if torch.all(predict == eos_id).item():
            break
    # 根据预测值id，解析翻译后的句子
    batch_prob_text = []
    for prob in prob_x:
        prob_text = []
        for prob_id in prob:
            if prob_id == sos_id:
                continue
            if prob_id == eos_id:
                break
            prob_text.append(id_to_char[prob_id.item()])
        batch_prob_text.append(''.join(prob_text))
    return batch_prob_text

def generate_poetry(model, input_text, vocab, max_length=50):
    model = model.module if MULTI_GPU else model
    model.eval()
    input_seq = torch.tensor([[sos_id] + [vocab.get(c, unk_id) for c in input_text]])
    input_seq = input_seq.to(device)  # 转换成ID
    generated = input_seq[:, 1:]

    for _ in range(max_length):
        src_pad_mask = get_padding_mask(input_seq, padding_idx=0).to(device)
        output = model.predict(input_seq, src_pad_mask)
        next_word_id = output.argmax(dim=-1).item()

        if next_word_id == eos_id:
            break
        input_seq = torch.cat([input_seq, torch.tensor([[next_word_id]]).to(device)], dim=1)
        generated = input_seq[:, 1:]
    result = "".join([list(vocab.keys())[list(vocab.values()).index(idx)] for idx in generated[0].tolist()])
    return result



def generation(text=None):
    char_to_id = get_vocab(vocab_path)['char_to_id']

    SRC_VOCAB_SIZE = len(char_to_id)

    model = make_model(SRC_VOCAB_SIZE, d_model, n_head, d_ff, N, dropout)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    text = input("原文:")
    while text != "exit":
        prob_sent = generate_poetry(model, text, char_to_id, max_len)
        print("译文:", prob_sent)
        text = input("生成:")

    print("已退出手动翻译模式!")


if __name__ == '__main__':
    generation()
