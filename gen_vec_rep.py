import os
import logging

import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer

logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)

gpu_list = None


class Bert(nn.Module):
    def __init__(self, BERT_PATH='./BERT_CCPoem_v1'):
        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained(BERT_PATH)

    def init_multi_gpu(self, device):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, cls=False):
        result = []
        # print(data)
        x = data['input_ids']
        y = self.bert(x, attention_mask=data['attention_mask'],
                         token_type_ids=data['token_type_ids'])[0]
        
        if(cls):
            result = y[:, 0, :].view(y.size(0), -1)
            result = result.cpu().tolist()
        else:
            result = []
            y = y.cpu()
            # y = torch.mean(y, 1)
            # result = y.cpu().tolist()
            for i in range(y.shape[0]):
                tmp = y[i][1:torch.sum(data['attention_mask'][i]) - 1, :]
                result.append(tmp.mean(0).tolist())

        return result


class BertFormatter():
    def __init__(self, BERT_PATH='./BERT_CCPoem_v1'):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    def process(self, data):
        res_dict = self.tokenizer.batch_encode_plus(
            data, pad_to_max_length=True)

        input_list = {'input_ids': torch.LongTensor(res_dict['input_ids']),
                      'attention_mask': torch.LongTensor(res_dict['attention_mask']),
                      "token_type_ids": torch.LongTensor(res_dict['token_type_ids'])}
        return input_list


def init(BERT_PATH="./BERT_CCPoem_v1"):
    global gpu_list
    gpu_list = []

    device_list = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if(device_list[0] == ""):
        device_list = []
    for a in range(0, len(device_list)):
        gpu_list.append(int(a))

    cuda = torch.cuda.is_available()
    logging.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logging.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    model = Bert(BERT_PATH)
    formatter = BertFormatter(BERT_PATH)
    if len(gpu_list) > 0:
        model = model.cuda()
    if(len(gpu_list) > 1):
        try:
            model.init_multi_gpu(gpu_list)
        except Exception as e:
            logging.warning(
                "No init_multi_gpu implemented in the model, use single gpu instead. {}".format(str(e)))
    return model, formatter


def predict_vec_rep(data, model, formatter):
    data = formatter.process(data)
    model.eval()

    for i in data:
        if(isinstance(data[i], torch.Tensor)):
            if len(gpu_list) > 0:
                data[i] = data[i].cuda()

    result = model(data)

    return result


def cos_sim(vector_a, vector_b, sim=True):

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    if(not sim):
        return cos
    sim = 0.5 + 0.5 * cos
    return sim


if __name__ == '__main__':
    model, formatter = init()
    result = predict_vec_rep(["一行白鹭上青天"], model, formatter)[0]
    print(result)

if __name__ == '__main__1':
    model, formatter = init()
    result_0 = predict_vec_rep(["一行白鹭上青天"], model, formatter)[0]
    result_1 = predict_vec_rep(['白鹭一行登碧霄'], model, formatter)[0]
    result_2 = predict_vec_rep(["飞却青天白鹭鸶"], model, formatter)[0]

    print(cos_sim(result_0, result_1))
    print(cos_sim(result_0, result_2))
