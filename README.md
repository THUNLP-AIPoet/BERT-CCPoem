## BERT-CCPoem

### Introduction

**BERT-CCPoem** is an BERT-based pre-trained model particularly for Chinese classical poetry, developed by Research Center for Natural Language Processing, Computational Humanities and Social Sciences, Tsinghua University (清华⼤学⼈⼯智能研究院⾃然语⾔处理与社会⼈⽂计算研究中⼼).

BERT-CCPoem is trained on a (almost) full collection of Chinese classical poems, CCPC-Full v1.0, consisting of 926,024 classical poems with 8,933,162 sentences. Basically, it can provide the vector (embedding) representation of any sentence in any Chinese classical poem, and thus be used in various downstream applications including intelligent poetry retrieval, recommendation and sentiment analysis.

A typical application is, you can use vector representation derived from BERT-CCPoem to get the most semantically similar sentences of a given sentence, in terms of the related cosine values. For example, provided a poem sentence "一行白鹭上青天", the top 10 most likely sentences given by BERT-CCPoem are as follows:. 

|Rank|	Poem sentence	|Cosine similarity| Rank |Poem sentence|Cosine similarity|
| ------------ | ------|--------- |  -------- | -------- | -------- |
|1	|白鹭一行登碧霄|	0.9331|	6|	一行白鸟掠清波|	0.9024|
|2|	一片青天白鹭前|	0.9185|	7|	时向青空飞白鹭|	0.9023|
|3	|飞却青天白鹭鸶|	0.9155|	8|	一行飞鸟来青天|	0.9005|
|4	|一双白鹭上云飞	|0.9118|	9	|一行白鹭下汀洲|	0.8994|
|5	|白鹭一行飞绿野|	0.9065|	10|	一行飞鹭下汀洲|	0.8962|

The following is the top 10 mostly likely sentences given by the string matching algorithm, for comparison:

|Rank	|Poem sentence|	Rank	|Poem sentence|
| ------------ | ------|--------- |  -------- | 
|1|	数行白鹭横青湖 | 6	|一行白鹭渺秋烟|
|2	|一片青天白鹭前|	7|	一行白鹭引舟行|
|3|	一行飞鸟来青天|	8	|一行白鹭过前山|
|4	|一行白鹭下汀洲	|9|	一行白雁遥天暮|
|5|	一行白鹭云间绕	|10|	一行白雁天边字|

### Model details

We use "BertModel" class in the open source project Transformers to train our model. BERT-CCPoem is fully based on CCPC1.0, and takes Chinese character as basic unit. Characters with frequency less than 3 is treated as [UNK], resulting in a vocabulary of 11, 809 character types.

The parameters of BERT-CCPoem are listed as follows: 

| model       | version  |  parameters  | vocab_size | model_size | download_url |
| ------------ | ------|--------- |  -------- | -------- | -------- |
| BERT-CCPoem | v1.0 | 8-layer, 512-hidden, 8-heads   | 11809 |  162MB | [download](https://thunlp.oss-cn-qingdao.aliyuncs.com/BERT_CCPoem_v1.zip) |


### How to use

* Download Bert-CCPoem v1.0:

```
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/BERT_CCPoem_v1.zip
unzip BERT_CCPoem_v1.zip
```

* Then, load BERT-CCPoem v1.0 with the specified path. For example, to generate the vector  representation of the sentence "一行白鹭上青天":

```
from transformers import BertModel, BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained('./BERT_CCPoem_v1') 
model = BertModel.from_pretrained('./BERT_CCPoem_v1')
input_ids = torch.tensor(tokenizer.encode("一行白鹭上青天")).unsqueeze(0) 
outputs, _ = model(input_ids)
sen_emb = torch.mean(outputs, 1)[0] # This is the vector representation of "一行白鹭上青天"
```

**Note** You may check out the sample programs [gen\_vec\_rep.py](https://github.com/THUNLP-AIPoet/BERT-CCPoem/gen_vec_rep.py) we offer.

#### Requirement.txt

```
torch>=1.2.0
transformers>=2.5.1
```

### Acknowledging and Citing BERT-CCPoem

We makes BERT-CCPoem available to research free of charge provided the proper reference is made using an appropriate citation.

When writing a paper or producing a software application, tool, or interface based on BERT-CCPoem, it is necessary to properly acknowledge using BERT-CCPoem as **“We use BERT-CCPoem, a pre-trained model for Chinese classical poetry, developed by Research Center for Natural Language Processing, Computational Humanities and Social Sciences, Tsinghua University, to ……”** and cite the GitHub website "https://github.com/THUNLP-AIPoet/BERT-CCPoem". 


### Contributors

**Professor:** Maosong Sun

**Students:** Zhipeng Guo, Jinyi Hu

### Contact Us

If you have any questions, suggestions or bug reports, please feel free to email hujy369@gmail.com or gzp9595@gmail.com.

