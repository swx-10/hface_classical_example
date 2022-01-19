import os
import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, BertConfig, TFAutoModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split

checkpoint = 'hfl/chinese-xlnet-base'

# 模型对应的tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 训练参数
num_epochs = 1
batch_size = 32

# 读取语料
corpus_dir = os.path.join(os.path.dirname(__file__), '..','corpus','ASAP_SENT')
train_file = os.path.join(corpus_dir, 'train.tsv')
dev_file = os.path.join(corpus_dir, 'dev.tsv')

df_train = pd.read_csv(train_file, sep='\t')
df_dev = pd.read_csv(dev_file,sep='\t',index_col='qid')
# 合并数据集
df_train = pd.concat((df_train,df_dev))
# 提取训练数据
X,y = df_train['text_a'].values, df_train['star'].values
# 拆分数据集
train_texts, val_texts, train_labels, val_labels =  train_test_split(X,y, test_size=.2)
# 文本转换为训练用语料
train_batch = tokenizer(list(train_texts), padding=True,return_tensors='np')
valid_batch = tokenizer(list(val_texts), padding=True, return_tensors='np')

num_train_steps = (len(train_batch['input_ids']) // batch_size) * num_epochs
# 学习率退火
lr_scheduler = PolynomialDecay(
    initial_learning_rate=5e-5,
    end_learning_rate=0.,
    decay_steps=num_train_steps
    )

# 加载文本分类推理模型
bert_config = BertConfig.from_pretrained(checkpoint, num_labels=5) # 分类类别为5
model = TFAutoModelForSequenceClassification.from_pretrained(
    checkpoint, 
    config=bert_config
    )
# 调整是否训练整个bert模型
model.get_layer('bert').trainable=False
# 模型编译
model.compile(
    optimizer=Adam(learning_rate=lr_scheduler),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.summary()
# 训练
model.fit(
    train_batch.data, 
    np.array(train_labels-1), 
    validation_data=valid_batch,
    epochs=num_epochs, 
    batch_size=batch_size)
# 保存
model_file = os.path.join(os.path.dirname(__file__), 'models', 'best_model.h5')
model.save(model_file)