from pathlib import Path
import re


'''conll'''
import pandas as pd
def prepro_conll():
    conll = pd.read_csv(r'.\data\conll\train.txt', delimiter=' ', header=None, skiprows=2, skip_blank_lines=False, usecols=[0,3], names=['word', 'label'])
    word_list = conll['word'].tolist()
    tag_list = conll['label'].tolist()
    m = list(zip(word_list, tag_list))
    def split_at_nan(some_list):
        sent = []
        all_sents = []
        for t in some_list:
            if isinstance(t[0], str) and isinstance(t[1], str):
                sent.append(t)
            else:
                all_sents.append(sent)
                sent = []
        return all_sents
    m_split = split_at_nan(m)
    texts = []
    tags = []
    for i in range(len(m_split)):
        try:
            a, b = zip(*m_split[i])
            texts.append(list(a))
            tags.append(list(b))
        except:
            continue
    return texts, tags

texts, tags = prepro_conll()
texts = [t for t in texts if len(t) in [6,7,8]]
tags = [ta for ta in tags if len(ta) in [6,7,8]]
#good_index =[i for i, x in enumerate(tags) if len(set(x)) != 1 ]
good_examples = []
for i, el in enumerate(texts):
    counter = 0
    for l in el:
        if l.isalpha():
            counter += 1
    if (counter / len(el)) > 0.8:
        good_examples.append(i)

texts = [texts[i] for i in good_examples]
tags = [tags[i] for i in good_examples]

good_index =[i for i, x in enumerate(tags) if len(set(x)) != 1 ]
texts = [texts[i] for i in good_index]
tags = [tags[i] for i in good_index]

kill_o_words = [[i for i,t in enumerate(t_seq) if t!= 'O'] for t_seq in tags]
for i in range(len(texts)):
    texts[i] = [texts[i][index] for index in kill_o_words[i]]
    tags[i] = [tags[i][index] for index in kill_o_words[i]]

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2, shuffle=True)
val_texts = train_texts[:64]
val_tags = train_tags[:64]
train_texts = train_texts[:64]
train_tags = train_tags[:64]


print('len train: ',len(train_texts), 'len val: ', len(val_texts))
print(val_texts[0])
unique_tags = set(tag for doc in tags for tag in doc)
unique_tags.add('O')

print(unique_tags)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)


import numpy as np

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)

'''

-set some % of 'other' in train_labels to -100
- get label indices from 
'''
# other_id = tag2id['O']
# import random
# for seq in range(len(train_labels)):
#     shoot_out_indices = random.sample([i for i,v in enumerate(train_labels[seq]) if v == other_id], int(len([i for i,v in enumerate(train_labels[seq]) if v == other_id]) / 1.2))
#     for i in shoot_out_indices:
#         train_labels[seq][i] = -100


import tensorflow as tf

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))

vocab_switch = {y:x for x,y in tokenizer.vocab.items()}
def flatten_lol(lol):
    flatten = []
    for subl in lol:
        for item in subl:
            flatten.append(item)
    return flatten

train_labels = [[v if v!= -100 else tag2id['O'] for v in subl] for subl in train_labels ]


''' '''
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
seq_len = len(train_encodings['input_ids'][0]) #inputs.shape[1]

input_layer_id = tf.keras.Input(shape=(seq_len, ), dtype='int32', name='input_ids')
input_layer_attention_mask = tf.keras.Input(shape=(seq_len, ), dtype='int32', name='attention_mask')
bert = TFBertModel.from_pretrained('bert-base-cased')
bert_final_state = bert(inputs=[input_layer_id, input_layer_attention_mask])[0]
bert_final_state_cl = tf.keras.layers.LSTM(units=100, return_sequences=True )(bert_final_state)
bert_final_state_cl = tf.keras.layers.Dense(units= len(unique_tags))(bert_final_state_cl)
model = tf.keras.models.Model(inputs = [input_layer_id, input_layer_attention_mask], outputs = bert_final_state_cl)
model.layers[2].trainable = True
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=6e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x = [np.array(train_encodings['input_ids']), np.array(train_encodings['attention_mask'])],
          batch_size=64, y= np.array(train_labels), epochs= 1)
''' '''


from sklearn.metrics import confusion_matrix, classification_report
from pprint import  pprint
predictions = model.predict(x = [np.array(train_encodings['input_ids']), np.array(train_encodings['attention_mask'])])
predictions.shape
pred_post = [np.argmax(value)  for preds in predictions for value in preds]
label_post = [value for labels in train_labels for value in labels]
print(confusion_matrix(pred_post, label_post))
report = classification_report(pred_post, label_post)
pprint(report)



