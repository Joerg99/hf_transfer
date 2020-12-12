'''
follows tutorial from https://huggingface.co/transformers/custom_datasets.html#token-classification-with-w-nut-emerging-entities
poor performance in general
probably because of class imbalance? Mostly words are of class 'O'

Boosts performance:
    - set trasformer to trainable
    - mask some of the 'O' words with -100
    with: removal of num_all_O / 1.3

    - performance without shoot out of 'O' and only B-Ent laebel
    [[   0    0]
     [ 358 1067]]
    and with shoot out:
    [[777 179]
     [260 167]]

General....
    - hard to evaluate on training data. word pieces and masking is annoying
    - model.compile(metric=something) does not work out of the box
To do:
    - final dense layer is... very simple -> replace with bi-lstm or such
    - data very simplified. should add more labels

'''


from pathlib import Path
import re
'''
WNUT PREPRO
def read_wnut(file_path):
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)
    return token_docs, tag_docs

texts, tags = read_wnut(r'.\data\wnut17train.conll')

good_examples = []
for i, el in enumerate(tags):
    counter = 0
    for l in el:
        if l != 'O':
            counter += 1
    if (counter / len(el)) > 0.1:
        good_examples.append(i)

texts = [texts[i] for i in good_examples]
tags = [tags[i] for i in good_examples]

tags = [['B-ent' if t.startswith('B') else t for t in sent]  for sent in tags  ]
tags = [['I-ent' if t.startswith('I') else t for t in sent]  for sent in tags  ]

'''


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

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2, shuffle=True)
val_texts = train_texts[:64]
val_tags = train_tags[:64]
train_texts = train_texts[:64]
train_tags = train_tags[:64]

print('len train: ',len(train_texts), 'len val: ', len(val_texts))
print(val_texts[0])
unique_tags = set(tag for doc in tags for tag in doc)
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
#     shoot_out_indices = random.sample([i for i,v in enumerate(train_labels[seq]) if v == other_id], int(len([i for i,v in enumerate(train_labels[seq]) if v == other_id]) / 1.1))
#     for i in shoot_out_indices:
#         train_labels[seq][i] = -100


import tensorflow as tf

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")

vocab_switch = {y:x for x,y in tokenizer.vocab.items()}


train_encodings.keys()

def id_to_label(id):
    if id != -100:
        return id2tag[id]
    else:
        return -100
# i = 58
# for j in range(len(train_encodings['input_ids'][i])):
#     print(train_encodings['input_ids'][i][j],vocab_switch[train_encodings['input_ids'][i][j]],
#           train_encodings['token_type_ids'][i][j],
#           train_encodings['attention_mask'][i][j],
#           train_labels[i][j], id_to_label(train_labels[i][j]))


train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
'''
texts, tags --> val_texts --> val_tags, val_encodings --> val_encodings, val_labels --> val_dataset
texts, tags: texts as list simple tokenized, 
val_texts: split to train/val
val_encodings: output from distilbert tokenizer. 
            padded to max length, 
            cls token in the beginning, 
            dict with input ids and attention mask,
            words represented by IDs,
            split to word pieces 
            vocab_switch = {y:x for x,y in tokenizer.vocab.items()}
            [vocab_switch[i] for i in val_encodings['input_ids'][0]]

val_tags: Labels per word (B-Entity, O ...) , before word piece toenkizing
val_labels: after encode_tags, -100, 1,2,3 labels  for word  pieces. -100 is the ignore label(for CLS, PAD and word piece  
'''
def flatten_lol(lol):
    flatten = []
    for subl in lol:
        for item in subl:
            flatten.append(item)
    return flatten

from transformers import TFBertForTokenClassification

model = TFBertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_tags)) #, output_hidden_states=True
#model.training=True
model.layers[0].trainable = True
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# model.classifier = tf.keras.layers.LSTM(len(unique_tags), return_sequences=True)
model.compile(optimizer=optimizer, loss=model.compute_loss) #model.compute_loss) # can also use any keras loss fn tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.summary()



from sklearn.metrics import confusion_matrix, classification_report
from pprint import  pprint
import pickle
val_labels = flatten_lol(val_labels)
evaluation_history = []
train_labels_4_eval = flatten_lol(train_labels)
#for epoch in range(8):
#print('epoch: ',epoch)

model.fit(train_dataset.shuffle(64).batch(16), batch_size=16 , verbose = 1, epochs=10)  #, class_weight={0:5, 1:5,2:1})
predictions = model.predict(train_dataset)

good_indexes = [i for i, l in enumerate(train_labels_4_eval) if l != -100]

label_post = [train_labels_4_eval[j] for j in good_indexes]
list_preds = []
for logi in predictions['logits']:
    list_preds.append(np.argmax(logi))
pred_post = [list_preds[j] for j in good_indexes]
print(confusion_matrix(pred_post, label_post))
report = classification_report(pred_post, label_post)
pprint(report)

pickle.dump(evaluation_history , open('no_shoot_out_all_layers_train_lr_5e5.pickle', 'wb'))
print('staying alive')





''' SAVE:  predict and validate on example (split output to length of max_sequence)
   model.fit(train_dataset.shuffle(128).batch(16), batch_size=16 , verbose = 1)
    predictions = model.predict(val_dataset)
    pred_split = np.array_split(predictions['logits'], len(val_texts))
    label_post = []
    pred_post = []
    for i in range(len(pred_split)):
        good_indexes = [i for i, l in enumerate(val_labels[i]) if l != -100]
        label_post.append([val_labels[i] for j in good_indexes]) #[id2tag[v] if v != -100 else 'O' for v in np.array(val_labels[i][:np.sum(val_encodings['attention_mask'][i])])])
        list_preds = []
        for logi in pred_split[i]:
            list_preds.append(np.argmax(logi))
        pred_post.append([list_preds[i] for j in good_indexes]) #[ id2tag[v] for v in np.array(list_preds[:np.sum(val_encodings['attention_mask'][i])])])

print(classification_report([2*[id2tag[v] if v != -100 else 'O' for v in np.array(val_labels[i][:np.sum(val_encodings['attention_mask'][i])])]],
                      [2*[ id2tag[v] for v in np.array(list_preds[:np.sum(val_encodings['attention_mask'][i])])]]))

'''








'''
from transformers import TFTrainer, TFTrainingArguments

training_args = TFTrainingArguments(
    output_dir=r'.\results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=r'.\logs',            # directory for storing logs
    logging_steps=10,
)

#with training_args.strategy.scope():
#    model = model

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
'''


'''
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
input_ids = inputs["input_ids"]
inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1

inputs['labels']

outputs = model(inputs)
model.summary()
model.predict(input_ids)
loss = outputs.loss
logits = outputs.logits
'''