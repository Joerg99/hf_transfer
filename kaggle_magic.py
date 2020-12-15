import os
import re
import json
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, TFDistilBertModel, DistilBertTokenizer


max_len = 80
#configuration = BertConfig()
data_csv = r".\data\kaggle_ner\ner_dataset.csv"

# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = "bert_base_uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
#tokenizer = DistilBertTokenizer(vocab_file="bert_base_uncased/vocab.txt", do_lower_case=True)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

def masked_ce_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, num_tags))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def create_model(num_tags):
    ## BERT encoder
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    #encoder = TFDistilBertModel.from_pretrained("bert-base-uncased")

    ## NER Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    embedding = layers.Dropout(0.3)(embedding)
    tag_logits = layers.Dense(num_tags + 1, activation='softmax')(embedding)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[tag_logits],
    )
    optimizer = keras.optimizers.Adam(lr=3e-5)

    model.compile(optimizer=optimizer, loss=masked_ce_loss, metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)])
    return model


############################
# process conll
############################

def prepro_conll():
    conll = pd.read_csv(r'.\data\conll\train.txt', delimiter=' ', header=None, skiprows=2, skip_blank_lines=False, usecols=[0,3], names=['word', 'label'])
    tag_encoder = preprocessing.LabelEncoder()
    tag_encoder.fit_transform([l for l in conll.label.unique() if isinstance(l, str)])
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
    return texts, tags, tag_encoder
texts, tags , tag_encoder = prepro_conll()

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
texts = np.array(texts)
tags = [tags[i] for i in good_examples]
tags = [tag_encoder.transform(sample) for sample in tags]
texts_train = texts[:128]
tags_train = tags[:128]
texts_eval = texts[128:192]
tags_eval = tags[128:192]

############################
def create_inputs_targets_conll(sentences, tags, tag_encoder):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "tags": []
    }
    num_tags = len(tag_encoder.classes_)

    for sentence, tag in zip(sentences, tags):
        input_ids = []
        target_tags = []
        for idx, word in enumerate(sentence):
            ids = tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(ids.ids)
            num_tokens = len(ids)
            target_tags.extend([tag[idx]] * num_tokens)

        # Pad truncate
        input_ids = input_ids[:max_len - 2]
        target_tags = target_tags[:max_len - 2]

        input_ids = [101] + input_ids + [102]
        target_tags = [tag_encoder.transform(['O'])[0]] + target_tags + [tag_encoder.transform(['O'])[0]]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        padding_len = max_len - len(input_ids)

        input_ids = input_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tags = target_tags + ([num_tags] * padding_len)

        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["token_type_ids"].append(token_type_ids)
        dataset_dict["attention_mask"].append(attention_mask)
        dataset_dict["tags"].append(target_tags)
        assert len(target_tags) == max_len, f'{len(input_ids)}, {len(target_tags)}'

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = dataset_dict["tags"]
    return x, y, tag_encoder, num_tags


x_train, y_train, tag_encoder, num_tags = create_inputs_targets_conll(texts_train, tags_train , tag_encoder)
x_eval, y_eval, _ , _ = create_inputs_targets_conll(texts_eval, tags_eval, tag_encoder)
model = create_model(num_tags)
from sklearn.metrics import confusion_matrix, classification_report
from pprint import pprint

y_eval_flat = [v for sent in list(y_eval) for v in sent]
valid_index_y_eval = [i for i, v in enumerate(y_eval_flat) if v != num_tags]
y_eval_flat = [y_eval_flat[i] for i in valid_index_y_eval]
print('uff')
####################
# Training
####################
for epoch in range(20):
    print('epoch: ', epoch)
    model.fit(x_train, y_train, verbose=1, batch_size=16, validation_split=0.1)
    #############
    #eval
    #############
    pred_test = model.predict(x_eval)

    pred_eval_flat = np.reshape(pred_test, [pred_test.shape[0] * pred_test.shape[1], pred_test.shape[2]])
    pred_eval_flat = [pred_eval_flat[i] for i in valid_index_y_eval]
    pred_eval_flat = np.argmax(pred_eval_flat, axis=1)


    print(confusion_matrix(pred_eval_flat, y_eval_flat))
    report = classification_report(pred_eval_flat, y_eval_flat)
    pprint(report)

print("pause")



def create_test_input_from_text(texts):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }
    for sentence in texts:
        input_ids = []
        for idx, word in enumerate(sentence.split()):
            ids = tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(ids.ids)
            num_tokens = len(ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        input_ids = input_ids[:max_len - 2]

        input_ids = [101] + input_ids + [102]
        n_tokens = len(input_ids)
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        padding_len = max_len - len(input_ids)

        input_ids = input_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["token_type_ids"].append(token_type_ids)
        dataset_dict["attention_mask"].append(attention_mask)

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    return x, n_tokens

###################
# eval
###################
pred_test = model.predict(x_train)

y_train_flat = [v  for sent in list(y_train) for v in sent]
valid_index_y_train = [i for i, v in enumerate(y_train_flat) if v != 9]
y_train_flat = [y_train_flat[i] for i in valid_index_y_train]

pred_test_flat = np.reshape(pred_test, [64*80, 10])
pred_test_flat = [pred_test_flat[i] for i in valid_index_y_train]
pred_test_flat = np.argmax(pred_test_flat, axis=1)


from sklearn.metrics import confusion_matrix, classification_report
from pprint import  pprint
print(confusion_matrix(pred_test_flat, y_train_flat))
report = classification_report(pred_test_flat, y_train_flat)
pprint(report)


##################
test_inputs = ["alex lives in london"]
x_test, n_tokens = create_test_input_from_text(test_inputs)
print('input tokens')
print(x_test[0][0][:n_tokens])
pred_test = model.predict(x_train)
pred_tags = np.argmax(pred_test, 2)[1][:n_tokens]  # ignore predictions of padding tokens

# create dictionary of tag and its index
le_dict = dict(zip(tag_encoder.transform(tag_encoder.classes_), tag_encoder.classes_))
print('predicted tags')
print([le_dict.get(_, '[pad]') for _ in pred_tags])


print('stay alive')