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
from transformers import BertTokenizer, TFBertModel, BertConfig


max_len = 80
configuration = BertConfig()
data_csv = r".\data\kaggle_ner\ner_dataset.csv"

# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = "bert_base_uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)

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
    model.compile(optimizer=optimizer, loss=masked_ce_loss, metrics=['accuracy'])
    return model


def process_csv(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
    enc_tag = preprocessing.LabelEncoder()
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])
    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, tag, enc_tag



############################
# process conll
############################
# sentences = np array, shape(47959,), array([list(['Thousands', 'blah']), list(['word', 'word',...]) ])
# tags = wie sentences     array([list([16, 16, 7, 16, 16 ...
# tag_encoder =
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


x_train, y_train, tag_encoder, num_tags = create_inputs_targets_conll(texts, tags , tag_encoder)
model = create_model(num_tags)

model.fit(x_train, y_train, epochs=1, verbose=1, batch_size=16, validation_split=0.1)


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


test_inputs = ["alex lives in london"]
x_test, n_tokens = create_test_input_from_text(test_inputs)
print('input tokens')
print(x_test[0][0][:n_tokens])
pred_test = model.predict(x_test)
pred_tags = np.argmax(pred_test, 2)[0][:n_tokens]  # ignore predictions of padding tokens

# create dictionary of tag and its index
le_dict = dict(zip(tag_encoder.transform(tag_encoder.classes_), tag_encoder.classes_))
print('predicted tags')
print([le_dict.get(_, '[pad]') for _ in pred_tags])


print('stay alive')