'''
mostly this:
https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
without tpu stuff

training on a toy dataset...
does read input, tokenizes and predicts
transformer layer can be set trainable (see code)
trainable True: takes long but results are good
tranable False: quick training but no good results

'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
import tqdm
from tokenizers import BertWordPieceTokenizer

# should replace simple lists with tf.data.Dataset objects
train = pd.read_csv(r'.\data\train.csv')
train = train[:500]
train_x = train.comment_text
# train_y = train.toxic

print(train_x)

EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 128
num_labels = 5


def fast_encode_predict_single_input(texts, tokenizer, chunk_size=256, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []

    for i in tqdm.tqdm(range(0, len(texts), chunk_size)):
        #         text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(texts)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=MAX_LEN):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding()  # max_length=maxlen)
    all_ids = []

    for i in tqdm.tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i + chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


def build_model(transformer, max_len=MAX_LEN):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    print('sequence output shape: ', sequence_output.shape)
    # cls_token = sequence_output[:, 0, :]
    # print('cls token output shape: ', cls_token.shape)
    out = Dense(num_labels, activation='softmax')(sequence_output)  # classifier using cls_token from transformer --> no conv1d necessary

    ## _todo:
    # change out layer to token based classification per word instead clf token (input max_seq_len, output num_label
    # change transformer layer output from clf token to output per word
    ##
    model = Model(inputs=input_word_ids, outputs=out)

    model.layers[1].trainable = False  # set distilbert model non-trainable. fast in training but bad results
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

#transformer_layer = (transformers.TFDistilBertModelForTokenClassification.from_pretrained('distilbert-base-multilingual-cased'))

transformer_layer = transformers.TFBertModel.from_pretrained('bert-base-uncased') -->possible alternative
model = build_model(transformer_layer, max_len=MAX_LEN)
#print(model.summary())
train_x = fast_encode(train_x, tokenizer=fast_tokenizer, maxlen=MAX_LEN)
train_y = train_x
train_x

# train_y[train_y < 10000] = [0,0,0,0,0]
# train_y[train_y > 10000] = [1,1,1,1,1]
# train_y = np.expand_dims(train_y, axis=2)
train_y = train_y.tolist()

for i in range(len(train_y)):
    for j in range(len(train_y[i])):
        if train_y[i][j] > 10000:
            train_y[i][j] = [0, 0, 1, 0, 0]
        elif train_y[i][j] < 200:
            train_y[i][j] = [1, 0, 0, 0, 0]
        else:
            train_y[i][j] = [0, 1, 0, 0, 0]

train_y = np.array(train_y)
print(train_y.shape)

#model.fit(train_x, train_y, epochs=2, validation_split=0.3, batch_size=16, verbose=1)
#
#
# in_encoded = fast_encode(["Wow, what a crazy time to be alive."], tokenizer=fast_tokenizer) #, max_len=MAX_LEN)
results = model.predict(train_x)

# print(results)