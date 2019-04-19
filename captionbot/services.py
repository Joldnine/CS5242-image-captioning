from __future__ import absolute_import, division, print_function, unicode_literals
import pickle
import os
import urllib
import numpy as np
import tensorflow as tf
# from PIL import Image
# import matplotlib.pyplot as plt

DIR_FP = os.path.dirname(os.path.realpath(__file__))
TOKENIZER_FP = os.path.join(DIR_FP, 'data/cache/tokenizer.pickle')
CAP_VAL_FP = os.path.join(DIR_FP, 'data/cache/cap_val.pickle')
IMG_NAME_VAL_FP = os.path.join(DIR_FP, 'data/cache/img_name_val.pickle')
CHECKPOINT_FP = os.path.join(DIR_FP, 'data/checkpoints/train/')

BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 256
UNITS = 512
FEATURE_SHAPE = 100
MAX_LENGTH = 49


def get_feature_extractor():
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model


def load_image(image_path):
    try:
        img = tf.io.read_file(image_path)
    except:
        req = urllib.request.Request(image_path)
        img = urllib.request.urlopen(req).read()
        # response = urllib.request.urlopen(req)
        # image_data = response.read()
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def unpickle():
    with open(TOKENIZER_FP, 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open(CAP_VAL_FP, 'rb') as handle:
        cap_val = pickle.load(handle)

    with open(IMG_NAME_VAL_FP, 'rb') as handle:
        img_name_val = pickle.load(handle)
    return tokenizer, cap_val, img_name_val


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class CaptionMe():
    def __init__(self):
        self.image_features_extract_model = get_feature_extractor()
        self.tokenizer, self.cap_val, self.img_name_val = unpickle()
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.encoder = CNN_Encoder(EMBEDDING_DIM)
        self.decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, self.vocab_size)
        self.optimizer = tf.keras.optimizers.Adam()
        self.restore()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

    def restore(self):
        ckpt = tf.train.Checkpoint(
            encoder=self.encoder,
            decoder=self.decoder,
            optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, CHECKPOINT_FP, max_to_keep=5)

        if ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
            ckpt.restore(ckpt_manager.latest_checkpoint)
        else:
            raise FileNotFoundError('checkpoint cannot be found')

    def get_caption(self, img_url):
#         attention_plot = np.zeros((MAX_LENGTH, FEATURE_SHAPE))
#         hidden = self.decoder.reset_state(batch_size=1)
#         temp_input = tf.expand_dims(load_image(img_url)[0], 0)
#         img_tensor_val = self.image_features_extract_model(temp_input)
#         img_tensor_val = tf.reshape(
#             img_tensor_val,
#             (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
#         features = self.encoder(img_tensor_val)
#         dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
#         result = []
#         for i in range(MAX_LENGTH):
#             predictions, hidden, attention_weights = self.decoder(
#                 dec_input, features, hidden)
#             attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
#             predicted_id = tf.argmax(predictions[0]).numpy()
#             if self.tokenizer.index_word[predicted_id] == '<end>':
#                 return result, attention_plot
#             result.append(self.tokenizer.index_word[predicted_id])
#             dec_input = tf.expand_dims([predicted_id], 0)
#         attention_plot = attention_plot[:len(result), :]
#         return result, attention_plot
#         hidden = decoder.reset_state(batch_size=1)
############################################################
        hidden = self.decoder.reset_state(batch_size=1)
        temp_input = tf.expand_dims(load_image(img_url)[0], 0)
        img_tensor_val = self.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(MAX_LENGTH):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result, ""

            dec_input = tf.expand_dims([predicted_id], 0)

        return result, ""


if __name__ == '__main__':
    url = os.path.join(DIR_FP, 'data/zzz.jpg')
    bot = CaptionMe()
    res, _ = bot.get_caption(url)
    print(res)