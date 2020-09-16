from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import bert_tokenization

def read_data(fname, source_word2idx, max_length, is_testing=False):
    source_data, aspect_y, opinion_y, source_mask, sentiment_y = list(), list(), list(), list(), list()
    position_m = list()
    sentiment_mask = list()

    review = open(fname + r'sentence.txt', 'r', encoding='utf-8').readlines()
    ae_data = open(fname + r'target.txt', 'r', encoding='utf-8').readlines()
    oe_data = open(fname + r'opinion.txt', 'r', encoding='utf-8').readlines()
    sa_data = open(fname + r'target_polarity.txt', 'r', encoding='utf-8').readlines()
    for index, _ in enumerate(review):
        '''
        Word Index
        '''
        sptoks = review[index].strip().split()

        idx = []
        mask = []
        len_cnt = 0
        for sptok in sptoks:
            if len_cnt < max_length:
                idx.append(source_word2idx[sptok.lower()])
                mask.append(1.)
                len_cnt += 1
            else:
                break

        source_data.append(idx + [0] * (max_length - len(idx)))
        source_mask.append(mask + [0.] * (max_length - len(idx)))

        ae_labels = ae_data[index].strip().split()
        aspect_label = []
        for l in ae_labels:
            l = int(l)
            if l == 0 :
                aspect_label.append([1, 0, 0])
            elif l == 1:
                aspect_label.append([0, 1, 0])
            elif l == 2:
                aspect_label.append([0, 0, 1])
            else:
                raise ValueError

        oe_labels = oe_data[index].strip().split()
        opinion_label = []
        for l in oe_labels:
            l = int(l)
            if l == 0 :
                opinion_label.append([1, 0, 0])
            elif l == 1:
                opinion_label.append([0, 1, 0])
            elif l == 2:
                opinion_label.append([0, 0, 1])
            else:
                raise ValueError

        sa_labels = sa_data[index].strip().split()
        sentiment_label = []
        sentiment_indi = []
        for l in sa_labels:
            l = int(l)
            if l == 0:
                sentiment_label.append([0, 0, 0])
                sentiment_indi.append(1. if is_testing else 0.) # In testing, we don't know the location of aspect terms.
            elif l == 1:
                sentiment_label.append([1, 0, 0])
                sentiment_indi.append(1.)
            elif l == 2:
                sentiment_label.append([0, 1, 0])
                sentiment_indi.append(1.)
            elif l == 3:
                sentiment_label.append([0, 0, 1])
                sentiment_indi.append(1.)
            elif l == 4:
                sentiment_label.append([0, 0, 0])
                sentiment_indi.append(1. if is_testing else 0.) # Ensure there is no label leakage in testing.
            else:
                raise ValueError

        aspect_y.append(aspect_label + [[0, 0, 0]] * (max_length - len(idx)))
        opinion_y.append(opinion_label + [[0, 0, 0]] * (max_length - len(idx)))
        sentiment_y.append(sentiment_label + [[0, 0, 0]] * (max_length - len(idx)))
        position_m.append(position_matrix(len(idx), max_length))
        sentiment_mask.append(sentiment_indi + [0.] * (max_length - len(idx)))

    return np.array(source_data), \
           np.array(aspect_y), \
           np.array(opinion_y), \
           np.array(sentiment_y), \
           np.array(source_mask), \
           np.array(sentiment_mask), \
           np.array(position_m)

def read_bert_data(fname, max_length, is_testing=False):
    aspect_y, opinion_y, sentiment_y, source_mask, sentiment_mask = list(), list(), list(), list(), list()
    bert_input, bert_mask, bert_segment = list(), list(), list()
    position_m = list()

    review = open(fname + r'sentence.txt', 'r', encoding='utf-8').readlines()
    ae_data = open(fname + r'target.txt', 'r', encoding='utf-8').readlines()
    oe_data = open(fname + r'opinion.txt', 'r', encoding='utf-8').readlines()
    sa_data = open(fname + r'target_polarity.txt', 'r', encoding='utf-8').readlines()
    vocab_file = "./bert-large/vocab.txt"
    tokenizer = bert_tokenization.WordpieceTokenizer(vocab_file=vocab_file)

    for index, _ in enumerate(review):
        '''
        Word Index
        '''
        raw_tokens = review[index].strip().split()
        ae_labels = list(map(int, ae_data[index].strip().split()))
        oe_labels = list(map(int, oe_data[index].strip().split()))
        sa_labels = list(map(int, sa_data[index].strip().split()))

        split_tokens = []
        split_ae_labels = []
        split_oe_labels = []
        split_sa_labels = []

        for ix, raw_token in enumerate(raw_tokens):
            raw_token = raw_token.lower()
            sub_tokens= tokenizer.tokenize(raw_token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                'For Aspect&Opinion Labels'
                if ae_labels[ix]==1 and jx>0:
                    split_ae_labels.append(2)
                else:
                    split_ae_labels.append(ae_labels[ix])

                if oe_labels[ix]==1 and jx>0:
                    split_oe_labels.append(2)
                else:
                    split_oe_labels.append(oe_labels[ix])

                split_sa_labels.append(sa_labels[ix])

        if len(split_tokens) > max_length - 2:
            print('Over Length')
            raise ValueError

        source_mask.append([1.] * len(split_tokens) + [0.] * (max_length - len(split_tokens)))


        onehot_ae_labels = []
        for l in split_ae_labels:
            if l == 0 :
                onehot_ae_labels.append([1, 0, 0])
            elif l == 1:
                onehot_ae_labels.append([0, 1, 0])
            elif l == 2:
                onehot_ae_labels.append([0, 0, 1])
            else:
                raise ValueError


        onehot_oe_labels = []
        for l in split_oe_labels:
            if l == 0 :
                onehot_oe_labels.append([1, 0, 0])
            elif l == 1:
                onehot_oe_labels.append([0, 1, 0])
            elif l == 2:
                onehot_oe_labels.append([0, 0, 1])
            else:
                raise ValueError

        onehot_sa_labels = []
        sa_indicator = []
        for l in split_sa_labels:
            if l == 0:
                onehot_sa_labels.append([0, 0, 0])
                sa_indicator.append(1. if is_testing else 0.) # In testing, we don't know the location of aspect terms.
            elif l == 1:
                onehot_sa_labels.append([1, 0, 0])
                sa_indicator.append(1.)
            elif l == 2:
                onehot_sa_labels.append([0, 1, 0])
                sa_indicator.append(1.)
            elif l == 3:
                onehot_sa_labels.append([0, 0, 1])
                sa_indicator.append(1.)
            elif l == 4:
                onehot_sa_labels.append([0, 0, 0])
                sa_indicator.append(1. if is_testing else 0.) # Ensure there is no label leakage in testing.
            else:
                raise ValueError

        aspect_y.append(onehot_ae_labels + [[0, 0, 0]] * (max_length - len(split_tokens)))
        opinion_y.append(onehot_oe_labels + [[0, 0, 0]] * (max_length - len(split_tokens)))
        sentiment_y.append(onehot_sa_labels + [[0, 0, 0]] * (max_length - len(split_tokens)))
        position_m.append(position_matrix(len(split_tokens), max_length))
        sentiment_mask.append(sa_indicator + [0.] * (max_length - len(split_tokens)))

        'Add [CLS] and [SEP] for BERT'
        bert_token_per = []
        bert_segment_per = []
        # bert_mask_per = []

        bert_token_per.append("[CLS]")
        bert_segment_per.append(0)

        for i, token in enumerate(split_tokens):
            bert_token_per.append(token)
            bert_segment_per.append(0)

        bert_token_per.append("[SEP]")
        bert_segment_per.append(0)

        bert_input_per = tokenizer.convert_tokens_to_ids(bert_token_per)

        bert_mask_per = [1] * len(bert_input_per)

        while len(bert_input_per) < max_length:
            bert_input_per.append(0)
            bert_mask_per.append(0)
            bert_segment_per.append(0)

        bert_input.append(bert_input_per)
        bert_mask.append(bert_mask_per)
        bert_segment.append(bert_segment_per)

    return np.array(aspect_y), \
           np.array(opinion_y), \
           np.array(sentiment_y), \
           np.array(source_mask), \
           np.array(sentiment_mask), \
           np.array(position_m), \
           np.array(bert_input), \
           np.array(bert_mask), \
           np.array(bert_segment)

def position_matrix(sen_len, max_len):
    a = np.zeros([max_len, max_len], dtype=np.float32)

    for i in range(sen_len):
        for j in range(sen_len):
            if i == j:
                a[i][j] = 0.
            else:
                a[i][j] = 1/(np.log2(2 + abs(i - j)))
                # a[i][j] = 1/(abs(i - j))

    return a

def softmask_2d(x, mask, scale=False):
    if scale == True:
        dim = tf.shape(x)[-1]
        max_x = tf.reduce_max(x, axis=-1, keepdims=True)
        max_x = tf.tile(max_x, [1, 1, dim])
        x -= max_x
    length = tf.shape(mask)[1]
    mask_d1 = tf.tile(tf.expand_dims(mask, 1), [1, length, 1])
    y = tf.multiply(tf.exp(x), mask_d1)
    sumx = tf.reduce_sum(y, axis=-1, keepdims=True)
    att = y / (sumx + 1e-10)

    mask_d2 = tf.tile(tf.expand_dims(mask, 2), [1, 1, length])
    att *= mask_d2
    return att

def count_parameter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return (total_parameters)

def min_max_normal(tensor):
    dim = tf.shape(tensor)[-1]
    max_value = tf.reduce_max(tensor, -1, keepdims=True)
    max_value = tf.tile(max_value, [1, 1, dim])
    min_value = tf.reduce_min(tensor, -1, keepdims=True)
    min_value = tf.tile(min_value, [1, 1, dim])
    norm_tensor = (tensor - min_value) / (max_value - min_value + 1e-6)

    return norm_tensor

def z_score_normal(tensor):
    dim = tf.shape(tensor)[-1]
    axes = [2]
    mean, variance = tf.nn.moments(tensor, axes, keep_dims=True)
    std = tf.sqrt(variance)
    mean = tf.tile(mean, [1, 1, dim])
    std = tf.tile(std, [1, 1, dim])
    norm_tensor = (tensor - mean) / (std + 1e-6)

    return norm_tensor

def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))

# Adopted from https://github.com/DHZS/tf-dropblock .
class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask



