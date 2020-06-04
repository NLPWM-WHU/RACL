from model import *
from utils import *
import time
import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SEED = 123
from numpy.random import seed
seed(SEED)
from tensorflow import set_random_seed
set_random_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='racl', type=str, help='model name')
parser.add_argument('--task', default='res14', type=str, help='res14 lap14 res15')
parser.add_argument('--batch_size', default=8, type=int, help='number of example per batch')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
parser.add_argument('--global_dim', default=300, type=int, help='dimension of global embedding')
parser.add_argument('--domain_dim', default=100, type=int, help='dimension of domain-specific embedding')
parser.add_argument('--kp1', default=0.5, type=float, help='dropout keep prob1')
parser.add_argument('--kp2', default=0.5, type=float, help='dropout keep prob2')
parser.add_argument('--reg_scale', default=1e-5, type=float, help='coefficient of regularization')
parser.add_argument('--filter_num', default=256, type=int, help='filter numbers')
parser.add_argument('--class_num', default=3, type=int, help='class number')
parser.add_argument('--load', default=0, type=int, help='load an existing checkpoint')
opt = parser.parse_args()

max_length_dict = {'res14':80, 'lap14':85, 'res15':70}
n_iter_dict = {'res14':80, 'lap14':80, 'res15':120}
kernel_size_dict = {'res14':3, 'lap14':3, 'res15':5}
hop_num_dict = {'res14':4, 'lap14':3, 'res15':4}

opt.max_sentence_len = max_length_dict[opt.task]
opt.n_iter = n_iter_dict[opt.task]
opt.kernel_size = kernel_size_dict[opt.task]
opt.hop_num = hop_num_dict[opt.task]

opt.warmup_iter = opt.n_iter - 20
opt.emb_dim = opt.global_dim + opt.domain_dim

opt.data_path = 'data/{}/'.format(opt.task)
opt.train_path = 'data/{}/train/'.format(opt.task)
opt.test_path = 'data/{}/test/'.format(opt.task)
opt.dev_path = 'data/{}/dev/'.format(opt.task)

def main(_):
    start_time = time.time()
    info = ''
    index = 0

    # For generating the word-idx mapping and the word vectors from scratch, just run embedding.py.
    print('Reuse Word Dictionary & Embedding')
    with open(opt.data_path + 'word2id.txt', 'r', encoding='utf-8') as f:
        word_dict = eval(f.read())
    w2v = np.load(opt.data_path + 'glove_embedding.npy')
    w2v_domain = np.load(opt.data_path + 'domain_embedding.npy')

    model = MODEL(opt, w2v, w2v_domain, word_dict)
    model.run()
    end_time = time.time()
    print('Running Time: {:.0f}m {:.0f}s'.format((end_time-start_time) // 60, (end_time-start_time) % 60))

if __name__ == '__main__':
    tf.app.run()
