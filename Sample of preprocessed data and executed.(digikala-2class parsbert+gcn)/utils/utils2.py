import pickle as pkl
import random
import sys
from math import log

import numpy as np
import re
import torch as th
from scipy import sparse as sp
from string import punctuation
from torch.utils import data as Data


def load_content_file(dataset_name):
    doc_content_list = []
    f = open('data/corpus/' + dataset_name + '.clean.txt', 'r',encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()
    return doc_content_list


def load_label_files(dataset_name):
    doc_name_list = []
    doc_train_list = []
    doc_test_list = []

    f = open('data/' + dataset_name + '.txt', 'r',encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()

    return doc_name_list, doc_train_list, doc_test_list


def extract_train_ids(doc_list, doc_name_list):
    _ids = []
    for name in doc_list:
        _id = doc_name_list.index(name)
        _ids.append(_id)
    random.shuffle(_ids)
    return _ids


def save_indexes(dataset_name, set_name, indexes):
    _ids_str = '\n'.join(str(index) for index in indexes)
    f = open('data/' + dataset_name + set_name + '.index', 'w',encoding="utf-8")
    f.write(_ids_str)
    f.close()


def shuffle_and_save(doc_name_list, doc_content_list, ids, dataset_name):
    shuffle_doc_name_list = []
    shuffle_doc_words_list = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_doc_words_list.append(doc_content_list[int(id)])
    shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
    shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

    f = open('data/' + dataset_name + '_shuffle.txt', 'w',encoding="utf-8")
    f.write(shuffle_doc_name_str)
    f.close()

    f = open('data/corpus/' + dataset_name + '_shuffle.txt', 'w',encoding="utf-8")
    f.write(shuffle_doc_words_str)
    f.close()
    return shuffle_doc_name_list, shuffle_doc_words_list


def build_vocab(shuffle_doc_words_list):
    word_freq = {}
    word_set = set()
    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    return word_freq, word_set, vocab, vocab_size


def word_doc_list(shuffle_doc_words_list):
    word_doc_list = {}
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    return word_doc_list, word_doc_freq


def word_2_ids(vocab_size, vocab, dataset):
    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    vocab_str = '\n'.join(vocab)

    f = open('data/corpus/' + dataset + '_vocab.txt', 'w',encoding="utf-8")
    f.write(vocab_str)
    f.close()

    return word_id_map


def extract_labels(shuffle_doc_name_list, dataset):
    label_set = set()
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split('\t')
        label_set.add(temp[2])
    label_list = list(label_set)

    label_list_str = '\n'.join(label_list)
    f = open('data/corpus/' + dataset + '_labels.txt', 'w',encoding="utf-8")
    f.write(label_list_str)
    f.close()

    return label_list


def split_train_val(train_ids, shuffle_doc_name_list, dataset):
    train_size = len(train_ids)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size
    real_train_doc_names = shuffle_doc_name_list[:real_train_size]
    real_train_doc_names_str = '\n'.join(real_train_doc_names)

    f = open('data/' + dataset + '.real_train.name', 'w',encoding="utf-8")
    f.write(real_train_doc_names_str)
    f.close()

    return real_train_size, val_size


def create_train_x_y(real_train_size, shuffle_doc_words_list, word_embeddings_dim, word_vector_map,
                     shuffle_doc_name_list, label_list):
    row_x = []
    col_x = []
    data_x = []

    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
        real_train_size, word_embeddings_dim))

    y = []
    for i in range(real_train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)

    return x, y


def create_test_x_y(test_ids, word_embeddings_dim, shuffle_doc_words_list, train_size,
                    word_vector_map,
                    shuffle_doc_name_list, label_list):
    test_size = len(test_ids)

    row_tx = []
    col_tx = []
    data_tx = []
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i + train_size]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                       shape=(test_size, word_embeddings_dim))

    ty = []
    for i in range(test_size):
        doc_meta = shuffle_doc_name_list[i + train_size]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ty.append(one_hot)
    ty = np.array(ty)

    return tx, ty


def create_words_vectors(vocab_size, word_embeddings_dim, vocab, word_vector_map):
    word_vectors = np.random.uniform(-0.01, 0.01,
                                     (vocab_size, word_embeddings_dim))

    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector
    return word_vectors


def create_all_x_y(train_size, word_embeddings_dim, shuffle_doc_words_list,
                   word_vector_map, vocab_size, word_vectors, shuffle_doc_name_list,
                   label_list, ):
    row_allx = []
    col_allx = []
    data_allx = []

    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            data_allx.append(doc_vec[j] / doc_len)

    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))

    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)

    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

    ally = []
    for i in range(train_size):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)

    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)

    ally = np.array(ally)

    return allx, ally


def create_doc_word_hetero(shuffle_doc_words_list, word_id_map):
    window_size = 20
    windows = []

    for doc_words in shuffle_doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)

    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    return windows, word_window_freq, word_pair_count


def create_pmi(windows, word_pair_count, word_window_freq, train_size, vocab):
    row = []
    col = []
    weight = []

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

    return row, col, weight


def create_adj(shuffle_doc_words_list, word_id_map, train_size, row, col, weight, vocab_size,
               word_doc_freq, vocab, test_size):
    doc_word_freq = {}

    for doc_id in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    return adj


def save_files(dataset, train_x, train_y, test_x, test_y, allx, ally, adj):
    f = open("data/ind.{}.x".format(dataset), 'wb')
    pkl.dump(train_x, f)
    f.close()

    f = open("data/ind.{}.y".format(dataset), 'wb')
    pkl.dump(train_y, f)
    f.close()

    f = open("data/ind.{}.tx".format(dataset), 'wb')
    pkl.dump(test_x, f)
    f.close()

    f = open("data/ind.{}.ty".format(dataset), 'wb')
    pkl.dump(test_y, f)
    f.close()

    f = open("data/ind.{}.allx".format(dataset), 'wb')
    pkl.dump(allx, f)
    f.close()

    f = open("data/ind.{}.ally".format(dataset), 'wb')
    pkl.dump(ally, f)
    f.close()

    f = open("data/ind.{}.adj".format(dataset), 'wb')
    pkl.dump(adj, f)
    f.close()


def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))

    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def encode_input(text, tokenizer, max_length):
    input = tokenizer(text, max_length=max_length, truncation=True, padding=True,
                      return_tensors='pt')
    return input.input_ids, input.attention_mask


def create_loader(text, max_length, nb_train, nb_val, nb_test, model, batch_size, label):
    input_ids, attention_mask = {}, {}
    input_ids_, attention_mask_ = encode_input(text, model.tokenizer, max_length=max_length)

    input_ids['train'], input_ids['val'], input_ids['test'] = input_ids_[:nb_train], \
                                                              input_ids_[
                                                              nb_train:nb_train + nb_val], \
                                                              input_ids_[-nb_test:]

    attention_mask['train'], attention_mask['val'], attention_mask['test'] = attention_mask_[
                                                                             :nb_train], \
                                                                             attention_mask_[
                                                                             nb_train:nb_train + nb_val], \
                                                                             attention_mask_[
                                                                             -nb_test:]

    datasets = {}
    loader = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = Data.TensorDataset(input_ids[split], attention_mask[split], label[split])
        loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True)

    return loader


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def create_gcn_loader(nb_train, nb_val, nb_test, nb_node, batch_size):
    train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
    val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
    test_idx = Data.TensorDataset(th.arange(nb_node - nb_test, nb_node, dtype=th.long))
    doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

    idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
    idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
    idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
    idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

    return idx_loader_train, idx_loader_val, idx_loader_test, idx_loader


def normalize_text(input_data: str, normalizer):
    """
    a method to normalize text

    :param input_data: input text to be normalized
    :param normalizer: normalizer object
    :return: normalize text
    """
    input_text = input_data.rstrip('\r\n').strip()

    normalized_text = normalizer.normalize(input_text)
    # normalized_text = normalized_text.lower()

    # Convert half-space to white space
    normalized_text = re.sub("\u200c", " ", normalized_text)

    # Remove more than 2 times repeat of character
    normalized_text = re.sub(r"(.)\1+", r"\1\1", normalized_text)

    # normalized_text = re.sub(r"/[!#$%&'()*+,-./:;<=>?@[\]^_`{|}~]/g?؟٬@", "",
    #                          normalized_text)
    normalized_text = re.sub(f'[{punctuation}؟،٪×÷»«]+', '', normalized_text)

    # Remove multiple spaces
    normalized_text = re.sub(r"\s\s+", " ", normalized_text)

    # Add space between Numbers and Alphabets in String
    normalized_text = re.sub(r"(\d+(\.\d+)?)", r" \1 ", normalized_text)
    return normalized_text
