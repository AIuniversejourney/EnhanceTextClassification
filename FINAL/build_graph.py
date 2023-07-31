import logging

from configuration.configuration import BaseConfig
from utils import load_content_file, load_label_files, extract_train_ids, save_indexes, \
    shuffle_and_save, build_vocab, word_doc_list, word_2_ids, extract_labels, split_train_val, \
    create_train_x_y, create_test_x_y, create_words_vectors, create_all_x_y, \
    create_doc_word_hetero, create_pmi, create_adj, save_files

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    DATASET = ARGS.dataset
    WORD_EMBEDDINGS_DIM = ARGS.word_embeddings_dim
    WORD_VECTOR_MAP = {}

    DOC_NAME_LIST, DOC_TRAIN_LIST, DOC_TEST_LIST = load_label_files(dataset_name=DATASET)
    DOC_CONTENT_LIST = load_content_file(dataset_name=DATASET)
    logging.info('loaded files')

    TRAIN_IDS = extract_train_ids(doc_name_list=DOC_NAME_LIST, doc_list=DOC_TRAIN_LIST)
    save_indexes(dataset_name=DATASET, set_name='.train', indexes=TRAIN_IDS)

    TEST_IDS = extract_train_ids(doc_name_list=DOC_NAME_LIST, doc_list=DOC_TEST_LIST)
    save_indexes(dataset_name=DATASET, set_name='.test', indexes=TEST_IDS)
    logging.info('extracted train and test ids')

    IDS = TRAIN_IDS + TEST_IDS
    SHUFFLE_DOC_NAME_LIST, SHUFFLE_DOC_WORDS_LIST = shuffle_and_save(
        doc_name_list=DOC_NAME_LIST,
        doc_content_list=DOC_CONTENT_LIST,
        ids=IDS,
        dataset_name=DATASET)
    logging.info('shuffle and save data')

    WORD_FREQ, WORD_SET, VOCAB, VOCAB_SIZE = build_vocab(
        shuffle_doc_words_list=SHUFFLE_DOC_WORDS_LIST)
    logging.info('built vocabs')

    WORD_DOC_LIST, WORD_DOC_FREQ = word_doc_list(shuffle_doc_words_list=SHUFFLE_DOC_WORDS_LIST)
    logging.info('word doc list vocabs')

    WORD_ID_MAP = word_2_ids(vocab_size=VOCAB_SIZE, vocab=VOCAB, dataset=DATASET)
    logging.info('word id map created')

    LABEL_LIST = extract_labels(shuffle_doc_name_list=SHUFFLE_DOC_NAME_LIST, dataset=DATASET)
    logging.info('extract label')

    REAL_TRAIN_SIZE, VAL_SIZE = split_train_val(train_ids=TRAIN_IDS,
                                                shuffle_doc_name_list=SHUFFLE_DOC_NAME_LIST,
                                                dataset=DATASET)
    logging.info('split train and val')

    TRAIN_X, TRAIN_Y = create_train_x_y(real_train_size=REAL_TRAIN_SIZE,
                                        shuffle_doc_words_list=SHUFFLE_DOC_WORDS_LIST,
                                        word_vector_map=WORD_VECTOR_MAP,
                                        word_embeddings_dim=WORD_EMBEDDINGS_DIM,
                                        shuffle_doc_name_list=SHUFFLE_DOC_NAME_LIST,
                                        label_list=LABEL_LIST)

    logging.info('train x and y created')

    TEST_X, TEST_Y = create_test_x_y(test_ids=TEST_IDS,
                                     word_embeddings_dim=WORD_EMBEDDINGS_DIM,
                                     shuffle_doc_words_list=SHUFFLE_DOC_WORDS_LIST,
                                     train_size=len(TRAIN_IDS),
                                     word_vector_map=WORD_VECTOR_MAP,
                                     shuffle_doc_name_list=SHUFFLE_DOC_NAME_LIST,
                                     label_list=LABEL_LIST)
    logging.info('test x and y created')

    WORD_VECTORS = create_words_vectors(vocab=VOCAB, word_vector_map=WORD_VECTOR_MAP,
                                        word_embeddings_dim=WORD_EMBEDDINGS_DIM,
                                        vocab_size=VOCAB_SIZE)

    ALLX, ALLY = create_all_x_y(train_size=len(TRAIN_IDS), word_embeddings_dim=WORD_EMBEDDINGS_DIM,
                                shuffle_doc_words_list=SHUFFLE_DOC_WORDS_LIST,
                                word_vector_map=WORD_VECTOR_MAP, vocab_size=VOCAB_SIZE,
                                word_vectors=WORD_VECTORS,
                                shuffle_doc_name_list=SHUFFLE_DOC_NAME_LIST,
                                label_list=LABEL_LIST)
    logging.info('all x and y created')

    WINDOWS, WORD_WINDOW_FREQ, WORD_PAIR_COUNT = create_doc_word_hetero(
        shuffle_doc_words_list=SHUFFLE_DOC_WORDS_LIST, word_id_map=WORD_ID_MAP)

    logging.info('windows and word window freq created')

    ROW, COL, WEIGHT = create_pmi(windows=WINDOWS, word_pair_count=WORD_PAIR_COUNT,
                                  word_window_freq=WORD_WINDOW_FREQ, train_size=len(TRAIN_IDS),
                                  vocab=VOCAB)

    logging.info('pmi created')

    ADJ = create_adj(shuffle_doc_words_list=SHUFFLE_DOC_WORDS_LIST,
                     word_id_map=WORD_ID_MAP, train_size=len(TRAIN_IDS), row=ROW, col=COL,
                     weight=WEIGHT, vocab_size=VOCAB_SIZE, word_doc_freq=WORD_DOC_FREQ, vocab=VOCAB,
                     test_size=len(TEST_IDS))

    logging.info('adj created')

    save_files(dataset=DATASET, train_x=TRAIN_X, train_y=TRAIN_Y, test_x=TEST_X, test_y=TEST_Y,
               allx=ALLX, ally=ALLY, adj=ADJ)

    logging.info('file saves')
