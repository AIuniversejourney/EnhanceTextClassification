import logging
import os

import dgl
import torch
import torch.nn.functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from configuration.configuration import BaseConfig
from model import BertGCN
from \
    utils import *
from utils import create_gcn_loader

CONFIG_CLASS = BaseConfig()
ARGS = CONFIG_CLASS.get_config()
CKPT_DIR = '../checkpoint/{}_{}'.format(ARGS.gcn_model, ARGS.dataset)
os.makedirs(CKPT_DIR, exist_ok=True)

SH = logging.StreamHandler(sys.stdout)
SH.setFormatter(logging.Formatter('%(message)s'))
SH.setLevel(logging.INFO)
FH = logging.FileHandler(filename=os.path.join(CKPT_DIR, 'training.log'), mode='w')
FH.setFormatter(logging.Formatter('%(message)s'))
FH.setLevel(logging.INFO)
LOGGER = logging.getLogger('training logger')
LOGGER.addHandler(SH)
LOGGER.addHandler(FH)
LOGGER.setLevel(logging.INFO)

CPU = th.device('cpu')
#GPU = th.device('cuda:0')

LOGGER.info('arguments:')
LOGGER.info(str(ARGS))
LOGGER.info('checkpoints will be saved in {}'.format(CKPT_DIR))


def update_feature():
    global MODEL, G, DOC_MASK
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(G.ndata['input_ids'][DOC_MASK], G.ndata['attention_mask'][DOC_MASK]),
        batch_size=1024
    )
    with th.no_grad():
        model = MODEL.to(CPU)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(CPU) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = G.to(CPU)
    g.ndata['cls_feats'][DOC_MASK] = cls_feat
    return g


def train_step(engine, batch):
    global MODEL, G, OPTIMIZER
    MODEL.train()
    model = MODEL.to(CPU)
    g = G.to(CPU)
    OPTIMIZER.zero_grad()
    (idx,) = [x.to(CPU) for x in batch]
    OPTIMIZER.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    OPTIMIZER.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


def test_step(engine, batch):
    global MODEL, G
    with th.no_grad():
        MODEL.eval()
        model = MODEL.to(CPU)
        g = G.to(CPU)
        (idx,) = [x.to(CPU) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true


if __name__ == '__main__':

    ADJ, FEATURES, Y_TRAIN, Y_VAL, Y_TEST, TRAIN_MASK, VAL_MASK, \
    TEST_MASK, TRAIN_SIZE, TEST_SIZE = load_corpus(ARGS.dataset)

    NB_NODE = FEATURES.shape[0]
    NB_TRAIN, NB_VAL, NB_TEST = TRAIN_MASK.sum(), VAL_MASK.sum(), TEST_MASK.sum()
    NB_WORD = NB_NODE - NB_TRAIN - NB_VAL - NB_TEST
    NB_CLASS = Y_TRAIN.shape[1]

    if ARGS.gcn_model == 'gcn':
        MODEL = BertGCN(nb_class=NB_CLASS, pretrained_model="C:/Users/AGM1/F.Gh/parsbert", m=ARGS.m,
                        gcn_layers=ARGS.gcn_layers,
                        n_hidden=ARGS.n_hidden, dropout=ARGS.dropout)

    if ARGS.pretrained_bert_ckpt is not None:
        CKPT = th.load(ARGS.pretrained_bert_ckpt, map_location=CPU)
        MODEL.bert_model.load_state_dict(CKPT['bert_model'])
        MODEL.classifier.load_state_dict(CKPT['classifier'])

    with open('C:/Users/AGM1/PycharmProjects/BertGCNfinal - digi2/data/corpus/' + ARGS.dataset + '_shuffle.txt', 'r',encoding="utf-8") as f:
        TEXT = f.read()
        TEXT = TEXT.replace('\\', '')
        TEXT = TEXT.split('\n')

    INPUT_IDS, ATTENTION_MASK = encode_input(TEXT, MODEL.tokenizer, max_length=ARGS.max_length)

    INPUT_IDS = th.cat([INPUT_IDS[:-NB_TEST],
                        th.zeros((NB_WORD, ARGS.max_length), dtype=th.long),
                        INPUT_IDS[-NB_TEST:]])

    ATTENTION_MASK = th.cat([ATTENTION_MASK[:-NB_TEST],
                             th.zeros((NB_WORD, ARGS.max_length), dtype=th.long),
                             ATTENTION_MASK[-NB_TEST:]])

    Y = Y_TRAIN + Y_TEST + Y_VAL
    Y_TRAIN = Y_TRAIN.argmax(axis=1)
    Y = Y.argmax(axis=1)

    # document mask used for update feature

    DOC_MASK = TRAIN_MASK + VAL_MASK + TEST_MASK

    # build DGL Graph
    ADJ_NORM = normalize_adj(ADJ + sp.eye(ADJ.shape[0]))
    G = dgl.from_scipy(ADJ_NORM.astype('float32'), eweight_name='edge_weight')
    G.ndata['input_ids'], G.ndata['attention_mask'] = INPUT_IDS, ATTENTION_MASK
    G.ndata['label'], G.ndata['train'], G.ndata['val'], G.ndata['test'] = \
        th.LongTensor(Y), th.FloatTensor(TRAIN_MASK), th.FloatTensor(VAL_MASK), th.FloatTensor(
            TEST_MASK)
    G.ndata['label_train'] = th.LongTensor(Y_TRAIN)
    G.ndata['cls_feats'] = th.zeros((NB_NODE, MODEL.feat_dim))

    LOGGER.info('graph information:')
    LOGGER.info(str(G))

    IDX_LOADER_TRAIN, IDX_LOADER_VAL, IDX_LOADER_TEST, IDX_LOADER = create_gcn_loader(
        nb_train=NB_TRAIN,
        nb_test=NB_TEST,
        nb_val=NB_VAL,
        nb_node=NB_NODE,
        batch_size=ARGS.batch_size_gcn, )

    OPTIMIZER = th.optim.Adam([
        {'params': MODEL.bert_model.parameters(), 'lr': ARGS.bert_lr},
        {'params': MODEL.classifier.parameters(), 'lr': ARGS.bert_lr},
        {'params': MODEL.gcn.parameters(), 'lr': ARGS.gcn_lr},
    ], lr=1e-3
    )
    SCHEDULER = lr_scheduler.MultiStepLR(OPTIMIZER, milestones=[30], gamma=0.1)

    TRAINER = Engine(train_step)


    @TRAINER.on(Events.EPOCH_COMPLETED)
    def reset_graph(trainer):
        SCHEDULER.step()
        update_feature()
        th.cuda.empty_cache()


    EVALUATOR = Engine(test_step)
    METRICS = {
        'acc': Accuracy(),
        'nll': Loss(th.nn.NLLLoss())
    }
    for n, f in METRICS.items():
        f.attach(EVALUATOR, n)


    @TRAINER.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        EVALUATOR.run(IDX_LOADER_TRAIN)
        metrics = EVALUATOR.state.metrics
        train_acc, train_nll = metrics["acc"], metrics["nll"]
        EVALUATOR.run(IDX_LOADER_VAL)
        metrics = EVALUATOR.state.metrics
        val_acc, val_nll = metrics["acc"], metrics["nll"]
        EVALUATOR.run(IDX_LOADER_TEST)
        metrics = EVALUATOR.state.metrics
        test_acc, test_nll = metrics["acc"], metrics["nll"]
        LOGGER.info(
            "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} "
            "loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
                .format(trainer.state.epoch, train_acc, train_nll,
                        val_acc, val_nll, test_acc, test_nll)
        )
        if val_acc > log_training_results.best_val_acc:
            LOGGER.info("New checkpoint")
            th.save(
                {
                    'bert_model': MODEL.bert_model.state_dict(),
                    'classifier': MODEL.classifier.state_dict(),
                    'gcn': MODEL.gcn.state_dict(),
                    'optimizer': OPTIMIZER.state_dict(),
                    'epoch': trainer.state.epoch,
                },
                os.path.join(
                    CKPT_DIR, 'checkpoint.pth'
                )
            )
            log_training_results.best_val_acc = val_acc


    log_training_results.best_val_acc = 0
    g = update_feature()
    TRAINER.run(IDX_LOADER, max_epochs=ARGS.nb_epochs_gcn)
