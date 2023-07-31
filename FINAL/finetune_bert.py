import logging
import os

import torch as th
import torch.nn.functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler

from configuration.configuration import BaseConfig
from model import BertClassifier
from utils import *

CONFIG_CLASS = BaseConfig()
ARGS = CONFIG_CLASS.get_config()
CKPT_DIR = '../checkpoint/{}'.format(ARGS.dataset)
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
GPU = th.device('cuda:0')

LOGGER.info('arguments:')
LOGGER.info(str(ARGS))


def train_step(engine, batch):
    global MODEL, OPTIMIZER
    MODEL.train()
    model = MODEL.to(GPU)
    OPTIMIZER.zero_grad()
    (input_ids, attention_mask, label) = [x.to(GPU) for x in batch]
    OPTIMIZER.zero_grad()
    y_pred = model(input_ids, attention_mask)
    y_true = label.type(th.long)
    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    OPTIMIZER.step()
    train_loss = loss.item()
    with th.no_grad():
        y_true = y_true.detach().cpu()
        y_pred = y_pred.argmax(axis=1).detach().cpu()
        train_acc = accuracy_score(y_true, y_pred)
    return train_loss, train_acc


def test_step(engine, batch):
    global MODEL
    with th.no_grad():
        MODEL.eval()
        model = MODEL.to(GPU)
        (input_ids, attention_mask, label) = [x.to(GPU) for x in batch]
        OPTIMIZER.zero_grad()
        y_pred = model(input_ids, attention_mask)
        y_true = label
        return y_pred, y_true


if __name__ == '__main__':

    ADJ, FEATURES, Y_TRAIN, Y_VAL, Y_TEST, TRAIN_MASK, VAL_MASK, \
    TEST_MASK, TRAIN_SIZE, TEST_SIZE = load_corpus(ARGS.dataset)

    NB_NODE = ADJ.shape[0]
    NB_TRAIN, NB_VAL, NB_TEST = TRAIN_MASK.sum(), VAL_MASK.sum(), TEST_MASK.sum()
    NB_WORD = NB_NODE - NB_TRAIN - NB_VAL - NB_TEST
    NB_CLASS = Y_TRAIN.shape[1]

    MODEL = BertClassifier(pretrained_model="C:/Users/AGM1/F.Gh/parsbert", nb_class=NB_CLASS)

    Y = th.LongTensor((Y_TRAIN + Y_VAL + Y_TEST).argmax(axis=1))

    LABEL = {'train': Y[:NB_TRAIN], 'val': Y[NB_TRAIN:NB_TRAIN + NB_VAL], 'test': Y[-NB_TEST:]}

    with open('C:/Users/AGM1/PycharmProjects/BertGCNfinal - digi2/data/corpus/' + ARGS.dataset + '_shuffle.txt', 'r',encoding="utf-8") as f:
        TEXT = f.read()
        TEXT = TEXT.replace('\\', '')
        TEXT = TEXT.split('\n')

    LOADER = create_loader(text=TEXT, max_length=ARGS.max_length, nb_train=NB_TRAIN,
                           nb_test=NB_TEST, nb_val=NB_VAL, model=MODEL, batch_size=ARGS.batch_size,
                           label=LABEL)

    OPTIMIZER = th.optim.Adam(MODEL.parameters(), lr=ARGS.bert_lr)
    SCHEDULER = lr_scheduler.MultiStepLR(OPTIMIZER, milestones=[30], gamma=0.1)

    TRAINER = Engine(train_step)
    EVALUATOR = Engine(test_step)

    METRICS = {
        'acc': Accuracy(),
        'nll': Loss(th.nn.CrossEntropyLoss())
    }
    for n, f in METRICS.items():
        f.attach(EVALUATOR, n)

    @TRAINER.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        EVALUATOR.run(LOADER['train'])
        metrics = EVALUATOR.state.metrics
        train_acc, train_nll = metrics["acc"], metrics["nll"]
        EVALUATOR.run(LOADER['val'])
        metrics = EVALUATOR.state.metrics
        val_acc, val_nll = metrics["acc"], metrics["nll"]
        EVALUATOR.run(LOADER['test'])
        metrics = EVALUATOR.state.metrics
        test_acc, test_nll = metrics["acc"], metrics["nll"]
        LOGGER.info(
            "\rEpoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} "
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
                    'optimizer': OPTIMIZER.state_dict(),
                    'epoch': trainer.state.epoch,
                },
                os.path.join(
                    CKPT_DIR, 'checkpoint.pth'
                )
            )
            log_training_results.best_val_acc = val_acc
        SCHEDULER.step()


    log_training_results.best_val_acc = 0
    TRAINER.run(LOADER['train'], max_epochs=ARGS.nb_epochs)
