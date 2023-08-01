import argparse
from pathlib import Path


class BaseConfig:
    """
    Base Config Class
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, default="annotation", help="Model name")
        self.parser.add_argument('--max_length', type=int, default=64,
                                 help='the input length for bert')
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--batch_size_gcn', type=int, default=128)
        self.parser.add_argument('-m', '--m', type=float, default=0.5,
                                 help='the factor balancing BERT and GCN prediction')
        self.parser.add_argument('--pretrained_bert_ckpt',
                                 default=Path(__file__).parents[
                                             2].__str__() + "/checkpoint/digi2/checkpoint.pth")

        self.parser.add_argument('--word_embeddings_dim', type=int, default=300)
        self.parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn'])
        self.parser.add_argument('--gcn_layers', type=int, default= 3)
        self.parser.add_argument('--n_hidden', type=int, default= 30,
                                 help='the dimension of gcn hidden layer,'
                                      ' the dimension for gat is n_hidden * heads')
        self.parser.add_argument('--heads', type=int, default=16,
                                 help='the number of attention heads for gat')
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--gcn_lr', type=float, default=1e-3)
        self.parser.add_argument('--bert_lr', type=float, default=1e-5)

        self.parser.add_argument('--nb_epochs', type=int, default=4)
        self.parser.add_argument('--nb_epochs_gcn', type=int, default=10)
        self.parser.add_argument('--dataset', default='digi2',
                                 choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'digi3', 'digi2'])
        self.parser.add_argument('--bert_init', type=str,
                                 default='/home/LanguageModels/ParsBERT_v3/',
                                 choices=['ParsBERT_v3', 'roberta-base', 'roberta-large',
                                          'bert-base-uncased',
                                          'bert-large-uncased'])

    def get_config(self):
        """
        to return arg pass and configuration args
        :return: parser
        """

        return self.parser.parse_args()
