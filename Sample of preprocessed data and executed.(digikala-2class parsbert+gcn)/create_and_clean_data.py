import pandas
import parsivar
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import build_vocab, normalize_text


def remove_under_(dataframe, threshold):
    comments_ = []
    labels_ = []
    for i, item in enumerate(list(dataframe.comment)):
        if len(item.split()) >= threshold:
            comments_.append(item)
            labels_.append(list(dataframe.recommend)[i])
    return pandas.DataFrame({'comment': comments_, 'recommend': labels_})


if __name__ == '__main__':
    NORMALIZER = parsivar.Normalizer()

    DIGI_DATASET = pandas.read_excel("C:/Users/AGM1/F.Gh/Digikala- 2Class.csv")

DIGI_DATASET = pandas.DataFrame({'comment': list(DIGI_DATASET['comment']),
                                     'recommend': list(DIGI_DATASET['recommend'])})

    OPTION = ['not_recommended', 'recommended']
    DIGI_DATASET = DIGI_DATASET.loc[DIGI_DATASET['recommend'].isin(OPTION)]
    DIGI_DATASET = DIGI_DATASET.dropna(subset=['comment', 'recommend'])
    print('separate 2 class ...')
    print('value counts : \n', DIGI_DATASET['recommend'].value_counts())

    DIGI_DATASET.comment = [str(item) for item in DIGI_DATASET.comment]
    DIGI_DATASET.comment = DIGI_DATASET.comment.apply(
        lambda x: normalize_text(x, NORMALIZER))
    print('value counts after normalize: \n', DIGI_DATASET['recommend'].value_counts())
    print('normalizer ....')

    DIGI_DATASET = remove_under_(dataframe=DIGI_DATASET, threshold=5)
    print('remove under threshold ....')
    print('value counts : \n', DIGI_DATASET['recommend'].value_counts())

    # Balancing the data
    original_train_sentences, original_labels_train = list(DIGI_DATASET['comment']), list(DIGI_DATASET['recommend'])
    vectorizer = TfidfVectorizer()
    vectorizer.fit(original_train_sentences)

    X_train_tf = vectorizer.transform(original_train_sentences)
    X_train_tf = X_train_tf.toarray()
    y_train = [0 if x == 'not_recommended' else 1 for x in original_labels_train]  # Convert labels to binary
    ROS = RandomOverSampler(sampling_strategy=1)
    X_train_ros, y_train_ros = ROS.fit_resample(X_train_tf, y_train)
    Counter(y_train_ros)

    # Splitting the data into train and test sets
    TRAIN, TEST, original_labels_train, original_labels_test = train_test_split(DIGI_DATASET, original_labels_train, test_size=0.1)
    TRAIN.comment, TEST.comment = list(TRAIN['comment']), list(TEST['comment'])

CONTEXT_FILE = open("C:/Users/AGM1/PycharmProjects/BertGCNfinal/BertGcn final- digi2/data/corpus/digi2B.txt", "w", encoding="utf-8")
CONTEXT_CLEAN_FILE = open("C:/Users/AGM1/PycharmProjects/BertGCNfinal/BertGcn final- digi2/data/corpus/digi2.cleanB.txt", "w", encoding="utf-8")
LABEL_FILE = open("C:/Users/AGM1/PycharmProjects/BertGCNfinal/BertGcn final- digi2/data/digi2B.txt", "w", encoding="utf-8")

for i, item in enumerate(list(TRAIN.comment)):
        CONTEXT_FILE.write(item)
        CONTEXT_FILE.write('\n')

        CONTEXT_CLEAN_FILE.write(item)
        CONTEXT_CLEAN_FILE.write('\n')

        LABEL_FILE.write(str(i))
        LABEL_FILE.write('\t')
        LABEL_FILE.write('train')
        LABEL_FILE.write('\t')
        LABEL_FILE.write(list(TRAIN.recommend)[i])
        LABEL_FILE.write('\n')
print('saved train ...')

for i, item in enumerate(list(TEST.comment)):
        CONTEXT_FILE.write(item)
        CONTEXT_FILE.write('\n')

        CONTEXT_CLEAN_FILE.write(item)
        CONTEXT_CLEAN_FILE.write('\n')

        LABEL_FILE.write(str(len(TRAIN) + i))
        LABEL_FILE.write('\t')
        LABEL_FILE.write('test')
        LABEL_FILE.write('\t')
        LABEL_FILE.write(list(TEST.recommend)[i])
        LABEL_FILE.write('\n')
print('saved test ...')
