import pandas as pd
import numpy as np
import random
import operator
from sklearn.metrics import confusion_matrix
import pprint
import json
import matplotlib.pyplot as plt


CATEGORIES = ['Internacional','Nacional',
'Destacadas','Deportes','Salud','Ciencia y Tecnologia',
'Entretenimiento','Economia', 'Noticias destacadas']

NORM_PERCENTAGES = {
    'Internacional': 1,
    'Nacional': 1,
    'Destacadas': 1,
    'Deportes': 1,
    'Salud':1,
    'Ciencia y Tecnologia': 1,
    'Entretenimiento': 1,
    'Economia': 1,
    'Noticias destacadas': 0.02
    }

EPSILON = 0.00001


def load_data(path, separator=',', encoding='ISO-8859-1'):
    df = pd.read_csv(path, sep= separator, encoding=encoding)
    return df


def normalize_data(df, axis_name, categories):
    """
    categories should be a dict containing each category with the percentage to use
    ie:
    {
        "destacadas": 0.2,
        "deportes": 0.3
    }

    Notice, each percentage is based on the total amount for that particular category
    """
    norm_data = pd.DataFrame()
    for category in categories.keys():
        items_indexes = df[getattr(df, axis_name) == category].index.tolist()
        percentage = categories[category]
        # we make sure the percentage is correct
        percentage =  percentage if abs(percentage) < 1 else 1
        percentage *= len(items_indexes)
        selected = random.choices(items_indexes, k= int(percentage))
        category_df = df.query('index in {}'.format(selected))
        norm_data = pd.concat([norm_data, category_df])

    return norm_data

def train_test_split(df, test_percentage):
    '''
    Split a dataframe in tran a test given a percentage
    percentage should be [0, 1]
    '''
    count = len(df)
    train_count = int(test_percentage * count)
    total_indexes = df.index.tolist()
    train_indexes = random.sample(total_indexes, k=train_count)
    train_df = df.query('index in {}'.format(train_indexes))
    test_df = df.drop(train_indexes)
    return train_df, test_df


class NaiveBayes(object):

    def __init__(self, df, epsilon):
        self.df = df
        tokens, vocabulary = self.tokenize(df)
        self.tokens = tokens
        self.vocabulary = vocabulary
        self.epsilon = epsilon

    def tokenize(self, df):
        '''
        This method is used to tokenize words to form a data frame
        it returns a dict containing each category and inside each category de frequency of each word
        ie:
        {
            'Nacional': {
                'de': 2,
                'argentina': 3,
                ...
            }
        }
        '''

        # generate the vocabulary for the hole train set
        vocabulary = set()
        for index, row in df.iterrows():
            title = row['titular']
            title_vec = title.split(' ')
            for word in title_vec:
                vocabulary.add(word.lower())

        tokens = {}
        for index, row in df.iterrows():
            title = row['titular']
            title_vec = title.split(' ')
            category_dict = tokens.get(row['categoria'], {})
            for word in title_vec:
                category_dict[word.lower()] = category_dict.get(word.lower(), 0 ) + 1
            tokens[row['categoria']] = category_dict


        return tokens, vocabulary

    def calculate_categories_prob(self, categories):
        probabilities = {}
        train_size = len(self.df)
        for category in categories:
            count = len(self.df[getattr(self.df,'categoria') == category])
            probabilities[category] = count / float(train_size)

        return probabilities

    def calculate_conditional_prob(self, categories):
        probabilities = {}
        for category in categories:
            category_word_count = sum(self.tokens.get(category, dict()).values())
            for word in self.vocabulary:
                probabilities['{}|{}'.format(word, category)] = self.calculate_conditional_word_vec(word, category_word_count, category)

        return probabilities

    def calculate_conditional_word_vec(self, word, category_word_count, category):
        '''
        calculate the probabilty of P[word | category]
        '''
        word_count = self.tokens.get(category, dict()).get(word, 0)
        return (word_count + 1) / float(category_word_count + len(self.vocabulary))

    def train(self, categories):
        self.categories = categories
        self.conditional_prob = self.calculate_conditional_prob(categories)
        self.categories_prob = self.calculate_categories_prob(categories)

    def infer(self, sentece):
        vec = sentece.split(' ')
        inference = {}
        for k, v in self.categories_prob.items():
            inference[k] = float(v)
            for word in vec:

                inference[k] *= float(self.conditional_prob.get('{}|{}'.format(word.lower(), k), self.epsilon))
        return inference

    def categories_to_numeric(self, categories):
        category_table = {}
        index = 0
        for category in categories:
            category_table[category] = index
            index += 1
        return category_table

    def calculate_metrics(self, matrix, example_count, index):

        tp = matrix[index][index]
        fp = 0
        fn = 0
        tn = 0
        # calculate false negative
        for i in matrix[index]:
            fn += i

        # calculate false positive
        for i in range(len(matrix)):
            fp += matrix[index][i]

        fp -= tp
        # calculate true negative
        tn = example_count - tp - fp - fn

        accuracy = (tp + tn) / (example_count)
        presicion = (tp) / (tp + fp)
        recall = tp/float(tp + fn)
        metrics = {
           'acc': accuracy,
           'presicion': presicion,
           'tp_rate': recall,
           'fp_rate':  fn/ float(fn + tn),
           'f1': (2.0 * presicion * recall)/ (presicion + recall)
        }
        return metrics

    def test(self, sentences, categories):
        '''
        sentences: list of sentences
        categories: actual category for each, same index, sentence
        This method will run each sentence and compare the category with the actual category
        '''
        # first we transform categories to numbers
        category_table = self.categories_to_numeric(self.categories)
        conversion_table = {v: k for k, v in category_table.items()}
        predicted = []
        for sentence in sentences:
            inference = self.infer(sentence)
            max_class = max(inference.items(), key=operator.itemgetter(1))[0]
            predicted.append(max_class)

        # calculate confussion_matrix
        matrix = confusion_matrix(categories, predicted, labels=CATEGORIES)

        # calculate accuracy and presicion
        # accuracy = tp + tn /total
        # presicion = tp/tp + fp
        accuracy = {}
        presicion = {}
        true_positive_rate = {}
        false_positive_rate = {}
        f1_score = {}
        for i in range(len(CATEGORIES)):
            metrics = self.calculate_metrics(matrix, len(categories), i)
            accuracy[CATEGORIES[i]] = metrics['acc']
            presicion[CATEGORIES[i]] = metrics['presicion']
            true_positive_rate[CATEGORIES[i]] = metrics['tp_rate']
            false_positive_rate[CATEGORIES[i]] = metrics['fp_rate']
            f1_score[CATEGORIES[i]] = metrics['f1']


        metrics = {
            'accuracy': accuracy,
            'presicion': presicion,
            'tp_rate': true_positive_rate,
            'fp_rate': false_positive_rate,
            'f1_score': f1_score,
            'roc_point': (false_positive_rate, true_positive_rate)
        }
        return matrix, metrics


def show_train(classifier, test_data, metric_path):
    # train the classifier to understand each class
    classifier.train(CATEGORIES)
    test_data.dropna()
    categories = test_data.categoria.tolist()
    sentences = test_data.titular.tolist()

    for i in range(len(categories)-1):
        try:
            if not(categories[i] in CATEGORIES):
                categories.pop(i)
                sentences.pop(i)
        except IndexError:
            break

    conf_matrix, metrics = classifier.test(sentences, categories)

    print('Confussion matrix')
    print(conf_matrix)


    with open(metric_path, 'w') as f:
        json.dump(metrics, f)


def main():
    path = './data/Noticias_argentinas.csv'
    data = load_data(path, ';')

    # normalize data, try to get equal amount of data for each class
    norm_data = normalize_data(data, 'categoria', NORM_PERCENTAGES)
    train_data, test_data = train_test_split(norm_data, 0.85)
    print('train size: {} \n test size: {}'.format(len(train_data), len(test_data)))
    # create classifier
    classifier = NaiveBayes(train_data, EPSILON)
    # show results
    show_train(classifier, test_data, 'metrics_85.json')

    while False:
        title = input('enter a title: ')
        result = classifier.infer(title)
        print('results')
        pprint.pprint(result)


def plot_graphs( paths = ['metrics_7.json', 'metrics_75.json', 'metrics_8.json', 'metrics_85.json', 'metrics_9.json']):

    classes = {}
    for path in paths:
        with open(path, 'r') as f:
            metrics = json.load(f)
            roc = metrics['roc_point']
            index = 'x'
            for axis in roc:
                for k, v in axis.items():
                    cls_series = classes.get(k, {'x': [], 'y': []})
                    cls_series[index].append(v)
                    classes[k] = cls_series
                index = 'y'

    for k in classes.keys():
        fig, ax = plt.subplots()
        ax.plot(classes[k]['x'], classes[k]['y'], 'bo')

        ax.set(xlabel='false positive rate', ylabel='false negative rate',
            title='ROC graph for {} class'.format(k))
        ax.grid()

        fig.savefig("{}.png".format(k))
        plt.show()

if __name__ == '__main__':
    main()



