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

    def __init__(self, df, epsilon, decision_boundary):
        self.df = df
        tokens, vocabulary = self.tokenize(df)
        self.tokens = tokens
        self.vocabulary = vocabulary
        self.epsilon = epsilon
        self.decision_boundary = decision_boundary

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

        for word in self.vocabulary:

            probabilities['{}'.format(word)] = self.calculate_conditional_word_vec(word, categories)

        return probabilities

    def calculate_conditional_word_vec(self, word, categories):
        '''
        calculate the probabilty of P[word | category]
        '''
        prob = {}
        for category in categories:
            category_word_count = len(self.tokens.get(category, dict()).values())
            word_count = self.tokens.get(category, dict()).get(word, 0)
            prob[category] = (word_count + 1) / float(category_word_count + len(self.vocabulary) )

        return prob
    def train(self, categories):
        self.categories = categories
        self.categories_prob = self.calculate_categories_prob(categories)
        self.conditional_prob = self.calculate_conditional_prob(categories)


    def get_sentce_probability(self, sentece_vec):
        '''Calculate the probabilty of a given sentece to exist'''
        prob = 1

        for word in sentece_vec:
            word_prob = 1/len(self.vocabulary) # prob de una palabra nueva que nunca vi
            for prob_cat in self.conditional_prob.get(word.lower(), dict()).values():
                word_prob += prob_cat
            prob *= word_prob
        return prob

    def infer(self, sentece):
        vec = sentece.split(' ')
        inference = {}
        sent_prob = self.get_sentce_probability(vec)
        for k, v in self.categories_prob.items():
            inference[k] = float(v)
            sentence_prob = 1
            for word in vec:
                # sentence_prob *= float(self.conditional_prob.get('{}|{}'.format(word.lower(), k), 1/len(self.vocabulary)))
                sentence_prob *= float(self.conditional_prob.get('{}'.format(word.lower()), dict()).get(k, 1/len(self.vocabulary)))

            inference[k] *= sentence_prob
            inference[k] /=  sent_prob

        # normalize values to represent a probability

        total_prob = sum(inference.values())
        for k in inference.keys():
            inference[k] /= total_prob

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
        fn -= matrix[index, index]

        # calculate false positive
        for i in range(len(matrix)):
            fp += matrix[i][index]

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
           'fp_rate':  fp/ float(fp + tn),
           'f1': (2.0 * presicion * recall)/ (presicion + recall)
        }
        return metrics

    def test(self, sentences, categories, boundary):
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
            max_class = max(inference.items(), key=operator.itemgetter(1))
            if max_class[1] > boundary:
                predicted.append(max_class[0])
            else:
                predicted.append('NONE')

        # calculate confussion_matrix
        possible = CATEGORIES + ['NONE']
        matrix = confusion_matrix(categories, predicted, labels=possible)
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


def show_train(classifier, test_data, boundary, metric_path):
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

    conf_matrix, metrics = classifier.test(sentences, categories, boundary)

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
    classifier = NaiveBayes(train_data, EPSILON, 0.3)
    # show results
    for i in [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.95]:
        show_train(classifier, test_data, i, 'metrics_{}.json'.format(i))

    while False:
        title = input('enter a title: ')
        result = classifier.infer(title)
        print('results')
        pprint.pprint(result)


def plot_graphs( paths = ['metrics_0.1.json', 'metrics_0.2.json', 'metrics_0.3.json', 'metrics_0.4.json', 'metrics_0.5.json', 'metrics_0.6.json', 'metrics_0.7.json', 'metrics_0.75.json', 'metrics_0.8.json', 'metrics_0.85.json', 'metrics_0.95.json']):

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
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        ax.plot(classes[k]['x'], classes[k]['y'])

        ax.set(xlabel='false positive rate', ylabel='true positive rate',
            title='ROC graph for {} class'.format(k))
        ax.grid()

        fig.savefig("{}.png".format(k))
        plt.show()

if __name__ == '__main__':
    main()
    plot_graphs()



