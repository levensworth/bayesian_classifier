import pandas as pd
import numpy as np
import random
import operator

def load_data(path, separator=',', encoding='ISO-8859-1'):
    df = pd.read_csv(path, sep= separator, encoding=encoding)
    return df


def normalize_data(df, axis_name, cateogires):
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
    for category in cateogires.keys():
        items_indexes = df[getattr(df, axis_name) == category].index.tolist()
        percentage = cateogires[category]
        # we make sure the percentage is correct
        percentage =  percentage if abs(percentage) < 1 else 1
        percentage *= len(items_indexes)
        selected = random.choices(items_indexes, k= int(percentage))
        category_df = df.query('index in {}'.format(selected))
        norm_data = pd.concat([norm_data, category_df])


    print(norm_data.head())
    return df


def create_model(df, classes):
    '''
    df: data frame
    classes: a list of values corresponding to actual output classses
    '''
    token_df = tokenize(df)


def tokenize(df, attr):
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

def calculate_categories_prob(df, categories):

    probabilities = {}
    train_size = len(df)
    for category in categories:
        count = len(df[getattr(df,'categoria') == category])
        probabilities[category] = count / float(train_size)

    return probabilities


def calculate_conditional_prob(tokens, vocabulary, categories):
    probabilities = {}
    for category in categories:
        category_word_count = sum(tokens.get(category, dict()).values())
        for word in vocabulary:
            probabilities['{}|{}'.format(word, category)] = calculate_conditional_word_vec(tokens, vocabulary, word, category_word_count, category)

    return probabilities


def calculate_conditional_word_vec(tokens, vocabulary, word, category_word_count, category):
    '''
    calculate the probabilty of P[word | category]
    '''
    word_count = tokens.get(category, dict()).get(word, 0)
    return (word_count + 1) / float(category_word_count + len(vocabulary))


def infer(conditional_prob, class_prob, vec):
    inference = {}
    for k, v in class_prob.items():
        inference[k] = float(v)
        for word in vec:
            # extract epsilon
            inference[k] *= float(conditional_prob.get('{}|{}'.format(word.lower(), k), 0.0001))
    return inference


def main():
    path = './data/Noticias_argentinas.csv'

    data = load_data(path, ';')
    categories = {
        'Internacional': 1,
        'Nacional': 1,
        'Destacadas': 1,
        'Deportes': 1,
        'Salud':1,
        'Ciencia y Tecnologia': 1,
        'Entretenimiento': 1,
        'Economia': 1,
        'Noticias destacadas': 0.028
    }
    norm_data = normalize_data(data, 'categoria', categories)
    # norm_data = data[:10000]
    tokens, vocabulary = tokenize(norm_data, None)
    categories = ['Internacional','Nacional',
    'Destacadas','Deportes','Salud','Ciencia y Tecnologia',
    'Entretenimiento','Economia']
    conditional_prob = calculate_conditional_prob(tokens, vocabulary, categories)
    categories_prob = calculate_categories_prob(norm_data, categories)
    title = 'Una crisis militar enrarece la relaci�n entre Occidente y Rusia antes del G-20' # es Internacional
    title_vec = title.split(' ')
    inference = infer(conditional_prob, categories_prob, title_vec)
    max_class = max(inference.items(), key=operator.itemgetter(1))[0]
    print(max_class)

if __name__ == '__main__':
    main()

