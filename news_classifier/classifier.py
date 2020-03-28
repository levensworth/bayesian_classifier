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

    for category in cateogires.keys():
        category_series = df[getattr(df, axis_name) == category]
        percentage = cateogires[category]
        # we make sure the percentage is correct
        percentage =  percentage if abs(percentage) < 1 else 1

    raise NotImplemented()
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
    as :
        cols:   buenos  aires  android  category ...
    rows:
        0        2        3       1       news
    '''
    tokens = pd.DataFrame()
    # generate the vocabulary for the hole train set
    vocabulary = set()
    for index, row in df.iterrows():
        title = row['titular']
        title_vec = title.split(' ')
        for word in title_vec:
            vocabulary.add(word.lower())

    for index, row in df.iterrows():
        title = row['titular']
        title_vec = title.split(' ')
        title_df = {'category': [row['categoria']]}
        for word in title_vec:
            title_df[word.lower()] = [title_df.get(word.lower(), [0])[0] + 1]
        title_df = pd.DataFrame(title_df)
        tokens = pd.concat([tokens, title_df])
    tokens = tokens.fillna(0)
    return tokens, vocabulary

def calculate_categories_prob(tokens, categories):

    probabilities = {}
    train_size = len(tokens)
    for category in categories:
        count = len(tokens[getattr(tokens,'category') == category])
        probabilities[category] = count / float(train_size)

    return probabilities


def calculate_conditional_prob(tokens, vocabulary, categories):
    probabilities = {}
    for category in categories:
        category_word_count = tokens.loc[tokens['category'] == category]
        category_word_count = category_word_count.sum(axis=1).sum()
        for word in vocabulary:
            probabilities['{}|{}'.format(word, category)] = calculate_conditional_word_vec(tokens, vocabulary, word, category_word_count, category)

    return probabilities


def calculate_conditional_word_vec(tokens, vocabulary, word, category_word_count, category):
    '''
    calculate the probabilty of P[word | category]
    '''
    category_examples = tokens[getattr(tokens, 'category') == category]
    word_count = category_examples[word].sum()
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
    # norm_data = normalize_data(data, 'categoria', categories)
    norm_data = data[:5000]
    tokens, vocabulary = tokenize(norm_data, None)
    categories = ['Internacional','Nacional',
    'Destacadas','Deportes','Salud','Ciencia y Tecnologia',
    'Entretenimiento','Economia','Noticias destacadas']
    conditional_prob = calculate_conditional_prob(tokens, vocabulary, categories)
    categories_prob = calculate_categories_prob(tokens, categories)
    title = 'En la Noche de las Helader�as Artesanales, 7 lugares para disfrutar de los mejores sabores' # es Nacional
    title_vec = title.split(' ')
    inference = infer(conditional_prob, categories_prob, title_vec)
    max_class = max(inference.items(), key=operator.itemgetter(1))[0]
    print(max_class)

if __name__ == '__main__':
    main()

