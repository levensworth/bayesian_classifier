import pandas as pd
import numpy as np

#  configs
DATA_PATH = './data/PreferenciasBritanicos.csv'
LABEL_COL = 'Nacionalidad'

# TODO: we are assuming that all the possible labels are given in the train set
# thus, if you only give positive examples the classifier will never yeld a negative.
# As it doesn't know the existance of that label.

def calculate_class_frequency(df, category_col, categories) :
    # procedure:
    # calculate the relative frequency for each class
    # remeber the class (y value) is the last column
    class_frequency = {}
    total_cases  = len(df)

    for category in categories:
        class_frequency[category] = len(df[df[category_col] == category]) / float(total_cases)

    return class_frequency, total_cases

def calculate_var_relative_frequencies(df, class_frequency):
    # we need to calculate the relative frequency for each var
    var_frequencies = {}

    # we calculate the var frequency applying the laplace correction

    for label in class_frequency.keys():
        population = df[getattr(df, LABEL_COL) == label]
        population_size = len(population)
        n_classes = 2 # represents all the cardinal of the attribute universe for the given class
        # for the queried population, calculate each var frecquency
        for col in population.columns:
            if not(col == LABEL_COL):
                var_frequencies['{}|{}'.format(col, label)] = (len(population.query('{} == 1'.format(col))) + 1) / (population_size + n_classes)

    return var_frequencies

def calculate_var_prob(df, total_cases):
    # calculate the frequency for each variabel, given the training set
    var_prob = {}
    for col in df.columns:
        if not(col == LABEL_COL):
            var_prob[col] = len(df.query('{} == 1'.format(col))) / total_cases
    return var_prob

def run_inference(df, x, class_frequency, var_frequencies, var_prob):
    # init posterior prob
    posterior_prob = {}
    for label in class_frequency.keys():
        posterior_prob[label] = 1

    # probabily of the given situation even happening
    x_prob = 1
    for prob in var_prob:
        x_prob *= var_prob[prob] if x[prob] == 1 else (1 - var_prob[prob])

    # now we calclate the probability for each label given X
    for label in posterior_prob.keys():
        for col in df.columns:
            if col != LABEL_COL:
                prob_true = var_frequencies['{}|{}'.format(col, label)]
                posterior_prob[label] *=   prob_true if x[col] == 1 else 1 - prob_true

        posterior_prob[label] *= class_frequency[label]

        posterior_prob[label] /= x_prob
    return posterior_prob

def main(X):
    # get the classes_frequency:
    data_frame = pd.read_csv(DATA_PATH)
    class_frequency, total_population = calculate_class_frequency(data_frame, 'Nacionalidad', ['I', 'E'])
    var_frequencies = calculate_var_relative_frequencies(data_frame, class_frequency)
    var_prob = calculate_var_prob(data_frame, total_population)
    posterior_prob = run_inference(data_frame, X, class_frequency, var_frequencies, var_prob)

    print('probabilities:')
    print(posterior_prob)
    if posterior_prob['I'] < posterior_prob['E']:
        print("For the Great Scottland!")
    else:
        print("You're talking to a true gentleman")


if __name__ == '__main__':
    x = dict()

    x['scones'] = int(input('Do you like scones? '))
    x['cerveza'] = int(input('Do you like beer? '))
    x['wiskey'] = int(input('Do you like Whiskey? '))
    x['avena'] = int(input('Do you like oatmeal? '))
    x['futbol'] = int(input('Do you like football? '))

    main(x)

# 1,0,1,0,0,E
# 0,0,0,1,0,I
'''

p[scone|I] = 3/6 = 0.5
p[cerveza|I] = 3/6 = 0.5
p[wiskey|I] = 2/6 = 1/3
p[avena|I] = 3/6 = 0.5
p[futbol|I] = 3/6 = 0.5

p[I] = 6/13
X = [0 0 0 1 1]

P(I/ X) = 
'''
