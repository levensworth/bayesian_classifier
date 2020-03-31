import pandas as pd
import numpy as np

#  configs
DATA_PATH = './data/PreferenciasBritanicos.csv'
LABEL_COL = 'Nacionalidad'

# TODO: we are assuming that all the possible labels are given in the train set
# thus, if you only give positive examples the classifier will never yeld a negative.
# As it doesn't know the existance of that label.

def calculate_class_frequency(df) :
    # procedure:
    # calculate the relative frequency for each class
    # remeber the class (y value) is the last column
    class_frequency = {}
    total_cases  = len(df[LABEL_COL])

    for person in df[LABEL_COL]:
        class_frequency[person] = (class_frequency.get(person, 0) + 1) / total_cases

    return class_frequency, total_cases

def calculate_var_relative_frequencies(df, class_frequency):
    # we need to calculate the relative frequency for each var
    var_frequencies = {}

    correction_needed = False
    for label in class_frequency.keys():
        population = df[getattr(df, LABEL_COL) == label]
        population_size = len(population)
        # for the queried population, calculate each var frecquency
        for col in population.columns:
            if not(col == LABEL_COL):
                var_frequencies['{}|{}'.format(col, label)] =  len(population.query('{} == 1'.format(col))) / population_size
                correction_needed = True if var_frequencies['{}|{}'.format(col, label)] == 0 else False

# if everything is okey and no probability is 0 we return
    if not correction_needed:
        return var_frequencies

# we calculate the var frequency applying the laplace correction
    var_frequencies = {}

    for label in class_frequency.keys():
        population = df[getattr(df, LABEL_COL) == label]
        population_size = len(population)
        n_classes = len(population.columns) # represents all the cardinal of the attribute universe for the given class
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
    # x_prob = 1
    # for prob in var_prob:
    #     x_prob *= var_prob[prob] if x[prob] == 1 else (1 - var_prob[prob])

    # now we calclate the probability for each label given X
    for label in posterior_prob.keys():
        for col in df.columns:
            if col != LABEL_COL and x[col] == 1:
                posterior_prob[label] *=  var_frequencies['{}|{}'.format(col, label)]

        posterior_prob[label] *= class_frequency[label]

        # posterior_prob[label] /= x_prob
    return posterior_prob

def main(X):
    # get the classes_frequency:
    data_frame = pd.read_csv(DATA_PATH)
    class_frequency, total_population = calculate_class_frequency(data_frame)
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

    x['scones'] = bool(input('Do you like scones? '))
    x['cerveza'] = bool(input('Do you like beer? '))
    x['wiskey'] = bool(input('Do you like Whiskey? '))
    x['avena'] = bool(input('Do you like oatmeal? '))
    x['futbol'] = bool(input('Do you like football? '))

    main(x)
