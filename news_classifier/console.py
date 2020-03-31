from classifier_2 import CATEGORIES, NORM_PERCENTAGES, EPSILON, load_data, normalize_data, NaiveBayes, train_test_split
import matplotlib.pyplot as plt
import json
import pprint


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

    while True:
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



