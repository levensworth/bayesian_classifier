import pandas as pd
from net import BayesianNetwork, Dag, Node


def get_data(path):
    df = pd.read_csv(path)
    for index in range(len(df)):
        val = df.at[index, 'gre']
        df.at[index, 'gre'] = 0 if val <= 500 else 1
        val = df.at[index, 'gpa']
        df.at[index, 'gpa'] = 0 if val <= 3.0 else 1

    return df

rank = Node('srank', None, [1, 2, 3, 4])
gre = Node('gre', None, [0, 1])
gpa = Node('gpa', None, [0,1])
accepted = Node('admit', None, [0, 1])

dag = Dag()
dag.add_node(rank)
dag.add_node(gre)
dag.add_node(gpa)
dag.add_node(accepted)

dag.add_vertex(rank, gre)
dag.add_vertex(rank, gpa)
dag.add_vertex(gre, accepted)
dag.add_vertex(gpa, accepted)
dag.add_vertex(rank, accepted)

net = BayesianNetwork(dag)

df = get_data('data/inscriptions.csv')

net.train(df)

rank_1 = Node('srank', 2, [2])
gre_1 = Node('gre', 0, [0])
gpa_1 = Node('gpa', 1, [1])

accepted = Node('admit', 1, [1])

result = net.infer([accepted], [rank_1, gpa_1, gre_1])

print('[RESULT] {} %'.format(result))

# E1) 0.6821215806194527
# E2) 0.3137895144997041

