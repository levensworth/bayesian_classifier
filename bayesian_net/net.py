import numpy as np
from dataclasses import field, dataclass
import pandas as pd
import hashlib

class BayesianNetwork(object):
    def __init__(self, dag):
        """
        A Network takes into account a dag = directed acyclic graph which
        constitutes the relationship between each variable
        """
        self.strucutre = dag
        self.probabilities = {}

    def train(self, df):
        """
        Using a dataframe as input we can calculate the probability for any given node
        using the dag hereachy.
        """
        # first we analyze independant variables
        nodes_to_analyze = self.strucutre.get_independent_nodes()
        for node in nodes_to_analyze:
            self.probabilities['{}'.format(node.key)] = self.calculate_probability(df, node)

        visited = nodes_to_analyze

        while len(visited) < len(self.strucutre.get_nodes()):
            length = len(visited)
            for index in range(length):
                node = visited[index]
                children = set(self.strucutre.get_children(node))
                visited_set = set(visited)
                children = children - visited_set
                for child in children:
                    parents = set(self.strucutre.get_parents(child))
                    if parents.issubset(visited_set):
                        # all parents have been properly calculated
                        self.probabilities['{}'.format(child.key)] = self.calculate_conditional_probability(df, child, parents)
                        visited.append(child)

        return True

    def calculate_probability(self, df, node):
        '''This returns P[node] which could be think as # positive cases / # cases '''
        probabilites = {}
        total_cases = node.universe

        for case in total_cases:
            positive_cases = len(df.query('{} == {}'.format(node.key, case)))
            probabilites['{}={}'.format(node.key, case)] = positive_cases / float(len(df))
        return probabilites

    def calculate_conditional_probability(self, df, node, parents):
        '''In this case we calculate the probability of a node give all the parents'''
        # probabilities is a dict of dicts
        # where each possible value of the node is a fist level dict
        # de sect contains the probability for each configuration
        probabilities = {}
        universe_size = len(df)
        for value in node.universe:
            # selected_cases = df[getattr(df, node.key) == value]
            node.value = value
            probabilities['{}'.format(value)] = self.recursive_search_parents_value(df, node, parents.copy(), universe_size)
        return probabilities

    def recursive_search_parents_value(self, df, child, parents, univers_size, case_key=None):
        parent = parents.pop()
        # this was a dataframe
        probabilities = {}
        if len(parents) == 0:
            # this gives you the amount of cases which satisfies all restrictions
            key = case_key
            for last_val in parent.universe:
                cases = df[getattr(df, parent.key) == last_val]
                selected_cases = len(cases[getattr(df, child.key) == child.value])
                cases = len(cases)
                if case_key != None:
                    key = '{},{}={}'.format(case_key, parent.key, last_val)
                else:
                    key = '{}={}'.format(parent.key, last_val)

                prob = float(selected_cases + 1) / (univers_size + len(child.universe))

                probabilities[key] = prob
            return probabilities

        key = case_key
        for parent_value in parent.universe:
            filter_df = df[getattr(df, parent.key) == parent_value]
            if case_key != None:
                key = '{},{}={}'.format(case_key, parent.key, parent_value)
            else:
                key = '{}={}'.format(parent.key, parent_value)

            probabilities.update(self.recursive_search_parents_value(filter_df, child, parents.copy(), univers_size, key))

        return probabilities

    def get_convinations(self, variables_set, convinantion=None):
        '''
        This function creates all possible convinations with the given restrictions to each node
        returns a set of lists where each list represent a convinantion
        '''
        if convinantion == None:
            convinantion = []

        convinantions = []
        var = variables_set.pop()
        if len(variables_set) == 0:
            for val in var.universe:
                new_convinantion = convinantion.copy()
                new_convinantion.append(Node(var.key, val, [val]))
                convinantions.append(new_convinantion)
            return convinantions

        for val in var.universe:
            new_convinantion = convinantion.copy()
            new_convinantion.append(Node(var.key, val, [val]))
            convinantions += self.get_convinations(variables_set.copy(), new_convinantion)

        return convinantions

    def calculate_inference_probabilities(self, convinations):
        prob = 0.0

        for conv in convinations:
            conv_probability = self.calculate_conv_probability(conv)
            prob += conv_probability

        return prob

    def calculate_conv_probability(self, convination):
        prob = 1.0
        # first get the independant variables prob
        nodes = set(convination)
        dep_nodes = nodes - (set(self.strucutre.get_independent_nodes()))
        indep_nodes = nodes - dep_nodes

        for node in indep_nodes:
            indep_prob = self.probabilities.get('{}'.format(node.key)).get('{}={}'.format(node.key, node.value), None)
            if indep_prob == None:
                print('[ERROR] you passed value {} for node {}, that is not in the universe of posibilities'
                .format(node.value, node.key))
                raise AttributeError
            prob *= indep_prob

        for node in dep_nodes:
            no_parents = set(convination) - (set(self.strucutre.get_parents(node)))
            parents = set(convination) - no_parents

            cases = self.probabilities.get('{}'.format(node.key)).get('{}'.format(node.value))
            dep_prob = 0.0
            for case, val in cases.items():
                is_valid_case = True
                while len(parents) > 0 and is_valid_case:
                    parent = parents.pop()
                    if not('{}={}'.format(parent.key, parent.value) in case):
                        is_valid_case = False

                if is_valid_case:
                    dep_prob += val
            prob *= dep_prob

        return prob

    def infer(self, ask_vec, given_vec):
        '''Returns the given probabilty
        ask_vec: a vector of nodes where you want to know the probabilty of that node having that value
        give_vec: a vector of nodes with the actual values a prior
        you can think of this as :
        returns P[ask_vec | given_vec] = sum(P[ask_vec, given_vec, free_var]) over the free_var
                                            / sum(P[free_var, given_vec]) over free_var + ask_vec
        '''
        restrictions = set(ask_vec)
        restrictions.update(set(given_vec))
        node_set = set(self.strucutre.get_nodes())
        restrictions.update(node_set)

        convinations_for_numerator = self.get_convinations(restrictions)

        numerator_prob = self.calculate_inference_probabilities(convinations_for_numerator)

        restrictions = set(given_vec)
        restrictions.update(set(self.strucutre.get_nodes()))
        convinations_for_denominator = self.get_convinations(restrictions)

        denominator_prob = self.calculate_inference_probabilities(convinations_for_denominator)

        return numerator_prob / denominator_prob

class Dag(object):
    """
    object representation of a directed acyclic graph
    """

    def __init__(self):
        self.nodes = {}
        self.vertexes = {}

    def add_node(self, node):
        """
        A node should implement two methods, get_key() and get_value() and shoul be hashable
        """
        self.nodes[node.key] = node

    def add_vertex(self, from_node, to_node):
        """
        A vertex should implement from_node() and to() and should be hashable
        """
        vertex = Vertex(from_node, to_node)
        self.nodes[from_node.key].out_vertexes.append(vertex)
        self.nodes[to_node.key].in_vertexes.append(vertex)
        self.vertexes[vertex] = vertex

    def get_node(self, node_key):
        '''If no node found the returns None'''
        return self.nodes.get(node_key, None)

    def get_nodes(self):
        return self.nodes.values()

    def get_independent_nodes(self):
        nodes = []
        for node in self.nodes.values():
            if len(node.in_vertexes) == 0:
                nodes.append(node)

        return nodes

    def get_children(self, node):
        children = []
        for vertex in self.vertexes:
            if vertex.from_node == node:
                children.append(vertex.to_node)
        return children

    def get_parents(self, node):
        '''Returns a list of first degree parents '''
        parents = set()
        for vertex in self.vertexes.keys():
            if vertex.to_node == node:
                parents.add(vertex.from_node)

        return parents


@dataclass(eq=True)
class Node:
    key: str = field(compare=True, hash=True)
    value: float
    universe: list
    in_vertexes: list = field(default_factory=list)
    out_vertexes: list = field(default_factory=list)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, value):
        try:
            return self.key == value.key
        except Exception:
            return False

@dataclass(eq=True)
class Vertex:

    from_node: Node = field(compare=True, hash=True)
    to_node: Node = field(compare=True, hash=True)

    def __hash__(self):
        return 17 * self.from_node.__hash__() + 53 * self.to_node.__hash__()

# tests

def given_a_node():
    return Node('a', 0.1)

def given_another_node():
    return Node('b', 0.4)

def when_vertex(from_node, to_node):
    return Vertex(from_node, to_node)


def test_vertex():
    node1 = given_a_node()
    node2 = given_another_node()
    vertex = when_vertex(node1, node2)
    assert vertex.from_node == node1
    assert vertex.to_node ==  node2


def test_node():
    node = given_a_node()
    assert node.key == 'a'
    assert node.value == 0.1




