#!/usr/bin/env python
# coding: utf-8

from operator import itemgetter, and_, add
from collections import Counter
from functools import reduce

import networkx as nx
import numpy as np


class closeness_centrality_estimation:

    def __init__(self):
        self.edges_path = '.'
        self.network_graph = nx.Graph()
        self.pivot_number = 1000
        self.epsilon = 0.01
        self.weight_name = 'weight'
        self.graph_type = 'undirected'
        self.edge_type = 'unweighted'
        self.centrality_type = 'normal'
        self.pivots = []
        self.pivots_as_source_distance = {}
        self.pivots_as_target_distance = {}

    def load_data(self, path_to_data, pivot_number=1000, epsilon=0.01, graph_type='undirected', edge_type='unweighted',
                  weight_name='weight', centrality_type='normal'):

        self.edges_path = path_to_data
        self.network_graph = nx.Graph()
        self.pivot_number = pivot_number
        self.epsilon = epsilon
        self.weight_name = weight_name

        if graph_type == "directed":
            self.graph_type = 'directed'
        elif graph_type == 'undirected':
            self.graph_type = 'undirected'
        else:
            print("Unsupported graph type, only 'directed' and 'undirected' are supported.")
            raise NotImplementedError()

        if edge_type == "weighted":
            self.edge_type = 'weighted'
        elif edge_type == 'unweighted':
            self.edge_type = 'unweighted'
        else:
            print("Unsupported edge type, only 'weighted' and 'unweighted' are supported.")
            raise NotImplementedError()

        if not path_to_data:
            path_to_data = self.edges_path
        try:
            if self.graph_type == 'undirected':
                if self.edge_type == 'unweighted':
                    self.network_graph = nx.read_edgelist(path_to_data)
                else:
                    self.network_graph = nx.read_edgelist(path_to_data, data=((self.weight_name, float),))
            else:
                if self.edge_type == 'unweighted':
                    self.network_graph = nx.read_edgelist(path_to_data, create_using=nx.DiGraph())
                else:
                    self.network_graph = nx.read_edgelist(path_to_data, create_using=nx.DiGraph(),
                                                          data=((self.weight_name, float),))

        except MemoryError:
            print(
                "Cannot load data due to memory error, please split your data or use other machines with large RAM...")
        except FileNotFoundError:
            print("File not found, please check your input...")

        if not isinstance(pivot_number, int):
            raise AttributeError("Pivot number must be an integer!")
        if not isinstance(epsilon, float):
            raise AttributeError("Epsilon must be a float.")
        if pivot_number > self.network_graph.number_of_nodes():
            raise ValueError("The number of pivots (%d) is larger than the number of nodes (%d)!" % (
            pivot_number, self.network_graph.number_of_nodes()))
        if epsilon > 1 or epsilon < 0:
            raise ValueError("The value of epsilon must be within (0,1).")
        if centrality_type != 'normal' and centrality_type != 'harmonic':
            raise NotImplementedError(
                "Currently only normal closeness centrality and harmonic centrality are supported.")

        print("Data load successfully ...")
        print("Type of the network: ", self.graph_type, self.edge_type)
        if self.edge_type == 'weighted':
            print("Labels used to characterize the weight of the edges: ", self.weight_name)
            print("------ You can specify this label name when initializing the class.")
        print("Number of nodes in the network: ", self.network_graph.number_of_nodes())
        print("Number of edges in the network: ", self.network_graph.number_of_edges())

    def configure_parameters(self, pivot_number, epsilon, centrality_type='normal'):
        if self.edges_path == '.':
            raise RuntimeError("Please load the data first!")
        if not isinstance(pivot_number, int):
            raise AttributeError("Pivot number must be an integer!")
        if not isinstance(epsilon, float):
            raise AttributeError("Epsilon must be a float.")
        if pivot_number > self.network_graph.number_of_nodes():
            raise ValueError("The number of pivots (%d) is larger than the number of nodes (%d)!" % (
            pivot_number, self.network_graph.number_of_nodes()))
        if epsilon > 1 or epsilon < 0:
            raise ValueError("The value of epsilon must be within (0,1).")
        if centrality_type != 'normal' and centrality_type != 'harmonic':
            raise NotImplementedError(
                "Currently only normal closeness centrality and harmonic centrality are supported.")

        self.pivot_number = pivot_number
        self.epsilon = epsilon
        self.centrality_type = centrality_type

    def get_graph(self):
        return self.network_graph

    def __source_to_targets_distances(self, graph, node_index):
        if self.edge_type == 'unweighted':
            __dicts = nx.shortest_path_length(graph, source=node_index)
        else:
            __dicts = nx.shortest_path_length(graph, source=node_index, weight=self.weight_name)
        return __dicts

    def __targets_to_source_distances(self, graph, node_index):
        if self.edge_type == 'unweighted':
            __dicts = nx.shortest_path_length(graph, target=node_index)
        else:
            __dicts = nx.shortest_path_length(graph, target=node_index, weight=self.weight_name)
        return __dicts

    def __get_distances(self, graph, node_index, start='source', order='min'):
        if start == 'source':
            __dicts = self.__source_to_targets_distances(graph, node_index)
            if order == 'min':
                __sorted_dicts = dict(sorted(__dicts.items(), key=itemgetter(1)))
            else:
                __sorted_dicts = dict(sorted(__dicts.items(), key=itemgetter(1), reverse=True))
        else:
            __dicts = self.__targets_to_source_distances(graph, node_index)
            if order == 'min':
                __sorted_dicts = dict(sorted(__dicts.items(), key=itemgetter(1)))
            else:
                __sorted_dicts = dict(sorted(__dicts.items(), key=itemgetter(1), reverse=True))
        return __sorted_dicts

    def __log_intermediate_report(self, pivot_index, pivot_number):
        if pivot_index == 0:
            print("Start...")
        else:
            print("Progress : %.2f%%. %d of %d have been done." % (
            (pivot_index + 1) / pivot_number * 100, pivot_index + 1, pivot_number))

    def __estimate_single_closeness_centrality_networkx(self, graph, node, number_of_nodes):
        accumulate_dists = 0
        centrality = 0
        if self.edge_type == 'unweighted':
            if self.centrality_type == 'normal':
                for pivot in self.pivots:
                    if pivot == node:
                        continue
                    accumulate_dists += nx.shortest_path_length(graph, node, pivot)
                centrality = (len(self.pivots) - 1) / accumulate_dists
            elif self.centrality_type == 'harmonic':
                for pivot in self.pivots:
                    if pivot == node:
                        continue
                    accumulate_dists += nx.shortest_path_length(graph, pivot, node)
                centrality = 1 / accumulate_dists * len(self.pivots) / number_of_nodes
        else:
            if self.centrality_type == 'normal':
                for pivot in self.pivots:
                    if pivot == node:
                        continue
                    accumulate_dists += nx.shortest_path_length(graph, node, pivot, weight=self.weight_name)
                centrality = (len(self.pivots) - 1) / accumulate_dists
            elif self.centrality_type == 'harmonic':
                for pivot in self.pivots:
                    if pivot == node:
                        continue
                    accumulate_dists += nx.shortest_path_length(graph, pivot, node, weight=self.weight_name)
                centrality = 1 / accumulate_dists * len(self.pivots) / number_of_nodes
        return centrality

    def __estimate_single_closeness_centrality(self, graph, node, number_of_nodes):
        accumulate_dists = 0
        centrality = 0
        if self.centrality_type == 'normal':
            for pivot in self.pivots:
                if pivot == node:
                    continue
                accumulate_dists += self.pivots_as_source_distance[pivot][node]
            centrality = (len(self.pivots) - 1) / accumulate_dists
        elif self.centrality_type == 'harmonic':
            for pivot in self.pivots:
                if pivot == node:
                    continue
                accumulate_dists += self.pivots_as_target_distance[pivot][node]
            centrality = 1 / accumulate_dists * len(self.pivots) / number_of_nodes
        return centrality

    def estimate_centrality(self, graph, node_list='All'):
        centralities = {}
        if node_list == 'All':
            for node in graph.nodes:
                cen = self.__estimate_single_closeness_centrality(graph, node, graph.number_of_nodes())
                centralities.update({node: cen})
        else:
            for node in node_list:
                cen = self.__estimate_single_closeness_centrality(graph, node, graph.number_of_nodes())
                centralities.update({node: cen})
        return centralities

    def compute_centrality(self, graph, node_list='All'):
        centrality = {}
        if node_list == 'All':
            if self.edge_type == 'unweighted':
                if self.centrality_type == 'normal':
                    centrality = nx.closeness_centrality(graph)
                else:
                    centrality = nx.harmonic_centrality(graph)
            else:
                if self.centrality_type == 'normal':
                    centrality = nx.closeness_centrality(graph, distance=self.weight_name)
                else:
                    centrality = nx.harmonic_centrality(graph, distance=self.weight_name)
        else:
            count = len(node_list)
            step = count % 20
            index = -1
            for node in node_list:
                if self.edge_type == 'unweighted':
                    if self.centrality_type == 'normal':
                        cen = nx.closeness_centrality(graph, node)
                    else:
                        cen = nx.harmonic_centrality(graph, node)
                else:
                    if self.centrality_type == 'normal':
                        cen = nx.closeness_centrality(graph, node, distance=self.weight_name)
                    else:
                        cen = nx.harmonic_centrality(graph, node, distance=self.weight_name)
                centrality.update({node: cen})
                index += 1
                if index % step:
                    self.__log_intermediate_report(index, count)
        return centrality

    # Need to complete later
    def compute_confidence(self, diameter):
        pass
        # node_count = self.network_graph.number_of_edges()
        # M = node_count / (node_count - 1) * diameter
        # Xi = self.epsilon * diameter
        # error = np.exp(-2 *self.pivot_number*(self.epsilon*(node_count-)))

    def RandomSample(self, graph='original', start='source'):
        if graph == 'original':
            graph = self.network_graph
        pivots = list(np.random.choice(graph.nodes, self.pivot_number))
        previous_pivots_source_dists = {}
        previous_pivots_target_dists = {}
        count = self.pivot_number/20
        for pivot_index in range(self.pivot_number):
            source_dist_dicts = self.__get_distances(graph, pivots[pivot_index], start='source')
            target_dist_dicts = self.__get_distances(graph, pivots[pivot_index], start='target')
            previous_pivots_source_dists.update({pivots[pivot_index]: source_dist_dicts})
            previous_pivots_target_dists.update({pivots[pivot_index]: target_dist_dicts})
            if pivot_index%count == 0:
                self.__log_intermediate_report(pivot_index, self.pivot_number)
        print("Successfully sampled %d pivots from the network based on MaxMin approach..." % self.pivot_number)
        self.pivots = pivots
        self.pivots_as_source_distance = previous_pivots_source_dists
        self.pivots_as_target_distance = previous_pivots_target_dists
        return pivots

    def RandomDegree(self, graph='original', start='source'):
        degree_distribution = []
        if graph == 'original':
            graph = self.network_graph
        if isinstance(graph, nx.DiGraph):
            if start == 'source':
                degree_distribution = graph.in_degree
            else:
                degree_distribution = graph.out_degree
        elif isinstance(graph, nx.Graph):
            degree_distribution = graph.degree
        else:
            raise NotImplementedError("Currently only pure directed and undirected graphs are supported")
        sum = np.sum(list(dict(degree_distribution).values()))
        probability_distribution = [i / sum  for i in list(dict(degree_distribution).values())]
        pivots = np.random.choice(graph.nodes, size=self.pivot_number, p=probability_distribution)
        previous_pivots_source_dists = {}
        previous_pivots_target_dists = {}
        count = self.pivot_number/20
        for pivot_index in range(self.pivot_number):
            source_dist_dicts = self.__get_distances(graph, pivots[pivot_index], start='source')
            target_dist_dicts = self.__get_distances(graph, pivots[pivot_index], start='target')
            previous_pivots_source_dists.update({pivots[pivot_index]: source_dist_dicts})
            previous_pivots_target_dists.update({pivots[pivot_index]: target_dist_dicts})
            if pivot_index%count == 0:
                self.__log_intermediate_report(pivot_index, self.pivot_number)
        print("Successfully sampled %d pivots from the network based on MaxMin approach..." % self.pivot_number)
        self.pivots = pivots
        self.pivots_as_source_distance = previous_pivots_source_dists
        self.pivots_as_target_distance = previous_pivots_target_dists

    # Self invented
    def MaxDist(self, graph='original', start='source'):
        if graph == 'original':
            graph = self.network_graph
        start_point = np.random.choice(graph.nodes)
        pivots = []
        previous_pivots_dists = {}
        longest_distance = {}
        count = self.pivot_number / 20
        next_pivot = 'None'
        pivots.append(start_point)
        for pivot_index in range(self.pivot_number):
            dicts = self.__get_distances(graph, start_point, start, order='max')
            previous_pivots_dists.update({start_point: dicts})
            if start_point in previous_pivots_dists[start_point].keys():
                del previous_pivots_dists[start_point][start_point]
            element = (list(previous_pivots_dists[str(start_point)].keys())[0],
                       previous_pivots_dists[str(start_point)].pop(
                           list(previous_pivots_dists[str(start_point)].keys())[0]))
            while element[0] in pivots:
                element = (list(previous_pivots_dists[str(start_point)].keys())[0],
                           previous_pivots_dists[str(start_point)].pop(
                               list(previous_pivots_dists[str(start_point)].keys())[0]))
            longest_distance.update({start_point: element})
            sorted_test_dist = sorted(longest_distance, key=lambda v: longest_distance[v][1], reverse=True)
            for key in sorted_test_dist:
                if longest_distance[key][0] in pivots:
                    continue
                else:
                    next_pivot = longest_distance[key][0]
                    break
            if next_pivot == 'None':
                raise RuntimeError("Cannot find enough pivots! Please try with smaller number of pivots.")
            for key in previous_pivots_dists.keys():
                if next_pivot in previous_pivots_dists[key].keys():
                    del previous_pivots_dists[key][next_pivot]
            start_point = next_pivot
            pivots.append(start_point)
            for key in longest_distance.keys():
                if longest_distance[key][0] in pivots:
                    element = (list(previous_pivots_dists[key].keys())[0],
                               previous_pivots_dists[key].pop(
                                   list(previous_pivots_dists[key].keys())[0]))
                    while element[0] in pivots:
                        element = (list(previous_pivots_dists[key].keys())[0],
                                   previous_pivots_dists[key].pop(
                                       list(previous_pivots_dists[key].keys())[0]))
                    longest_distance.update({key: element})
            if pivot_index % count == 0:
                self.__log_intermediate_report(pivot_index, self.pivot_number)
        print("Successfully sampled %d pivots from the network based on MaxMin approach..." % self.pivot_number)
        self.pivots = pivots
        return pivots

    # Use MaxMin approach to sample the data
    def MaxMin(self, graph='original', start='source'):
        if graph == 'original':
            graph = self.network_graph
        start_point = np.random.choice(graph.nodes)
        pivots = []
        previous_pivots_source_dists = {}
        previous_pivots_target_dists = {}
        shortest_distance = {}
        count = self.pivot_number / 20
        next_pivot = 'None'
        pivots.append(start_point)
        for pivot_index in range(self.pivot_number):
            source_dist_dicts = self.__get_distances(graph, start_point, start='source')
            target_dist_dicts = self.__get_distances(graph, start_point, start='target')
            previous_pivots_source_dists.update({start_point: source_dist_dicts})
            previous_pivots_target_dists.update({start_point: target_dist_dicts})
            if start == 'source':
                # previous_pivots_dists = previous_pivots_source_dists
                if pivot_index == 0:
                    shortest_distance = source_dist_dicts
                else:
                    combined_distance = {"previous": shortest_distance, "current": source_dist_dicts}
                    shortest_distance = dict(reduce(and_, map(Counter, combined_distance.values())))
            else:
                # previous_pivots_dists = previous_pivots_target_dists
                if pivot_index == 0:
                    shortest_distance = target_dist_dicts
                else:
                    combined_distance = {"previous": shortest_distance, "current": target_dist_dicts}
                    shortest_distance = dict(reduce(and_, map(Counter, combined_distance.values())))
            # if start_point in previous_pivots_dists[start_point].keys():
            #     del previous_pivots_source_dists[start_point][start_point]
            sorted_shortest_dist = sorted(shortest_distance, key=lambda v: shortest_distance[v], reverse=True)
            for key in sorted_shortest_dist:
                if key in pivots:
                    continue
                else:
                    next_pivot = key
                    break
            if next_pivot == 'None':
                raise RuntimeError("Cannot find enough pivots! Please try with smaller number of pivots.")
            #             for key in previous_pivots_dists.keys():
            #                 if next_pivot in previous_pivots_dists[key].keys():
            #                     del previous_pivots_dists[key][next_pivot]
            start_point = next_pivot
            pivots.append(start_point)
            if pivot_index % count == 0:
                self.__log_intermediate_report(pivot_index, self.pivot_number)
        print("Successfully sampled %d pivots from the network based on MaxMin approach..." % self.pivot_number)
        source_dist_dicts = self.__get_distances(graph, start_point, start='source')
        target_dist_dicts = self.__get_distances(graph, start_point, start='target')
        previous_pivots_source_dists.update({start_point: source_dist_dicts})
        previous_pivots_target_dists.update({start_point: target_dist_dicts})
        self.pivots = pivots
        self.pivots_as_source_distance = previous_pivots_source_dists
        self.pivots_as_target_distance = previous_pivots_target_dists
        return pivots

    def MaxSum(self, graph='original', start='source'):
        if graph == 'original':
            graph = self.network_graph
        start_point = np.random.choice(graph.nodes)
        pivots = []
        previous_pivots_source_dists = {}
        previous_pivots_target_dists = {}
        previous_pivots_dists = {}
        sum_distance = {}
        count = self.pivot_number / 20
        next_pivot = 'None'
        pivots.append(start_point)
        for pivot_index in range(self.pivot_number):
            source_dist_dicts = self.__get_distances(graph, start_point, start='source')
            target_dist_dicts = self.__get_distances(graph, start_point, start='target')
            previous_pivots_source_dists.update({start_point: source_dist_dicts})
            previous_pivots_target_dists.update({start_point: target_dist_dicts})
            if start == 'source':
                # previous_pivots_dists = previous_pivots_source_dists
                if pivot_index == 0:
                    sum_distance = source_dist_dicts
                else:
                    combined_distance = {"previous": sum_distance, "current": source_dist_dicts}
                    sum_distance = dict(reduce(add, map(Counter, combined_distance.values())))
            else:
                # previous_pivots_dists = previous_pivots_target_dists
                if pivot_index == 0:
                    sum_distance = target_dist_dicts
                else:
                    combined_distance = {"previous": sum_distance, "current": target_dist_dicts}
                    sum_distance = dict(reduce(and_, map(Counter, combined_distance.values())))
            # if start_point in previous_pivots_dists[start_point].keys():
            #     del previous_pivots_source_dists[start_point][start_point]
            sorted_sum_dist = sorted(sum_distance, key=lambda v: sum_distance[v], reverse=True)
            for key in sorted_sum_dist:
                if key in pivots:
                    continue
                else:
                    next_pivot = key
                    break
            if next_pivot == 'None':
                raise RuntimeError("Cannot find enough pivots! Please try with smaller number of pivots.")
            #             for key in previous_pivots_dists.keys():
            #                 if next_pivot in previous_pivots_dists[key].keys():
            #                     del previous_pivots_dists[key][next_pivot]
            start_point = next_pivot
            pivots.append(start_point)
            if pivot_index % count == 0:
                self.__log_intermediate_report(pivot_index, self.pivot_number)
        print("Successfully sampled %d pivots from the network based on MaxSum approach..." % self.pivot_number)
        source_dist_dicts = self.__get_distances(graph, start_point, start='source')
        target_dist_dicts = self.__get_distances(graph, start_point, start='target')
        previous_pivots_source_dists.update({start_point: source_dist_dicts})
        previous_pivots_target_dists.update({start_point: target_dist_dicts})
        self.pivots = pivots
        self.pivots_as_source_distance = previous_pivots_source_dists
        self.pivots_as_target_distance = previous_pivots_target_dists
        return pivots, previous_pivots_dists

    def MinSum(self, graph='original', start='source'):
        if graph == 'original':
            graph = self.network_graph
        start_point = np.random.choice(graph.nodes)
        pivots = []
        previous_pivots_source_dists = {}
        previous_pivots_target_dists = {}
        previous_pivots_dists = {}
        sum_distance = {}
        count = self.pivot_number / 20
        next_pivot = 'None'
        pivots.append(start_point)
        for pivot_index in range(self.pivot_number):
            source_dist_dicts = self.__get_distances(graph, start_point, start='source')
            target_dist_dicts = self.__get_distances(graph, start_point, start='target')
            previous_pivots_source_dists.update({start_point: source_dist_dicts})
            previous_pivots_target_dists.update({start_point: target_dist_dicts})
            if start == 'source':
                # previous_pivots_dists = previous_pivots_source_dists
                if pivot_index == 0:
                    sum_distance = source_dist_dicts
                else:
                    combined_distance = {"previous": sum_distance, "current": source_dist_dicts}
                    sum_distance = dict(reduce(add, map(Counter, combined_distance.values())))
            else:
                # previous_pivots_dists = previous_pivots_target_dists
                if pivot_index == 0:
                    sum_distance = target_dist_dicts
                else:
                    combined_distance = {"previous": sum_distance, "current": target_dist_dicts}
                    sum_distance = dict(reduce(and_, map(Counter, combined_distance.values())))
            # if start_point in previous_pivots_dists[start_point].keys():
            #     del previous_pivots_source_dists[start_point][start_point]
            sorted_sum_dist = sorted(sum_distance, key=lambda v: sum_distance[v])
            for key in sorted_sum_dist:
                if key in pivots:
                    continue
                else:
                    next_pivot = key
                    break
            if next_pivot == 'None':
                raise RuntimeError("Cannot find enough pivots! Please try with smaller number of pivots.")
            #             for key in previous_pivots_dists.keys():
            #                 if next_pivot in previous_pivots_dists[key].keys():
            #                     del previous_pivots_dists[key][next_pivot]
            start_point = next_pivot
            pivots.append(start_point)
            if pivot_index % count == 0:
                self.__log_intermediate_report(pivot_index, self.pivot_number)
        print("Successfully sampled %d pivots from the network based on MinSum approach..." % self.pivot_number)
        source_dist_dicts = self.__get_distances(graph, start_point, start='source')
        target_dist_dicts = self.__get_distances(graph, start_point, start='target')
        previous_pivots_source_dists.update({start_point: source_dist_dicts})
        previous_pivots_target_dists.update({start_point: target_dist_dicts})
        self.pivots = pivots
        self.pivots_as_source_distance = previous_pivots_source_dists
        self.pivots_as_target_distance = previous_pivots_target_dists
        return pivots, previous_pivots_dists

    def Mixed(pivot_number):
        pass