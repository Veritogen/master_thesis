import networkx as nx


class GraphFeatures:
    '''
    class that will take a gexf file as input and will return a dictionary containing the characteristics of the graph.
    '''
    def __init__(self, graph):
        """
        Initiate class with a graph.
        :param graph: Networkx graph to extract the features from
        """
        self.graph = graph
        self.no_nodes = self.graph.number_of_nodes()
        self.max_degree = max(dict(self.graph.degree()).values())

    def return_features(self):
        """
        Method to return all features for a given graph.
        :return: Dictionary of graph features.
        """
        feature_dict = {
            'max_degree': self.max_degree_norm(),
            'average_degree': self.average_degree(),
            'density': self.density(),
            'longest_path': self.longest_path(),
            'degree_centralization': self.degree_centralization(),
            'betweenness_centralization': self.betweenness_centralization(),
            # excluded because not between 0 and 1
            #'closeness_centralization': self.closeness_centralization()

        }
        triads = self.no_triads_per_type()
        for triad, count in triads.items():
            feature_dict[triad] = count
        return feature_dict

    def max_degree_norm(self):
        """
        :return: Returns the maximum degree of the given graph.
        """
        return self.max_degree/self.no_nodes

    def average_degree(self):
        """
        :return: Returns the average degree of the given graph.
        """
        return sum(dict(self.graph.degree).values())/self.no_nodes

    def density(self):
        """
        :return: Returns the density of the given graph.
        """
        return nx.density(self.graph)

    def longest_path(self):
        """
        :return: Returns the longest path of the given graph, normalized by dividing by the number of nodes within the
        graph.
        """
        return nx.dag_longest_path_length(self.graph)/self.no_nodes

    def no_triads_per_type(self):
        """
        :return: Returns the triad count for each selected triad within the given graph, normalized by dividing by the
        number of nodes within the graph.
        """
        possible_triads = {'021U', '012', '021D', '003', '030T', '021C'}
        triad_dict = nx.triadic_census(self.graph)
        return {triad_type: triad_dict[triad_type]/self.no_nodes for triad_type in triad_dict.keys()
                if triad_type in possible_triads}

    def transitivity(self):
        """
        :return: Retunrs the transitivity of the given graph.
        """
        return nx.transitivity(self.graph)

    def degree_centralization(self):
        """
        :return: Returns the degree centralization of the given graph, according to Freeman 1978 - Centrality in social
        networks.
        """
        graph_degree_dict = dict(self.graph.degree())
        max_degree = max(graph_degree_dict.values())
        degree_sum = 0
        for node_id in graph_degree_dict.keys():
            degree_sum += (max_degree - graph_degree_dict[node_id])
        return degree_sum / (self.no_nodes ** 2 - 3 * self.no_nodes + 2)

    def betweenness_centralization(self):
        """
        :return: Returns the betweenness centralization of the given graph, according to Freeman 1978 - Centrality in
        social networks.
        """
        betweenness_dict = nx.betweenness_centrality(self.graph)
        max_betweenness = max(betweenness_dict.values())
        betweenness_sum = 0
        for node_id in betweenness_dict.keys():
            betweenness_sum += (max_betweenness - betweenness_dict[node_id])
        return betweenness_sum / (self.no_nodes - 1)

    def closeness_centralization(self):
        """
        :return: Returns the closeness centralization of the given graph, according to Freeman 1978 - Centrality in
        social networks.
        """
        closeness_dict = nx.closeness_centrality(self.graph)
        max_closeness = max(closeness_dict.values())
        closeness_sum = 0
        for node_id in closeness_dict:
            closeness_sum += (max_closeness - closeness_dict[node_id])
        return closeness_sum/((self.no_nodes ** 2 - 3 * self.no_nodes + 2) / (2 * self.no_nodes - 3))
