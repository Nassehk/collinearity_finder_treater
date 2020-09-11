#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:21:48 2020

@author: nassehk
"""
import networkx as nx
import matplotlib.pyplot as plt

class cluster():
    def __init__(self, pairs=None):
        '''
        Parameters:
            Pairs = list of tuple in the form of [(node1,node2, weight1),(node3,node4,weight2)]
        '''
        self.pairs = set()
        self.nodes = set()
        self.name = None
        if pairs != None:
            for pair in pairs:
                self.nodes.update([pair[0],pair[1]])
                self.pairs.add(pair)

    def update_with(self, pair, force_update = False):
        """
        Addds a new pair to the cluster. The pair must have a node in common with cluster otherwise Exception will be raised. If force_update is set to True, a pair with no common node can be forced into cluster.
        Parameters:
            pair : A list or tuple in the form of (node1(str),node2(str), weight(float64)).
            force_update (boolean): True or False
        Returns:
            No return
        """
        if force_update:
            self.nodes.update([pair[0],pair[1]])
            self.pairs.add(pair)
        else:    
            if self.can_accept(pair):
                self.nodes.update([pair[0],pair[1]])
                self.pairs.add(pair)
            else:
                raise Exception(f'The pair {pair} can not be added to this cluster because it does not have any shared node with the current cluster nodes.')
        
    def can_accept(self, pair):
        ''' Checks is a pair has a node in common with the current cluster. Return boolean'''
        return (pair[0] in self.nodes or pair[1] in self.nodes)
    
    def merge_with_cluster(self, cluster_new, force_merge = False):
        '''
        Merges an input cluster with the current cluster. There has to be a node in common between two clusters to be able to merge. If force_merge=True clusters with no common node can be merged.
        Parameters:
            cluster_new: a new cluster instance that is to merge with current cluster.
            force_merge: boolean 
        '''
        def merge():
            self.nodes = self.nodes.union(cluster_new.nodes)
            self.pairs = self.pairs.union(cluster_new.pairs)
        
        if force_merge:
            merge()
        else:    
            if self.nodes.intersection(cluster_new.nodes) != set():
                merge()
            else:
                raise Exception(f'The clusters cannot be merged because they do not have any common node.')

    def plot(self, fig_size = (10,10), dpi= 200, max_line_width = 5, min_line_width = 1, min_alpha = 0.2, threshold=None, font_size=20):
        '''
        Plots the cluster.
        '''
        if threshold == None:
            threshold = min([pair[2] for pair in self.pairs])
        graph_pairs = [(pair[0],pair[1]) for pair in self.pairs]
        graph = nx.Graph()
        plt.figure(figsize = fig_size, dpi = dpi)
        graph.add_edges_from(graph_pairs)
        pos = nx.spring_layout(graph)

        for pair in self.pairs:
            nx.draw_networkx_nodes(graph, pos, node_size=300, edgecolors = (0.3,0.3,0.3), linewidths = 1, node_color = '#FFF')

            alph = (abs(pair[2])-threshold)/(1-threshold)
            if alph < min_alpha:
                alph = min_alpha
#            print(pair[2])

            w = (abs(pair[2])-threshold)/(1-threshold)*max_line_width
            if w < min_line_width:
                w = min_line_width

            color = 'b'    
            if pair[2]<0:
                color = 'r'

            nx.draw_networkx_edges(graph, pos, edge_color = color, edgelist = [[pair[0],pair[1]]], width = w, alpha = alph )
            nx.draw_networkx_labels(graph, pos, font_size=font_size, font_weight = 'bold', font_family='sans-serif', font_color=(0,0,0), alpha = 0.9)
            
        plt.title(self.name)
        plt.show()