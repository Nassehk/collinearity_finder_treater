#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:51:57 2020

@author: nassehk
"""
import pandas as pd
from sklearn.decomposition import PCA
from cluster import cluster
      

def identify_cluster(X_data_df, threshold = 0.7, correlation_id_method = 'pearson'):
    cor = X_data_df.corr(method = correlation_id_method)

    clusters = []
    for j,col in enumerate (cor.columns):
        for i,row in enumerate (cor.columns[0:j]):
            if abs(cor.iloc[i,j])>threshold:
                current_pair = (col,row, cor.iloc[i,j])
                current_pair_added = False
                for _c in clusters:
                    if _c.can_accept(current_pair):
                        _c.update_with(current_pair)
                        current_pair_added = True
                if current_pair_added == False:
                    clusters.append(cluster(pairs = [current_pair]))
    final_clusters = []
    
    #It is possible to have mergable clusters. Check if the idetified clusters can be merged.
    for _cluster in clusters:
        added_to_final = False
        for final_c in final_clusters:
            if _cluster.nodes.intersection(final_c.nodes) != set():
                final_c.merge_with_cluster(_cluster)
                added_to_final = True
        if added_to_final == False:
            final_clusters.append(_cluster)
    for i, _cluster in enumerate(final_clusters):
        _cluster.name = f'cluster_{i}'
    return final_clusters


def _pca(X_data,n=1):
    pca = PCA(n_components = n, svd_solver = 'auto')
    pca.fit(X_data)
    X_pca = pca.transform(X_data)      
    return (X_pca ,pca.explained_variance_ratio_, pca.components_, pca)               


class collinear_data():
    def __init__ (self, collinear_df):
        self.collinear_df = collinear_df
#        self._clusters = self.clusters(threshold = 0.7)
#        self.cluster_variables = {cl.name:cl.nodes for cl in self._clusters}
        self.pca_obj_dict = None
        
    def clusters(self, threshold = 0.7):
        self._clusters = identify_cluster(self.collinear_df , threshold = threshold)
        self.cluster_variables = {cl.name:cl.nodes for cl in self._clusters}
        for _cl in self._clusters:
            setattr(self,_cl.name, _cl)

    
    def _add_pc_to_collin_df(self, raw_data_df, pc_data, cluster_name, column_to_drop):
        raw_data = raw_data_df.copy()
        for i in range(pc_data.shape[1]):
            raw_data[f'{cluster_name}_pc{i}'] = pc_data[:,i]
        raw_data.drop(column_to_drop, inplace = True, axis = 1)
        return raw_data
        
    def non_collinear_df(self, threshold = 0.5, 
                         min_total_variance_ratio_explained = 0.9, 
                         verbose = True):
        self.clusters(threshold=threshold)
        final_df = self.collinear_df.copy()
#        conversion_dict={}
        pca_obj_dict = {}
#        cluster_linearity_index_dict={}
        for cluster_ in self._clusters:
#            print ('**',len(cluster_.nodes))
            for num_component in range(1,len(cluster_.nodes)): 
                pc_data, expl_variance, component, pca_obj = _pca(self.collinear_df[cluster_.nodes], n=num_component)
                if sum(expl_variance) > min_total_variance_ratio_explained:
                    break
            if verbose:
                print ('*'*10)
                print (cluster_.name)
                print (f'feature name = {str(cluster_.nodes)[1:-1]}')
                print (f'number of PC needed = {len(expl_variance)}')
                for i , variance in enumerate(expl_variance):
                    print (f'explained variance by PC_{i} = {variance}')
            final_df = self._add_pc_to_collin_df(final_df, pc_data, cluster_.name,cluster_.nodes)
            pca_obj_dict[cluster_.name]  = pca_obj
        self.pca_obj_dict = pca_obj_dict
        return final_df
    
    def convert_new_collin_data(self, sample_collin_df):
        '''
        Converts a dataframe containing collinear variables to the \
        non_collinear version that can be used with the non_collinear \
        training set. This function is meant to be used after the clusters \
        are identified so first run non_collinear_df method to identify 
        clusters and create conversion_dict.
        '''
        final_result = sample_collin_df.copy()
        if self.pca_obj_dict == None:
            raise Exception ("'conversion_dict' missing. Please run 'non_collinear_df' method first.")
        for cl in  self._clusters:
            collin_data = final_result[cl.nodes]
            pc_data = self.pca_obj_dict[cl.name].transform(collin_data)
            final_result = self._add_pc_to_collin_df(final_result, pc_data, cl.name, cl.nodes)
        return final_result
            
        
def sample_data(file):
    return pd.read_csv(file, index_col = 'Time')


if __name__ == '__main__':
    raw_data = sample_data('sample_X_data.csv')
#    define a threshold for identifying collinear pairs
    thresh = 0.7

# =============================================================================
# You can identify the clusters and see visualize them with graphs without 
# doing any pricessing. Uncomment he next three lines if you want to do so.    
# =============================================================================
#    clusters = identify_cluster(raw_data, threshold = thresh)
#    for cl in clusters:
#        cl.plot()

# =============================================================================
# This is the normal way of using this library. First create a collinear_data 
# object by providing the raw data
# =============================================================================

    colin_data = collinear_data(raw_data)
    point_col = pd.DataFrame(raw_data.iloc[2,:]).T
    non_colin_data = colin_data.non_collinear_df(threshold = thresh)
    for cl in colin_data._clusters:
        cl.plot(font_size = 10)
    point_non_colin = colin_data.convert_new_collin_data(point_col)
        