#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: li
"""

import pandas as pd
import numpy as np
import networkx as nx
import csv

from sklearn.cluster import Birch
from sklearn import preprocessing


smoothing_window = 12

# Anomaly Detection
def birch_ad_with_smoothing(latency_df, threshold):
    # anomaly detection on response time of service invocation. 
    # input: response times of service invocations, threshold for birch clustering
    # output: anomalous service invocation
    
    anomalies = []
    for svc, latency in latency_df.iteritems():
        # No anomaly detection in db
        if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
            latency = latency.rolling(window=smoothing_window, min_periods=1).mean()
            x = np.array(latency)
            x = np.where(np.isnan(x), 0, x)
            normalized_x = preprocessing.normalize([x])

            X = normalized_x.reshape(-1,1)

#            threshold = 0.05

            brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
            brc.fit(X)
            brc.predict(X)

            labels = brc.labels_
#            centroids = brc.subcluster_centers_
            n_clusters = np.unique(labels).size
            if n_clusters > 1:
                anomalies.append(svc)
    return anomalies

def rt_invocations(faults_name):
    # retrieve the response time of each invocation from data collection
    # input: prefix of the csv file
    # output: round-trip time
    
    latency_filename = faults_name + '_latency_source_50.csv'  # inbound
    latency_df_source = pd.read_csv(latency_filename) 
    latency_df_source['unknown_front-end'] = 0
    
    latency_filename = faults_name + '_latency_destination_50.csv' # outbound
    latency_df_destination = pd.read_csv(latency_filename) 
    
    latency_df = latency_df_destination.add(latency_df_source)    

    return latency_df


def attributed_graph(faults_name):
    # build the attributed graph 
    # input: prefix of the file
    # output: attributed graph

    filename = faults_name + '_mpg.csv'
    df = pd.read_csv(filename)
    
    DG = nx.DiGraph()    
    for index, row in df.iterrows():
        source = row['source']
        destination = row['destination']
        if 'rabbitmq' not in source and 'rabbitmq' not in destination and 'db' not in destination and 'db' not in source:
            DG.add_edge(source, destination)

    for node in DG.nodes():
        if 'kubernetes' in node: 
            DG.nodes[node]['type'] = 'host'
        else:
            DG.nodes[node]['type'] = 'service'
#    
#    print(DG.nodes(data=True))
            
#    plt.figure(figsize=(9,9))
#    nx.draw(DG, with_labels=True, font_weight='bold')
#    pos = nx.spring_layout(DG)
#    nx.draw(DG, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True, )
#    labels = nx.get_edge_attributes(DG,'weight')
#    nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels)
#    plt.show()
                
    return DG 

def node_weight(svc, anomaly_graph, baseline_df, faults_name):

    #Get the average weight of the in_edges
    in_edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
#        print(u, v)
        num = num + 1
        in_edges_weight_avg = in_edges_weight_avg + data['weight']
    if num > 0:
        in_edges_weight_avg  = in_edges_weight_avg / num

    filename = faults_name + '_' + svc + '.csv'
    df = pd.read_csv(filename)
    node_cols = ['node_cpu', 'node_network', 'node_memory']
    max_corr = 0.01
    metric = 'node_cpu'
    for col in node_cols:
        temp = abs(baseline_df[svc].corr(df[col]))
        if temp > max_corr:
            max_corr = temp
            metric = col
    data = in_edges_weight_avg * max_corr
    return data, metric

def svc_personalization(svc, anomaly_graph, baseline_df, faults_name):

    filename = faults_name + '_' + svc + '.csv'
    df = pd.read_csv(filename)
    ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
    max_corr = 0.01
    metric = 'ctn_cpu'
    for col in ctn_cols:
        temp = abs(baseline_df[svc].corr(df[col]))     
        if temp > max_corr:
            max_corr = temp
            metric = col


    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight']

    for u, v, data in anomaly_graph.out_edges(svc, data=True):
        if anomaly_graph.nodes[v]['type'] == 'service':
            num = num + 1
            edges_weight_avg = edges_weight_avg + data['weight']

    edges_weight_avg  = edges_weight_avg / num

    personalization = edges_weight_avg * max_corr

    return personalization, metric



def anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha):
    # Get the anomalous subgraph and rank the anomalous services
    # input: 
    #   DG: attributed graph
    #   anomlies: anoamlous service invocations
    #   latency_df: service invocations from data collection
    #   agg_latency_dff: aggregated service invocation
    #   faults_name: prefix of csv file
    #   alpha: weight of the anomalous edge
    # output:
    #   anomalous scores 
    
    # Get reported anomalous nodes
    edges = []
    nodes = []
#    print(DG.nodes())
    baseline_df = pd.DataFrame()
    edge_df = {}
    for anomaly in anomalies:
        edge = anomaly.split('_')
        edges.append(tuple(edge))
#        nodes.append(edge[0])
        svc = edge[1]
        nodes.append(svc)
        baseline_df[svc] = latency_df[anomaly]
        edge_df[svc] = anomaly

#    print('edge df:', edge_df)
    nodes = set(nodes)
#    print(nodes)

    personalization = {}
    for node in DG.nodes():
        if node in nodes:
            personalization[node] = 0

    # Get the subgraph of anomaly
    anomaly_graph = nx.DiGraph()
    for node in nodes:
#        print(node)
        for u, v, data in DG.in_edges(node, data=True):
            edge = (u,v)
#            print(edge)
            if edge in edges:
                data = alpha
            else:
                normal_edge = u + '_' + v
                data = baseline_df[v].corr(latency_df[normal_edge])

            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

       # Set personalization with container resource usage
        for u, v, data in DG.out_edges(node, data=True):
            edge = (u,v)
            if edge in edges:
                data = alpha
            else:

                if DG.nodes[v]['type'] == 'host':
                    data, col = node_weight(u, anomaly_graph, baseline_df, faults_name)
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[u].corr(latency_df[normal_edge])
            data = round(data, 3)
            anomaly_graph.add_edge(u,v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']


    for node in nodes:
        max_corr, col = svc_personalization(node, anomaly_graph, baseline_df, faults_name)
        personalization[node] = max_corr / anomaly_graph.degree(node)
#        print(node, personalization[node])

    anomaly_graph = anomaly_graph.reverse(copy=True)
#
    edges = list(anomaly_graph.edges(data=True))

    for u, v, d in edges:
        if anomaly_graph.nodes[node]['type'] == 'host':
            anomaly_graph.remove_edge(u,v)
            anomaly_graph.add_edge(v,u,weight=d['weight'])

#    plt.figure(figsize=(9,9))
##    nx.draw(DG, with_labels=True, font_weight='bold')
#    pos = nx.spring_layout(anomaly_graph)
#    nx.draw(anomaly_graph, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True, )
#    labels = nx.get_edge_attributes(anomaly_graph,'weight')
#    nx.draw_networkx_edge_labels(anomaly_graph,pos,edge_labels=labels)
#    plt.show()
#
##    personalization['shipping'] = 2
#    print('Personalization:', personalization)



    anomaly_score = nx.pagerank(anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000)

    anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)

#    return anomaly_graph
    return anomaly_score

def print_rank(anomaly_score, target):
    num = 10
    for idx, anomaly_target in enumerate(anomaly_score):
        if target in anomaly_target:
            num = idx + 1
            continue
    print(target, ' Top K: ', num)
    return num


if __name__ == '__main__':
    
    # Tuning parameters
    alpha = 0.55  
    ad_threshold = 0.045  
    
    
    folders = ['1', '2', '3', '4', '5']
    faults_type = ['svc_latency', 'service_cpu', 'service_memory'] #, 'service_memory', 'svc_latency'
#    faults_type = ['svc_latency', 'service_cpu']
    targets = ['front-end', 'catalogue', 'orders', 'user', 'carts', 'payment', 'shipping']
        
    for folder in folders:
        for fault_type in faults_type:
            for target in targets:
                if target == 'front-end' and fault_type != 'svc_latency':
                    #'skip front-end for service_cpu and service_memory'
                    continue 
                print('target:', target, ' fault_type:', fault_type)
                
                # prefix of csv files 
                faults_name = '../faults/' + folder + '/' + fault_type + '_' + target
                
                latency_df = rt_invocations(faults_name)
                
                if (target == 'payment' or target  == 'shipping') and fault_type != 'svc_latency':
                    threshold = 0.02
                else:
                    threshold = ad_threshold   
                
                # anomaly detection on response time of service invocation
                anomalies = birch_ad_with_smoothing(latency_df, threshold)
#                print(anomalies)
                
                # get the anomalous service
                anomaly_nodes = []
                for anomaly in anomalies:
                    edge = anomaly.split('_')
                    anomaly_nodes.append(edge[1])
                
                anomaly_nodes = set(anomaly_nodes)
                
#                print(anomaly_nodes)
                
                # construct attributed graph
                DG = attributed_graph(faults_name)

               
                anomaly_score = anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha)
                print(anomaly_score)

                
                anomaly_score_new = []
                for anomaly_target in anomaly_score:
#                        print(anomaly_target[0])
                    if anomaly_target[0] in targets:
                        anomaly_score_new.append(anomaly_target)

                num = print_rank(anomaly_score_new, target)

                

                filename = 'MicroRCA_results.csv'                    
                with open(filename,'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([folder, target, fault_type, num, anomaly_score_new[:num], anomaly_nodes]) 

