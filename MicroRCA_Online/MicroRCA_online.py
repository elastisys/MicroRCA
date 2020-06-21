#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: li
"""

import requests
import time
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import networkx as nx
import argparse
#import csv

from sklearn.cluster import Birch
from sklearn import preprocessing
#import seaborn as sns

## =========== Data collection ===========

metric_step = '5s'
smoothing_window = 12

# kubectl get nodes -o wide | awk -F ' ' '{print $1 " : " $6":9100"}'
node_dict = {
                'kubernetes-minion-group-103j' : '10.166.0.21:9100',
                'kubernetes-minion-group-k2nz' : '10.166.15.235:9100',
                'kubernetes-minion-group-kvcr' : '10.166.0.13:9100',
                'kubernetes-minion-group-r23j' : '10.166.0.14:9100',
        }


        

def latency_source_50(prom_url, start_time, end_time, faults_name):

    latency_df = pd.DataFrame()


    response = requests.get(prom_url,
                            params={'query': 'histogram_quantile(0.50, sum(irate(istio_request_duration_seconds_bucket{reporter=\"source\", destination_workload_namespace=\"sock-shop\"}[1m])) by (destination_workload, source_workload, le))',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']

        values = list(zip(*values))
        if 'timestamp' not in latency_df:
            timestamp = values[0]
            latency_df['timestamp'] = timestamp
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64')  * 1000


    response = requests.get(prom_url,
                            params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"source\"}[1m])) by (destination_workload, source_workload) / 1000',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
#        print(svc)
        values = result['values']

        values = list(zip(*values))
        if 'timestamp' not in latency_df:
            timestamp = values[0]
            latency_df['timestamp'] = timestamp
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64').rolling(window=smoothing_window, min_periods=1).mean()

    filename = faults_name + '_latency_source_50.csv'
    latency_df.set_index('timestamp')
    latency_df.to_csv(filename)
    return latency_df


def latency_destination_50(prom_url, start_time, end_time, faults_name):

    latency_df = pd.DataFrame()


    response = requests.get(prom_url,
                            params={'query': 'histogram_quantile(0.50, sum(irate(istio_request_duration_seconds_bucket{reporter=\"destination\", destination_workload_namespace=\"sock-shop\"}[1m])) by (destination_workload, source_workload, le))',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']

        values = list(zip(*values))
        if 'timestamp' not in latency_df:
            timestamp = values[0]
            latency_df['timestamp'] = timestamp
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64')  * 1000


    response = requests.get(prom_url,
                            params={'query': 'sum(irate(istio_tcp_sent_bytes_total{reporter=\"destination\"}[1m])) by (destination_workload, source_workload) / 1000',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
#        print(svc)
        values = result['values']

        values = list(zip(*values))
        if 'timestamp' not in latency_df:
            timestamp = values[0]
            latency_df['timestamp'] = timestamp
            latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        latency_df[name] = pd.Series(metric)
        latency_df[name] = latency_df[name].astype('float64').rolling(window=smoothing_window, min_periods=1).mean()

    filename = faults_name + '_latency_destination_50.csv'
    latency_df.set_index('timestamp')
    latency_df.to_csv(filename)
    return latency_df

def svc_metrics(prom_url, start_time, end_time, faults_name):
    response = requests.get(prom_url,
                            params={'query': 'sum(rate(container_cpu_usage_seconds_total{namespace="sock-shop", container_name!~\'POD|istio-proxy|\'}[1m])) by (pod_name, instance, container)',
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    for result in results:
        df = pd.DataFrame()
        svc = result['metric']['container']
        pod_name = result['metric']['pod_name']
        nodename = result['metric']['instance']

#        print(svc)
        values = result['values']

        values = list(zip(*values))
        if 'timestamp' not in df:
            timestamp = values[0]
            df['timestamp'] = timestamp
            df['timestamp'] = df['timestamp'].astype('datetime64[s]')
        metric = pd.Series(values[1])
        df['ctn_cpu'] = metric
        df['ctn_cpu'] = df['ctn_cpu'].astype('float64')

        df['ctn_network'] = ctn_network(start_time, end_time, pod_name)
        df['ctn_network'] = df['ctn_network'].astype('float64')
        df['ctn_memory'] = ctn_memory(start_time, end_time, pod_name)
        df['ctn_memory'] = df['ctn_memory'].astype('float64')

#        response = requests.get('http://localhost:9090/api/v1/query',
#                                params={'query': 'sum(node_uname_info{nodename="%s"}) by (instance)' % nodename
#                                        })
#        results = response.json()['data']['result']
#
#        print(results)
#
#
#        instance = results[0]['metric']['instance']
        instance = node_dict[nodename]

        df_node_cpu = node_cpu(start_time, end_time, instance)
        df = pd.merge(df, df_node_cpu, how='left', on='timestamp')


        df_node_network = node_network(start_time, end_time, instance)
        df = pd.merge(df, df_node_network, how='left', on='timestamp')

        df_node_memory = node_memory(start_time, end_time, instance)
        df = pd.merge(df, df_node_memory, how='left', on='timestamp')
    

        filename = faults_name + '_' + svc + '.csv'
        df.set_index('timestamp')
        df.to_csv(filename)

def ctn_network(prom_url, start_time, end_time, pod_name):
    response = requests.get(prom_url,
                            params={'query': 'sum(rate(container_network_transmit_packets_total{namespace="sock-shop", pod_name="%s"}[1m])) / 1000 * sum(rate(container_network_transmit_packets_total{namespace="sock-shop", pod_name="%s"}[1m])) / 1000' % (pod_name, pod_name),
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    values = results[0]['values']

    values = list(zip(*values))
    metric = pd.Series(values[1])
    return metric


def ctn_memory(prom_url, start_time, end_time, pod_name):
    response = requests.get(prom_url,
                            params={'query': 'sum(rate(container_memory_working_set_bytes{namespace="sock-shop", pod_name="%s"}[1m])) / 1000 ' % pod_name,
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']

    values = results[0]['values']

    values = list(zip(*values))
    metric = pd.Series(values[1])
    return metric


def node_network(prom_url, start_time, end_time, instance):
    response = requests.get(prom_url,
                            params={'query': 'rate(node_network_transmit_packets{device="eth0", instance="%s"}[1m]) / 1000' % instance,
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    values = results[0]['values']

    values = list(zip(*values))
    df = pd.DataFrame()
    df['timestamp'] = values[0]
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df['node_network'] = pd.Series(values[1])
    df['node_network'] = df['node_network'].astype('float64')
#    return metric
    return df

def node_cpu(prom_url, start_time, end_time, instance):
    response = requests.get(prom_url,
                            params={'query': 'sum(rate(node_cpu{mode != "idle",  mode!= "iowait", mode!~"^(?:guest.*)$", instance="%s" }[1m])) / count(node_cpu{mode="system", instance="%s"})' % (instance, instance),
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    values = results[0]['values']
    values = list(zip(*values))
#    metric = values[1]
#    print(instance, len(metric))
#    print(values[0])
    df = pd.DataFrame()
    df['timestamp'] = values[0]
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df['node_cpu'] = pd.Series(values[1])
    df['node_cpu'] = df['node_cpu'].astype('float64')
#    return metric
    return df

def node_memory(prom_url, start_time, end_time, instance):
    response = requests.get(prom_url,
                            params={'query': '1 - sum(node_memory_MemAvailable{instance="%s"}) / sum(node_memory_MemTotal{instance="%s"})' % (instance, instance),
                                    'start': start_time,
                                    'end': end_time,
                                    'step': metric_step})
    results = response.json()['data']['result']
    values = results[0]['values']

    values = list(zip(*values))
#    metric = values[1]
#    return metric
    df = pd.DataFrame()
    df['timestamp'] = values[0]
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df['node_memory'] = pd.Series(values[1])
    df['node_memory'] = df['node_memory'].astype('float64')
#    return metric
    return df

# Create Graph
def mpg(prom_url, faults_name):
    DG = nx.DiGraph()
    df = pd.DataFrame(columns=['source', 'destination'])
    response = requests.get(prom_url,
                            params={'query': 'sum(istio_tcp_received_bytes_total) by (source_workload, destination_workload)'
                                    })
    results = response.json()['data']['result']

    for result in results:
        metric = result['metric']
        source = metric['source_workload']
        destination = metric['destination_workload']
#        print(metric['source_workload'] , metric['destination_workload'] )
        df = df.append({'source':source, 'destination': destination}, ignore_index=True)
        DG.add_edge(source, destination)
        
        DG.node[source]['type'] = 'service'
        DG.node[destination]['type'] = 'service'

    response = requests.get(prom_url,
                            params={'query': 'sum(istio_requests_total{destination_workload_namespace=\'sock-shop\'}) by (source_workload, destination_workload)'
                                    })
    results = response.json()['data']['result']

    for result in results:
        metric = result['metric']
        
        source = metric['source_workload']
        destination = metric['destination_workload']
#        print(metric['source_workload'] , metric['destination_workload'] )
        df = df.append({'source':source, 'destination': destination}, ignore_index=True)
        DG.add_edge(source, destination)
        
        DG.node[source]['type'] = 'service'
        DG.node[destination]['type'] = 'service'

    response = requests.get(prom_url,
                            params={'query': 'sum(container_cpu_usage_seconds_total{namespace="sock-shop", container_name!~\'POD|istio-proxy\'}) by (instance, container)'
                                    })
    results = response.json()['data']['result']
    for result in results:
        metric = result['metric']
        if 'container' in metric:
            source = metric['container']
            destination = metric['instance']
            df = df.append({'source':source, 'destination': destination}, ignore_index=True)
            DG.add_edge(source, destination)
            
            DG.node[source]['type'] = 'service'
            DG.node[destination]['type'] = 'host'

    filename = faults_name + '_mpg.csv'
##    df.set_index('timestamp')
    df.to_csv(filename)
    return DG


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



def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Root cause analysis for microservices')

    parser.add_argument('--folder', type=str, required=False,
                        default='1',
                        help='folder name to store csv file')
    
    parser.add_argument('--length', type=int, required=False,
                    default=150,
                    help='length of time series')

    parser.add_argument('--url', type=str, required=False,
                    default='http://localhost:9090/api/v1/query',
                    help='url of prometheus query')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    folder = args.folder
    len_second = args.length
    prom_url = args.url
      
    faults_name = folder
    
    end_time = time.time()
    start_time = end_time - len_second


    # Tuning parameters
    alpha = 0.55  
    ad_threshold = 0.045  

    latency_df_source = latency_source_50(prom_url, start_time, end_time, faults_name)
    latency_df_destination = latency_destination_50(prom_url, start_time, end_time, faults_name)
    latency_df = latency_df_destination.add(latency_df_source) 
    
    
    svc_metrics(prom_url, start_time, end_time, faults_name)
    
    DG = mpg(prom_url, faults_name)

    # anomaly detection on response time of service invocation
    anomalies = birch_ad_with_smoothing(latency_df, ad_threshold)
    
    # get the anomalous service
    anomaly_nodes = []
    for anomaly in anomalies:
        edge = anomaly.split('_')
        anomaly_nodes.append(edge[1])
    
    anomaly_nodes = set(anomaly_nodes)
     
    anomaly_score = anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha)
    print(anomaly_score)

    
    anomaly_score_new = []
    for anomaly_target in anomaly_score:
        node = anomaly_target[0]
#                        print(anomaly_target[0])
        if DG.nodes[node]['type'] == 'service':
            anomaly_score_new.append(anomaly_target)
    print(anomaly_score_new)





