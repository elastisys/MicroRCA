## How to run MicorRCA in Cloud-Native system

#### Presequisite of Cloud-Native system
1. Deploy microservices in Kubernetes
1. Deploy node-exporter
1. Deploy isito with prometheus

<!-- #### Data collection
Query application-level and system-level metrics from prometheus. A list of metrics is as follows:
* response times of service invocations (from istio)
* resource utilization of containers
* resource utilization of hosts -->

#### Presequisite of running MicroRCA
* python 3+
* [nx](https://networkx.github.io/documentation/stable/index.html)
* sklearn-learn

#### Clone the Repository
`git clone https://github.com/elastisys/MicroRCA.git`

####  Run MicroRCA online and turn the parameters
Feed the collected data to MicroRCA. It detects the anomalous response times between two communicating services and triggers the root cause analysis procedures. To run the code, please customize the code as follows:
1. update `node_dict` with the node name and ip address in your cluster manually
2. customize the monitoring interval `metric_step`  manually
3. run the code with parameters. An example is: <br/>
`python MicroRCA_online.py --folder '1' --length 150 --url 'http://localhost:9090/api/v1/query'`
4. tune parameters `alpha` and `ad_threshold` 
