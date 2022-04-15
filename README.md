# BlockFlex v1.0

BlockFlex is out learning-based storage harvesting framework, which can harvest flash-based storage resources at fine-grained granularity in modern clould platforms.

## 1. Overview
The following packages are necessary to install before running the following scripts.
```shell
sudo apt update
sudo apt install ...
```

## 2. Trace Analysis
Figure 1:
We use the Alibaba Cloud Traces [1] for Figure 1. 
For plotting container utilization we sample 30000 containers which can be downloaded from the following link: (TODO LINK)
The results for the container are the same regardless if you use our file or the original container_usage.csv from Alibaba.
For plotting machine utilization, we use the full machine_usage.csv trace from Alibaba.
Finally, run the following:
```shell
python3 container_parser.py && python3 ali_container_usage.py
python3 machine_parser.py && python3 ali_machine_usage.py
```
This will create Figures 1a and 1b.



## 3. Predictor Analysis


## 4. BlockFlex


## 5. Sources
[1]. https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md
