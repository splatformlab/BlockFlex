# BlockFlex v1.0

BlockFlex is out learning-based storage harvesting framework, which can harvest flash-based storage resources at fine-grained granularity in modern clould platforms.

## 1. Overview
The following packages are necessary to install before running the following scripts.
```shell
sudo apt update
sudo apt install ...
#Easy command line download of google drive files
pip install gdown
```

## 2. Trace Analysis (Motivation)
### Figure 1
We use the Alibaba Cloud Traces[1] for Figure 1.

For plotting container utilization we sample 30,000 containers.  The results for the container plot are the same regardless if you use our parsed file or the original container_usage.csv from Alibaba. If you use container_usage.csv, you will need to modify the first line of the script accordingly. 

For plotting machine utilization, we use the full machine_usage.csv trace from Alibaba.
Finally, run the following (should take under 10 minutes):
```shell
# (Optional) Grab the smaller container file (~1.3 GB)
gdown 1JMTT2CyMB_dyA86OfNwTPZUd2PhTX4ZW
gunzip container_usage_sub.csv.gz
#Creates the input file to the next command (ali_container_usage.dat)
python3 container_parser.py 
#Plot the container utilization (creates ali_container_usage.pdf)
python3 ali_container_usage.py
#Creates the input file to the next command (ali_machine_usage.dat)
python3 machine_parser.py 
#Plot the machine utilization (creates ali_machine_usage.pdf)
python3 ali_machine_usage.py
```
This will create Figures 1a and 1b in ali_container_usage.pdf and ali_machine_usage.pdf.

### Figure 2
We use Google's open source cluster traces[2] for Figure 2.

<details>
<summary>We will include the process to derive the results from scratch in this dropdown. It is faster to use the intermediate data files we provide below.</summary>
<br>

First, follow the instructions in [2] to download the full dataset.
```shell
#Copy the scripts into the google trace directory
cp prio_events_parser.py google-cloud-sdk/clusterdata-2011-2/
cp usage_parser.py google-cloud-sdk/clusterdata-2011-2/
#Run them (!! WARNING THIS WILL TAKE A LONG TIME)
#Produces the parsed_all_prio_events.csv file which contains the high priority VMs
python3 prio_events_parser.py
#Produces the usages_500.csv which is the total trace filtered by priority events
python3 usage_parser.py
#Copy the resulting files back into the Traces directory
cp parsed_all_prio_events.csv ../../
cp usages_500.csv ../../
```
</details>

Alternatively, use the provided parsed_all_prio_events.csv and download usages_500.csv:
```shell
#Download usages_500.csv (~60GB)
gdown 1maec7UF_6U8kMIHRGDpbPvTaqp6QrMxb
gunzip usages_500.csv.gz
```
Next, compute the cdf for utilization in the google trace:
(Takes under 25 minutes)
```shell
#Creates the input for the plotting (google_util_cdf.dat)
python3 google_util_parser.py 
#Plots the cdf for utilizations
python3 google_util_cdf.py
```
The final output is stored in google_util_cdf.pdf

### Figure 3
Ommitted due to business confidentiality

### Figure 4
TODO JINGHAN

### Figure 5
Ommitted due to business confidentiality

### Figure 6 
Ommitted due to business confidentiality

## 3. Predictor Analysis
The prediction directory contains some parsed traces of workloads, the predictors themselves as well as the scripts for plotting the accuracy measurements.

We begin by fetching the traces and preparing the inputs for the predictors. 
We include how each trace was produced in the following dropdowns and their associated scripts are in Predictions/TraceGen:  
<details>
<summary>Alibaba/Google</summary>
<br>
We randomly select 5 containers/VMs and isolate their workloads from the overall trace. They are preprocessed to be represented by {duration, amount} pairs where the VM is using amount resources for duration period of time. The predictors parse this into the same input format as described in the paper.
For reference, the scripts (train_gen_ali.py and train_gen_gl.py) to generate these pairs are included.
</details>
<details>
<summary>terasort</summary>
We install hadoop-3-3 using the hadoop documentation: https://hadoop.apache.org/docs/r3.3.0/hadoop-project-dist/hadoop-common/SingleCluster.html 

The run_hadoop.sh script provides a reference to how we collect the traces. We generate a 75GB dataset with TeraGen and then sort it with TeraSort. Traces are collected using blktrace. The size information is collected with check.sh which uses the 'du' command to track the size of the hadoop directory.
<br>
</details>
<details>
<summary>graphchi</summary>
<br>
We install graphchi using their git repo: https://github.com/GraphChi/graphchi-cpp

The run_graphchi.sh script provides a reference to how we collect the traces. We use graphchi's sample implementation of pagerank and input the Friendster graph. As with hadoop, the size information is collected with check.sh
</details>
<details>
<summary>ml_prep</summary>
We download the 2017 ImageNet dataset and repeatedly run simple preprocessing operations to mimic a batch processing of images for ML training. The python file and run_ml_prep.sh are included for reference.
<br>
</details>

The preparation scripts mimic the statistic collections of BlockFlex by reporting aggregations across the reporting window. This gives the same set of inputs the predictor would receive if it were receiving inputs from BlockFlex, but in an offline manner.

```shell
#Enter the prep directory
cd Prep
#Download the traces (~21 GB)
gdown 1hWZJOKNyumce0UNo45zb3_pfyytwg2YC
#Unzip them, should have a number of file with the format WORKLOAD_NUM.sizing and WORKLOAD_num.trace
tar -xzvf traces.tar.gz
#Run the preparation scripts
#Should create a set of files with the format WORKLOAD_NUM_stats in the input/bw directory
./regen_bw.sh
#Should create a set of files with the format WORKLOAD_NUM_sizing in the input/sz directory
./regen_size.sh
#Should create a set of files with the format WORKLOAD_NUM_dur in the input/dur_bw and input/dur_sz directories
./regen_duration.sh
#We are done in this directory
cd ..
```

Now that we have the inputs prepared, we go into the predictor directory and run them on the inputs
```shell

```


## 4. BlockFlex


## 5. Sources
[1]. https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md  
[2]. https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md