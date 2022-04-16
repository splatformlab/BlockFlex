import torch.nn as nn
#from hwcounter import count, count_end
import torch
import numpy as np
import sys
import sklearn
import math
import mmap
import os
import time

DEVICE = torch.device("cpu")

#in_file = "NOT USING INPUT"
#ML_PREP
#in_file = "ml_prep_stats"
#TERASORT
#in_file = "terasort_stats"
#PAGERANK (GRAPHCHI)
#in_file = "pagerank_stats"

#if len(sys.argv) >= 2:
#    in_file = sys.argv[1]
#print("Running with " + in_file)
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=16, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.input_size = input_size
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.softmax = nn.Softmax(dim=2)
        self.init_hidden(1)
        #print(self)

    def init_hidden(self, batch_size):
        self.hidden_cell = (torch.zeros(1,batch_size,self.hidden_layer_size).to(DEVICE),
                            torch.zeros(1,batch_size,self.hidden_layer_size).to(DEVICE))

    def forward(self, inputs):
        features = self.input_size
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]
        self.init_hidden(batch_size)
        lstm_out, self.hidden_cell = self.lstm(inputs.view(seq_length, batch_size, features), self.hidden_cell)
        predictions = self.linear(lstm_out.view(seq_length, batch_size, self.hidden_layer_size))
        predictions = self.softmax(predictions)
        return predictions[-1]


fname = './bw_inputs.txt'
def get_inputs():
    cur_version = 0
    with open(fname, 'r') as fd:
        mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ, offset=0)
        while True:
            #Reset to the start of the file
            mm.seek(0)
            ret_inps = list(map(int, mm.read().decode("utf-8").strip().split()))
            if ret_inps[0] > cur_version and len(ret_inps) > 6:
                cur_version = ret_inps[0]
                #print(f"Starting Iteration: {cur_version}")
                #print(ret_inps)
                yield ret_inps[1:]
            elif ret_inps[0] < 0:
                mm.close()
                return None
            time.sleep(1)


def train_online(model, optimizer, loss_function, buf):
    #Store the history of variables
    pred = []
    train_sequence = []
    #variation =0

    #Previous prediction, used for accuracy reporting
    buf_score = None
    #Stat collection variables
    correct = 0
    train_samples = 0
    train_time = 0
    pred_samples = 0
    pred_time = 0
    o_pred_sum = 0
    u_pred_sum = 0
    above = 0
    below = 0
    total = 0
    warmup = 5

    for temp_seq in get_inputs():
        #split the inputs and labels
        train_sequence.append(temp_seq)
        if len(train_sequence) <= window: continue
        batch_seq = []
        t_seq = preprocess_single(train_sequence,window)
        if t_seq is None:
            break
        batch_seq.append(t_seq)
        #continue
        list_inputs, list_labels = list(zip(*batch_seq))
        inputs = torch.stack(list_inputs)
        labels = torch.stack(list_labels)
        #Get accuracy from previous iteratoin
        if buf_score is not None:
            label_score = torch.argmax(labels)
            if label_score < tag_score:
                if total >= warmup: 
                    above += 1
                    o_pred_sum += tag_score - label_score
            elif label_score > buf_score:
                if total >= warmup: 
                    below += 1
                    u_pred_sum += label_score - buf_score
            else:
                if total >= warmup: 
                    correct += 1
            pred_samples += 1
            #print(f"Pred: {tag_score}, buffer: {buf_score}, label: {label_score}")
            #variation += abs(tag_score - labels)

        #TODO figure out an alt way to get this to work here
        #start = count()
        model.train()
        optimizer.zero_grad()
        outputs = torch.unsqueeze(model(inputs), 0)

        #TRAIN
        if torch.argmax(outputs) < torch.argmax(labels):
            train_label = torch.zeros(1,1,channels)
            train_label[0][0][min(15,torch.argmax(labels)+1)] = 1
            single_loss = loss_function(outputs, labels)
            single_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            outputs = torch.unsqueeze(model(inputs), 0)
            single_loss = loss_function(outputs, labels)
            single_loss.backward()
            optimizer.step()
        else:
            single_loss = loss_function(outputs, labels)
            single_loss.backward()
            optimizer.step()
        #train_time += count_end() - start
        train_samples += 1
        model.eval()
        #Predict for the next timestep
        with torch.no_grad():
            batch_seq = []
            batch_seq.append(pred_single(train_sequence, window))
            list_inputs, list_labels = list(zip(*batch_seq))
            inputs = torch.stack(list_inputs)
            #start = count()
            #tag_score = model(inputs)
            tag_score = torch.argmax(model(inputs))
            #pred_time += count_end() - start
            buf_score = int(math.ceil(tag_score * buf))
            if buf_score >= channels:
                buf_score = channels-1
        total += 1
    total-=warmup
    print(f"overpred: {above/total}, correct: {correct/total}, underpred: {below/total}")
    print(f"avg_overpred: {o_pred_sum} {above}")
    print(f"avg_underpred: {u_pred_sum} {below}")
    #print(f"time per: {(end-start)/(num_batches)}")
    #print("avg_variation ", variation/len(train_sequence))

    #Adjust for cycle overhead of the tracking itself
    train_over_adjusted = train_time - (train_samples * cyc_overhead)
    train_over_adjusted/=train_samples
    pred_over_adjusted = pred_time - (pred_samples * cyc_overhead)
    pred_over_adjusted/=pred_samples
    #print(f"Training Overhead: Samples: {train_samples} Time: {train_time} Per: {train_over_adjusted}")
    #print(f"Prediction Overhead: Samples: {pred_samples} Time: {pred_time} Per: {pred_over_adjusted}")

    return pred
    #checkpoint = {'model':model, 
    #        'state_dict': model.state_dict(),
    #        'optimizer':optimizer.state_dict()}
    #torch.save(checkpoint, 'checkpoint')
    
def pred_single(input_data, win):
    i = len(input_data)-1
    train_seq = torch.FloatTensor(input_data[i-win+1:i+1]).to(DEVICE)
    train_label= torch.FloatTensor(input_data[i:i+1]).to(DEVICE)
    return (train_seq, train_label)

def preprocess_single(input_data, win):
    i = len(input_data)-1
    #sends in the input bandwidths for the window length
    train_seq = torch.FloatTensor(input_data[i-win:i]).to(DEVICE)
    train_label= torch.FloatTensor(input_data[i:i+1]).to(DEVICE)
    #See the input format below, pulling out the sum_tot_bw
    temp_label = torch.split(train_label, 1, dim=1)[4]
    train_label = torch.zeros(1,channels)
    train_label[0][min(int(temp_label[0][0]),15)] = 1.0
    train_seq/=16
    ret_seq = (train_seq, train_label)
    return ret_seq


def preprocessing(input_data, win):
    inout_seq = []
    L = len(input_data)
    for i in range(L-win):
        #sends in the input bandwidths for the window length
        train_seq = torch.FloatTensor(input_data[i:i+win]).to(DEVICE)
        train_label= torch.FloatTensor(input_data[i+win:i+win+1]).to(DEVICE)
        #See the input format below, pulling out the sum_tot_bw
        temp_label = torch.split(train_label, 1, dim=1)[4]
        train_label = torch.zeros(1,channels)
        train_label[0][min(int(temp_label[0][0]),15)] = 1.0
        train_seq/=16
        inout_seq.append((train_seq, train_label))
    return inout_seq

def read_input(in_file):
    #Input format:
    #max_r_bw, max_w_bw, max_tot_bw, max_r_op, max_w_op, max_tot_op, 
    #min_r_bw, min_w_bw, min_tot_bw, min_r_op, min_w_op, min_tot_op, 
    #avg_r_bw, avg_w_bw, avg_tot_bw, avg_r_op, avg_w_op, avg_tot_op, 
    #sum_r_bw, sum_w_bw, sum_tot_bw, sum_r_op, sum_w_op, sum_tot_op, 
    #med_r_bw, med_w_bw, med_tot_bw, med_r_op, med_w_op, med_tot_op, 
    #std_r_bw, std_w_bw, std_tot_bw, std_r_op, std_w_op, std_tot_op
    input_data = []
    #Open the input data file and read everything in
    with open(in_file, "r") as f:
        for line in f:
            temp = line.split(",")
            for i in range(len(temp)):
                temp[i] = float(temp[i])
                if i % 6 < 3:
                    temp[i] = temp[i]/MB
                    temp[i] = int(temp[i]/channel_bw) + 1
                else:
                    temp[i] = int(temp[i]/iops_chl) + 1
            #max, min, avg for bw, iops
            add_list = [temp[2], temp[5], temp[8], temp[11], temp[14], temp[17]]
            input_data.append(add_list)
    return input_data


window = 3
MB = 1000000
iops_chl =  500000//16
channel_bw = 64
channels = 16

cyc_overhead = 350

def main():

    #init lstm
    model = LSTM(input_size=6, output_size=channels).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_function = nn.MSELoss()

    #Overprovisioning
    buf = 1.05

    pred_online=[]
    pred_online = train_online(model, optimizer, loss_function, buf)

    #TODO Uncomment?
    '''
    checkpoint = {'model':model, 
            'state_dict': model.state_dict(),
            'optimizer':optimizer.state_dict()}
    torch.save(checkpoint, 'bw_size.model')
    '''
