import torch.nn as nn
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hwcounter import count, count_end
import numpy as np
import math
import sys
import sklearn

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

#windows = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
windows = [18]

mode = "train"
in_file = "c_10005.txt"

if len(sys.argv) >= 2:
    in_file = sys.argv[1]
print("Running with " + in_file)
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=10, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.input_size = input_size
        self.linear = nn.Linear(hidden_layer_size, output_size)
        #TODO Testing using a softmax output
        #TODO CHECK THIS
        self.softmax = nn.Softmax(dim=2)
        #self.init_hidden()
        print(self)

    def init_hidden(self, batch_size):
        self.hidden_cell = (torch.zeros(1,batch_size,self.hidden_layer_size).to(DEVICE),
                            torch.zeros(1,batch_size,self.hidden_layer_size).to(DEVICE))

    def forward(self, inputs):
        # inputs.shape = (seq_length, batch_size)
        features = self.input_size
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]
        self.init_hidden(batch_size)
        #print(inputs.view(seq_length, batch_size, features))
        lstm_out, self.hidden_cell = self.lstm(inputs.view(seq_length, batch_size, features), self.hidden_cell)
        predictions = self.linear(lstm_out.view(seq_length, batch_size, self.hidden_layer_size))
        #TODO TESTING SOFTMAX
        predictions = self.softmax(predictions)
        #print(inputs.shape, predictions.shape, predictions[-1][-1].shape)
        return predictions[-1]



def train_online(model, train_sequence, epochs=15, batch_size=1):
    num_batches = len(train_sequence)//batch_size
    print(f"Total epochs: {epochs}, total batches {num_batches}")
    #print_gran = 200
    pred = []
    variation =0
    global pred_time
    global pred_samples
    global train_time
    global train_samples
    correct = 0
    above = 0
    below = 0
    total = 0
    u_pred_sum = 0
    o_pred_sum = 0
    warmup = 5
    for j in range(num_batches):
        batch_seq = train_sequence[j*batch_size:(j+1)*batch_size]
        #split the inputs and labels
        list_inputs, list_labels = list(zip(*batch_seq))
        inputs = torch.stack(list_inputs)
        labels = torch.stack(list_labels)
        start = count()
        model.train()
        optimizer.zero_grad()
        #outputs = model(inputs)
        #This seems to be needed in order to make the dimensions match, its just a dummy dimension
        outputs = torch.unsqueeze(model(inputs), 0)
        #outputs = torch.floor(outputs) + 1
        single_loss = loss_function(outputs, labels)
        #if j % print_gran == print_gran-1:
        #    print(f"Single Loss: {single_loss}")
        single_loss.backward()
        optimizer.step()
        train_time += count_end() - start
        train_samples += 1
        #Predict for the next timestep
        model.eval()
        with torch.no_grad():
            if j !=num_batches-1:
                batch_seq = train_sequence[(j+1)*batch_size:(j+2)*batch_size]
                list_inputs, list_labels = list(zip(*batch_seq))
                inputs = torch.stack(list_inputs)
                labels = torch.stack(list_labels)
                label_score = torch.argmax(labels)
                start = count()
                #tag_score = model(inputs)
                #print(f"Pred: {tag_score}, label: {labels}")
                tag_score = torch.argmax(model(inputs))
                pred_time += count_end() - start
                buf_score = int(math.ceil(tag_score * buf))
                if buf_score >= channels:
                    buf_score = channels-1
                #label_score = labels
                #print(tag_score)
                #print(label_score)
                #print(f"Pred: {tag_score}, label: {label_score}")
                #tag_score = torch.ceil(tag_score)
                #tag_score = model(inputs.unsqueeze(0))
                #pred.append(tag_score.cpu().detach().numpy()[0])
                pred_samples += 1
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
                total += 1
                #JUST COMMENTING THIS OUT FOR PERF
                #pred.append(tag_score.cpu().detach().numpy())
                variation += abs(tag_score - labels)

    total-=warmup
    print(f"overpred: {above/total}, correct: {correct/total}, underpred: {below/total}")
    print(f"avg_overpred: {o_pred_sum} {above}")
    print(f"avg_underpred: {u_pred_sum} {below}")
    #print(f"time per: {(end-start)/(epochs*num_batches)}")
    #print("avg_variation ", variation/len(train_sequence))
    return pred
    #checkpoint = {'model':model, 
    #        'state_dict': model.state_dict(),
    #        'optimizer':optimizer.state_dict()}
    #torch.save(checkpoint, 'checkpoint')


def test(model, test_seq):
    #Test what the scores are after training
    variation = 0
    pred = []
    model.eval()
    with torch.no_grad():
        for i, (seq,label) in enumerate(test_seq):
            tag_score = model(seq.unsqueeze(0))
            #TODO
            pred.append(tag_score.cpu().detach().numpy()[0])
            variation += abs(tag_score - label)
    print("avg_variation ", variation/len(test_seq))
    return pred

def preprocessing(input_data, win):
    inout_seq = []
    L = len(input_data)
    lookahead = 0
    for i in range(L-win-lookahead):
        #sends in the input bandwidths for the window length
        train_seq = torch.FloatTensor(input_data[i:i+win]).to(DEVICE)
        train_label= torch.FloatTensor(input_data[i+win+lookahead:i+win+lookahead+1]).to(DEVICE)
        #See the input format below, pulling out the sum_tot_bw
        #TODO CHANGES FOR SOFTMAX, otherwise just append "temp_label"
        temp_label = torch.split(train_label, 1, dim=1)[0]
        train_label = torch.zeros(1,channels)
        train_label[-1][int(temp_label[0][0])] = 1.0
        inout_seq.append((train_seq, train_label))
    return inout_seq

def read_input(in_file):
    #Input format:
    input_data = []
    #Open the input data file and read everything in
    with open(in_file, "r") as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            temp = line.split(" ")
            length = float(temp[0])
            bw = int(float(temp[1]))
            while length > 0:
                input_data.append([bw])
                length -= rem
            if len(input_data) > train_lim:
                break
    return input_data


window = 3
epochs = 15
batch_size=1 
MB = 1000000
channel_bw = 64
channels = 16
train_time = 0
train_samples = 0
pred_time = 0
pred_samples = 0
cyc_overhead = 350
train_lim = 10000

rem = 18
buf = 1
buffers = [1, 1.05, 1.1, 1.2, 1.3, 1.4]
if len(sys.argv) >= 3:
    #rem = windows[int(sys.argv[2])]
    buf = buffers[int(sys.argv[2])]

model = LSTM(input_size=1, output_size=channels).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_function = nn.MSELoss()
#Pull the sequence of total/r/w bws from the stats file
input_data = read_input(in_file)


input_seq = preprocessing(input_data, window)

#init lstm

num_samples = len(input_seq)
print(f"num_samples: {num_samples}")

pred_online=[]
if mode == "train":
    pred_online = train_online(model, input_seq, epochs=epochs, batch_size=batch_size)
    #pred_online = train(model, input_seq[:num_samples//2], epochs=epochs, batch_size=batch_size)
elif mode == "test":
    checkpoint = torch.load("checkpoint", map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

#Adjust for cycle overhead of the tracking itself
train_over_adjusted = train_time - (train_samples * cyc_overhead)
train_over_adjusted/=train_samples
pred_over_adjusted = pred_time - (pred_samples * cyc_overhead)
pred_over_adjusted/=pred_samples
print(f"Training Overhead: Samples: {train_samples} Time: {train_time} Per: {train_over_adjusted}")
print(f"Prediction Overhead: Samples: {pred_samples} Time: {pred_time} Per: {pred_over_adjusted}")

'''
plot_input = []
deb_output = []
plot_channel = []
for inp in input_data:
    plot_input.append(inp[14])
    deb_output.append(str(inp[14]) + " " + str(inp[17]))

with open(in_file+"_deb", 'w') as f:
    print("\n".join(map(str,plot_input)), file=f)
#    print(",".join(map(str,pred_online)), file=f)

#TODO
#window = 0
plt.plot(np.arange(len(plot_input)), plot_input, label='input')
#plt.plot(np.arange(0, num_samples//2)+window, pred_train)
#tot,read,write = list(zip(*pred_test))
#tot = list(zip(*pred_test))
'''
checkpoint = {'model':model, 
        'state_dict': model.state_dict(),
        'optimizer':optimizer.state_dict()}
torch.save(checkpoint, 'ali_bw_check.model')

'''
plt.plot(np.arange(len(pred_online))+window, pred_online, label='pred')
plt.xlabel("Time (s)")
plt.ylabel("Number of Channels")
plt.legend()
plt.gca().set_ylim(ymin=0)
plt.savefig(f"{in_file}.png")

'''
