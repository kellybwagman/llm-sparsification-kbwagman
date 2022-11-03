import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import GPT2Tokenizer, GPT2Model, EncoderDecoderModel, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration
import transformers
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import math
import numpy as np

HIST_BIN_SIZE = 0.1
MAX_PARAM_VALUE = 5
HIST_NUM_BINS = int(MAX_PARAM_VALUE/HIST_BIN_SIZE)

# Creates histograms of parameters by layer and for model overall
def count_big_parameters(model, name):
    table_counts = PrettyTable(["Layers", "Parameters"])
    table_hist = PrettyTable(["Layers", "Hist of Params"])
    table_percents = PrettyTable(["Layers", "Percent Params > {0}".format(HIST_BIN_SIZE)])
    total_params = 0
    model_hist = torch.zeros(HIST_NUM_BINS)
    for layer_name, layer in model.named_parameters():
        if not layer.requires_grad: continue
        
        # count parameters
        num_params = layer.numel()
        table_counts.add_row([layer_name, num_params])
        total_params+=num_params
        
        # get histogram of parameter values in one layer
        num_tensor = 0
        layer_hist = torch.zeros(HIST_NUM_BINS)
        for tensor in layer.data:
            tensor_abs = torch.abs(tensor)
            hist = torch.histc(tensor, bins=HIST_NUM_BINS, min=0, max=MAX_PARAM_VALUE)
            layer_hist = torch.add(layer_hist, hist)
            num_tensor+=1
        table_hist.add_row([layer_name, layer_hist])
        percent_big = ((num_params-layer_hist[0].item()) / num_params)*100
        model_hist = torch.add(model_hist, layer_hist)
        table_percents.add_row([layer_name, '{0}%'.format(round(percent_big, 2))])
    
    #print(table_counts)
    #print(table_hist)
    print(table_percents)
    percent_model_big = ((total_params-model_hist[0].item()) / total_params)*100
    print('Percent of model parameters > {0} is {1}%'.format(HIST_BIN_SIZE, round(percent_model_big, 2)))
    #print(model_hist)
    print("Total params: {0}".format(total_params))
    
    # plot histogram of parameter values in model overall
    plt.bar(x=np.arange(0, MAX_PARAM_VALUE, HIST_BIN_SIZE), height=model_hist.numpy(), align='edge')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Parameter Value (Absolute Value)')
    plt.ylabel('Number of Parameters')
    plt.title('Histogram of Absolute Values of Parameters - {0}'.format(name))
    plt.show()

# prunes model at a global level given the percentage to prune
# percent_prune must be a float between 0 and 1
def prune_model(model, percent_prune):
    parameters_to_prune = []    
    for name, module in model.named_modules():
        #print(type(module))
        if isinstance(module, transformers.pytorch_utils.Conv1D):
            parameters_to_prune.append((module, 'weight'))
        if isinstance(module, torch.nn.modules.linear.Linear):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=percent_prune)

    return model


def main():
    model_1 = GPT2Model.from_pretrained('gpt2') 
    model_2 = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
    model_3 = T5ForConditionalGeneration.from_pretrained("t5-small")

    models = [('GPT-2', model_1), ('BERT', model_2), ('T5', model_3)]

    for name, model in models:
        print('Starting model...')
        print('Starting analysis...')
        count_big_parameters(model, name)
        print('Starting pruning...')
        percent_prune = [0.1, .5, .9, .95, .99]
        for p in percent_prune:
            pruned_model = prune_model(model, p)


if __name__=="__main__":
    main()

