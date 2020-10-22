import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, n_layers, n_inputs, n_hidden_units, n_classes):
        super(Classifier, self).__init__()
        layers = []
        
        if n_layers == 1: # Logistic Regression
            if n_classes > 2:
                layers.append(nn.Linear(n_inputs, n_classes))
            else:
                layers.append(nn.Linear(n_inputs, 1))
        else: # Neural Networks
            layers.append(nn.Linear(n_inputs, n_hidden_units))
            layers.append(nn.ReLU())
            for i in range(n_layers-2):
                layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                layers.append(nn.ReLU())
            
            if n_classes > 2:
                layers.append(nn.Linear(n_hidden_units, n_classes))
            else:
                layers.append(nn.Linear(n_hidden_units, 1))
        
        if n_classes > 2:
            layers.append(nn.Softmax(dim=1))
        else:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
#

class Discriminator(nn.Module):
    def __init__(self, n_layers, n_hidden_units, n_classes, n_sensitive_attrs, EO_option=False):
        super(Discriminator, self).__init__()
        
        n_inputs = n_classes if n_classes > 2 else 1
        n_outputs = n_sensitive_attrs if n_sensitive_attrs > 2 else 1
        
        layers = []
        if n_layers == 1: # Logistic Regression
            if EO_option:
                layers.append(nn.Linear(n_inputs+1, n_outputs))
            else:
                layers.append(nn.Linear(n_inputs, n_outputs))
        elif n_layers == 2:
            if EO_option:
                layers.append(nn.Linear(n_inputs+1, n_hidden_units))
            else:
                layers.append(nn.Linear(n_inputs, n_hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden_units, n_outputs))
        
        if n_outputs == 1:
            layers.append(nn.Sigmoid())
        elif n_outputs > 1:
            layers.append(nn.Softmax(dim=1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)