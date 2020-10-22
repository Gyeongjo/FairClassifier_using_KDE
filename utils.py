import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn


def normal_pdf(x):
    import math
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def normal_cdf(y, h=0.01, tau=0.5):
    # Approximation of Q-function given by López-Benítez & Casadevall (2011)
    # based on a second-order exponential function & Q(x) = 1 - Q(-x):
    Q_fn = lambda x: torch.exp(-0.4920*x**2 - 0.2887*x - 1.1893)

    m = len(y)
    y_prime = (tau - y) / h
    sum_ = torch.sum(Q_fn(y_prime[y_prime > 0])) \
           + torch.sum(1 - Q_fn(torch.abs(y_prime[y_prime < 0]))) \
           + 0.5 * len(y_prime[y_prime == 0])
    return sum_ / m

def Huber_loss(x, delta):
    if abs(x) < delta:
        return (x ** 2) / 2
    return delta * (x.abs() - delta / 2)

def Huber_loss_derivative(x, delta):
    if x > delta:
        return delta
    elif x < -delta:
        return -delta
    return x

def get_fairness_metrics(Y, Z, Ytilde, n_classes, n_sensitive_attrs):
    DDP = 0
    DEO = 0
    for y in range(n_classes):
        Pr_Ytilde_y = (Ytilde == y).mean()
        Ytilde_y_given_Y_y = np.logical_and(Ytilde==y, Y==y)
        for z in range(n_sensitive_attrs):
            DDP += abs(np.logical_and(Ytilde==y, Z==z).mean() / (Z==z).mean() - Pr_Ytilde_y)
            DEO += abs(np.logical_and(Ytilde_y_given_Y_y, Z==z).mean() / np.logical_and(Y==y, Z==z).mean() - Ytilde_y_given_Y_y.mean() / (Y==y).mean())
    return DDP, DEO


class BCELossAccuracy():
    def __init__(self):
        self.loss_function = nn.BCELoss()
    
    @staticmethod
    def accuracy(y_hat, labels):
        with torch.no_grad():
            y_tilde = (y_hat > 0.5).int()
            accuracy = (y_tilde == labels.int()).float().mean().item()
        return accuracy
    
    def __call__(self, y_hat, labels):
        loss = self.loss_function(y_hat, labels)
        accuracy = self.accuracy(y_hat, labels)
        return loss, accuracy
#

class CELossAccuracy():
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()
    
    @staticmethod
    def accuracy(y_hat, labels):
        with torch.no_grad():
            y_tilde = y_hat.argmax(axis=1)
            accuracy = (y_tilde == labels).float().mean().item()
        return accuracy
    
    def __call__(self, y_hat, labels):
        loss = self.loss_function(y_hat, labels)
        accuracy = self.accuracy(y_hat, labels)
        return loss, accuracy
#
       
class FairnessLoss():
    def __init__(self, h, tau, delta, notion, n_classes, n_sensitive_attrs, sensitive_attrs):
        self.h = h
        self.tau = tau
        self.delta = delta
        self.fairness_notion = notion
        self.n_classes = n_classes
        self.n_sensitive_attrs = n_sensitive_attrs
        self.sensitive_attrs = sensitive_attrs
        if self.n_classes > 2:
            self.tau = 0.5
        assert self.fairness_notion in ['DP', 'EO']
    
    def DDP_loss(self, y_hat, Z):
        m = y_hat.shape[0]
        backward_loss = 0
        logging_loss = 0
        
        if self.n_classes == 2:
            Pr_Ytilde1 = normal_cdf(y_hat.detach(), self.h, self.tau)
            for z in self.sensitive_attrs:
                Pr_Ytilde1_Z = normal_cdf(y_hat.detach()[Z==z], self.h, self.tau)
                m_z = Z[Z==z].shape[0]
               
                Prob_diff_Z = Pr_Ytilde1_Z - Pr_Ytilde1
                
                _dummy = \
                    torch.dot(
                        normal_pdf((self.tau - y_hat.detach()[Z==z]) / self.h).view(-1),
                        y_hat[Z==z].view(-1)
                    ) / (self.h * m_z) -\
                    torch.dot(
                        normal_pdf((self.tau - y_hat.detach()) / self.h).view(-1),
                        y_hat.view(-1)
                    ) / (self.h * m)
               
                _dummy *= Huber_loss_derivative(Prob_diff_Z, self.delta)
                
                backward_loss += _dummy
                
                logging_loss += Huber_loss(Prob_diff_Z, self.delta)
                
        else:
            idx_set = list(range(self.n_classes)) if self.n_classes > 2 else [0]
            for y in idx_set:
                Pr_Ytilde1 = normal_cdf(y_hat[:,y].detach(), self.h, self.tau)
                for z in self.sensitive_attrs:
                    Pr_Ytilde1_Z = normal_cdf(y_hat[:,y].detach()[Z==z], self.h, self.tau)
                    m_z = Z[Z==z].shape[0]

                    Prob_diff_Z = Pr_Ytilde1_Z - Pr_Ytilde1
                    _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
                    _dummy *= \
                        torch.dot(
                            normal_pdf((self.tau - y_hat[:,y].detach()[Z==z]) / self.h).view(-1),
                            y_hat[:,y][Z==z].view(-1)
                        ) / (self.h * m_z) -\
                        torch.dot(
                            normal_pdf((self.tau - y_hat[:,y].detach()) / self.h).view(-1),
                            y_hat[:,y].view(-1)
                        ) / (self.h * m)

                    backward_loss += _dummy
                    logging_loss += Huber_loss(Prob_diff_Z, self.delta).item()
        return backward_loss, logging_loss
    
    def DEO_loss(self, y_hat, Y, Z):
        backward_loss = 0
        logging_loss = 0
        
        if self.n_classes == 2:
            for y in [0,1]:
                Pr_Ytilde1_Y = normal_cdf(y_hat[Y==y].detach(), self.h, self.tau)
                m_y = (Y==y).sum().item()
                for z in self.sensitive_attrs:
                    Pr_Ytilde1_YZ = normal_cdf(y_hat[torch.logical_and(Y==y, Z==z)].detach(), self.h, self.tau)
                    m_zy = torch.logical_and(Y==y, Z==z).sum().item()

                    Prob_diff_Z = Pr_Ytilde1_YZ - Pr_Ytilde1_Y
                    _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
                    _dummy *= \
                        torch.dot(
                            normal_pdf((self.tau - y_hat[torch.logical_and(Y==y, Z==z)].detach()) / self.h).view(-1), 
                            y_hat[torch.logical_and(Y==y, Z==z)].view(-1)
                        ) / (self.h * m_zy) -\
                        torch.dot(
                            normal_pdf((self.tau - y_hat[Y==y].detach()) / self.h).view(-1), 
                            y_hat[Y==y].view(-1)
                        ) / (self.h * m_y)

                    backward_loss += _dummy
                    logging_loss += Huber_loss(Prob_diff_Z, self.delta).item()
        else:
            for y in range(self.n_classes):
                Pr_Ytilde1_Y = normal_cdf(y_hat[:,y][Y==y].detach(), self.h, self.tau)
                m_y = (Y==y).sum().item()
                for z in self.sensitive_attrs:
                    Pr_Ytilde1_YZ = normal_cdf(y_hat[:,y][torch.logical_and(Y==y, Z==z)].detach(), self.h, self.tau)
                    m_zy = torch.logical_and(Y==y, Z==z).sum().item()

                    Prob_diff_Z = Pr_Ytilde1_YZ - Pr_Ytilde1_Y
                    _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
                    _dummy *= \
                        torch.dot(
                            normal_pdf((self.tau - y_hat[:,y][torch.logical_and(Y==y, Z==z)].detach()) / self.h).view(-1), 
                            y_hat[:,y][torch.logical_and(Y==y, Z==z)].view(-1)
                        ) / (self.h * m_zy) -\
                        torch.dot(
                            normal_pdf((self.tau - y_hat[:,y][Y==y].detach()) / self.h).view(-1), 
                            y_hat[:,y][Y==y].view(-1)
                        ) / (self.h * m_y)

                    backward_loss += _dummy
                    logging_loss += Huber_loss(Prob_diff_Z, self.delta).item()
        return backward_loss, logging_loss
    
    def __call__(self, y_hat, Y, Z):
        if self.fairness_notion == 'DP':
            return self.DDP_loss(y_hat, Z)
        else:
            return self.DEO_loss(y_hat, Y, Z)