import random
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from utils import get_fairness_metrics, BCELossAccuracy, CELossAccuracy, FairnessLoss


def train(dataset, net, optimizer, lr_scheduler, fairness, lambda_,
          h, tau, delta, device, n_epochs=200, batch_size=2048, seed=0):
    n_classes = dataset.n_classes
    n_sensitive_attrs = dataset.n_sensitive_attrs
    sensitive_attrs = dataset.sensitive_attrs
    data_loader = dataset.get_data_loader(batch_size=batch_size)
    XZ_test, Y_test, Z_test = dataset.get_test_data()
    XZ_test_, Y_test_, _ = dataset.get_test_tensor()

    # An empty dataframe for logging experimental results
    df = pd.DataFrame()
    df_ckpt = pd.DataFrame()
    
    cross_entropy_loss = CELossAccuracy() if n_classes > 2 else BCELossAccuracy()
    fairness_loss = FairnessLoss(h, tau, delta, fairness, n_classes, n_sensitive_attrs, sensitive_attrs)
    costs = []
    for epoch in range(n_epochs):
        for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
            xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
            Yhat = net(xz_batch)
            Ytilde = torch.round(Yhat.detach().reshape(-1))
            cost = 0

            # prediction loss
            p_loss, acc = cross_entropy_loss(Yhat.squeeze(), y_batch)
            cost += (1 - lambda_) * p_loss

            f_loss, f_loss_item = fairness_loss(Yhat, y_batch, z_batch)
            cost += lambda_ * f_loss

            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()
            costs.append((1 - lambda_) * p_loss.item() + lambda_ * f_loss_item)
            
            # Print the cost per 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == len(data_loader):
                print('Epoch [{}/{}], Batch [{}/{}], Cost: {:.4f}'.format(epoch+1, n_epochs,
                                                                          i+1, len(data_loader),
                                                                          cost.item()), end='\r')
        if lr_scheduler is not None:
            lr_scheduler.step()
    
    Yhat_test = net(XZ_test_.to(device)).squeeze()
    p_loss, acc = cross_entropy_loss(Yhat_test, Y_test_.to(device))
    if n_classes > 2:
        Ytilde_test = Yhat_test.detach().cpu().numpy().argmax(axis=1)
    else:
        Ytilde_test = (Yhat_test.detach().cpu().numpy() > 0.5).astype(float)
    DDP, DEO = get_fairness_metrics(Y_test, Z_test, Ytilde_test, n_classes, n_sensitive_attrs)
    
    return acc, DDP, DEO