import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

    
def svm_model(X, Y, model, epoch, batch, weight, b, c=0.01):
    hid = (weight * X).sum(1) + b
    return hid
    # X = torch.FloatTensor(X)
    # Y = torch.FloatTensor(Y)
    # N = len(Y)

    # optimizer = optim.SGD(model.parameters(), lr=c)
    # model.train()
    
    # for epoch in range(epoch):
    #     perm = torch.randperm(N)
    #     sum_loss = 0
    #     for i in range(0, N, batch):
    #         x = X[perm[i : i + batch]]
    #         y = Y[perm[i : i + batch]]
    
    #         optimizer.zero_grad()
    #         output = model(x).squeeze()
    #         weight = model.weight.squeeze()
            
    #         loss = torch.mean(torch.clamp(1 - y * output, min = 0))
    #         loss += c * (weight.t() @ weight) / 2.0
    #         loss.backward()
    
    #         optimizer.step()
    #         sum_loss += float(loss)
        
    #     print("Linear SVM Model Epoch : {:4d}\tloss : {}".format(epoch, sum_loss / N))