import torch

loss = torch.nn.BCELoss()
avg_loss = 0.0

# inputs shape [B,2]
def CustomLoss(inputs, targets):
    for i in range(len(inputs)):
        l = loss(inputs[i], targets[i])
        avg_loss += l

    return avg_loss


