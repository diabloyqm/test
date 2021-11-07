import torch.nn as nn
from torch.utils.data import DataLoader
import torch
def test_inference(args, model, test_dataset):

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cpu'
    loss_function = nn.NLLLoss().to(device)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    for batch_index, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        # infer
        outputs = model(images)
        batch_loss = loss_function(outputs, labels)
        loss += batch_loss.item()

        # prediction
        _, predict_labels = torch.max(outputs,1)
        predict_labels = predict_labels.view(-1)
        correct += torch.sum(torch.eq(predict_labels, labels)).item()
        total +=len(labels)

    accuracy = correct/total
    return accuracy, loss
