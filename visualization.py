import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def plot_embeddings(embeddings, targets, classes, xlim=None, ylim=None):
    cuda = torch.cuda.is_available()
    np.random.seed(7)
    colors = np.array(sns.color_palette("Paired"))
    np.random.shuffle(colors)
    
    plt.figure(figsize=(10,10))
    
    for i in range(len(colors)):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)

    
def extract_embeddings(dataloader, model):
    cuda = torch.cuda.is_available()
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader), 2))
        labels = np.zeros(len(dataloader))
        k = 0
        for samples, target in dataloader:
            if cuda:
                samples = samples.cuda()
            if isinstance(model, torch.nn.DataParallel):
                embeddings[k:k+len(samples)] = model.module.get_embedding(samples).data.cpu().numpy()
            else:
                embeddings[k:k+len(samples)] = model.get_embedding(samples).data.cpu().numpy()
            labels[k:k+len(samples)] = target.numpy()
            k += len(samples)
    return embeddings, labels
