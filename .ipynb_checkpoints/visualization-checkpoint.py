import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def plot_embeddings(embeddings, targets, classes, xlim=None, ylim=None):
    cuda = torch.cuda.is_available()
    np.random.seed(7)
    colors = sns.color_palette("Paired")
    col_order = [ 5,  1, 7,  9,  2,  10,  0,  3, 11,  4,  6,  8]
    colors = [colors[i] for i in col_order]
    
    plt.figure(figsize=(10,10))
    
    for i in range(len(colors)):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)


def sns_plot_embeddings(embeddings, labels, classes, **kwargs):
    sns.set_style("ticks")
    palette = {classes[i]:sns.color_palette('Paired')[i] for i in classes.keys()}
    data = pd.DataFrame(embeddings)
    data['label'] = labels.reshape(-1,1).astype('int')
    data['meta'] = data['label'].map(classes).astype('category')
    data[kwargs['style']] = [i[1] for i in data['meta'].str.split(":")]
    # plot
    sns.scatterplot(x=0, y=1, hue=kwargs['hue'], style=kwargs['style'], palette=palette, data=data,
                s=100, alpha=kwargs['alpha'])

    
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
