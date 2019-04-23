import numpy as np
import torch

def mcdue_sampling(size, model, dataloader, use_gpu=torch.cuda.is_available(), T=100):
    
    y_pred = []
    model.freeze_mask_dropout()
    with torch.autograd.no_grad():
        for i in tqdm(range(T), desc='Testing model'):
            for inputs, ys in dataloader:
                if use_gpu:
                    inputs = [tsr.cuda() for tsr in inputs]
                    ys = ys.cuda()

                outputs = model(inputs)
                y_pred.append(outputs.tolist())

    stds = np.std(y_pred, axis=0)
    sorted_idx = np.argsort(stds)[::-1]
    return sorted_index[:size]

def random_sampling(size, model, dataloader, use_gpu=False):
    idx = np.random.permutation(np.arange(dataloader.dataset.__len__()))
    return idx[:size]