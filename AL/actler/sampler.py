import numpy as np
import torch
from tqdm import tqdm

def mcdue_sampling(size, model, dataloader, use_gpu=torch.cuda.is_available(), T=10):
    
    y_preds = []
    with torch.autograd.no_grad():
        for i in tqdm(range(T), desc='Testing model'):
            model.freeze_mask_dropout(new_p=0.5)
            y_pred = []
            for inputs, ys in dataloader:
                if use_gpu:
                    inputs = [tsr.cuda() for tsr in inputs]
                    ys = ys.cuda()

                outputs = model(inputs)
                y_pred.append(outputs.tolist())
            y_preds.append(np.concatenate(y_pred))
    model.unfreeze_mask_dropout()
    stds = np.std(y_preds, axis=0)
    sorted_idx = np.argsort(stds.ravel())[::-1]
    return sorted_idx[:size]

def random_sampling(size, model, dataloader, use_gpu=False):
    idx = np.random.permutation(np.arange(dataloader.dataset.__len__()))
    return idx[:size]