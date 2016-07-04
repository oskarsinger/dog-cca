import numpy as np

from optimization.utils import get_gram

def get_batch_and_gram_lists(ds_list, gs_list):

    batch_list = [ds.get_data()
                  for ds in ds_list]
    gram_list = [gs.get_gram(batch)
                 for (gs, batch) in zip(gs_list, batch_list)]

    return (batch_list, gram_list)

def init_data(ds_list, gs_list, online=False):

    (Xs, Sxs) = get_batch_and_gram_lists(ds_list, gs_list)

    if not online:
        # Find a better solution to this
        n = min([X.shape[0] for X in Xs])

        # Remove to-be-truncated examples from Gram matrices
        removed = [X[n:] if X.ndim == 1 else X[n:,:] for X in Xs]
        Sxs = [Sx - get_gram(r) for r, Sx in zip(removed, Sxs)]

        # Truncate extra examples
        Xs = [X[:n] if X.ndim == 1 else X[:n,:] for X in Xs]

    return (Xs, Sxs)

