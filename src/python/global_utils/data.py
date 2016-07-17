import numpy as np

from optimization.utils import get_gram
from data.missing import MissingData

def get_batch_and_gram_lists(ds_list, gs_list, Xs, Sxs):

    new_batch_list = [ds.get_data()
                      for ds in ds_list]
    missing = [type(batch) is MissingData 
               for batch in new_batch_list]
    batch_list = [Xs[i] if missing[i] else new_batch_list[i]
                  for i in xrange(new_batch_list)]
    get_gram_update = lambda i: gs_list[i].get_gram(new_batch_list[i])
    gram_list = [Sxs[i] if missing[i] else get_gram_update(i)
                 for i in xrange(gs_list)]

    return (batch_list, gram_list, missing)

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

