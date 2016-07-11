def check_k(ds_list, k):

    if not is_k_valid(ds_list, k):
        raise ValueError(
            'Parameter k must be <= minimum column dimension among views.')

def is_k_valid(ds_list, k):

    p = min([ds.cols() for ds in ds_list])

    return k <= p

