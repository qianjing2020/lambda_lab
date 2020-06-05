import numpy as np

def split_sequence(sequence, n_steps_in, n_steps_out):
    """split a univariate sequence into samples, input sequence: numpy array, n_steps: number of time steps shifted to create features, or number of variable cols generated, or number of lags in statistic terms """

    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this sequence
        end_idx = i + n_steps_in
        out_end_idx = end_idx + n_steps_out
        # check if we are beyond the sequence
        if out_end_idx > len(sequence):
            break
        # gather input and output parts of the sequence
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx:out_end_idx]
     
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)