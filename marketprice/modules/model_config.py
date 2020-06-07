class ModelConfig:
    def __init__(self):
        pass

    def simple_configs(max_length, seasonal=[1]):
        # create a set of configs, offset is seasonality
        configs = list()
        for i in range(1, max_length + 1):
            # number of observation used as history
            for o in seasonal:
                for t in ['persist', 'mean', 'median']:
                    cfg = [i, o, t]
                    configs.append(cfg)
        return configs

    
    def exp_smoothing_configs(seasonal=[None]): models = list()
    # define config lists
        t_params = ['add', 'mul', None]
        d_params = [True, False] s_params = ['add', 'mul', None]
        p_params = seasonal b_params = [True, False]
        r_params = [True, False]
        # create config instances 
        for t in t_params:
            for d in d_params:
                for s in s_params:
                    for p in p_params:
                        for r in r_params:
                            cfg = [t, d, s, p, b, r]
                            models.append(cfg)
        return models