import context
from modules.grid_search import grid_search, simple_configs
from test_sequence import sale

data = sale.to_numpy().flatten()

n_test = 7 # number of observation used for test 

# model configs
max_length = len(data) - n_test # max length used as history
cfg_list = simple_configs(max_length)
breakpoint()
# grid search
scores = grid_search(data, cfg_list, n_test)

# list top 10 configs
for cfg, error in scores[:10]:
    print(cfg, error)