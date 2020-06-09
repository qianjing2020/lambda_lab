import context
import matplotlib.pyplot as plt
from modules.grid_search import grid_search
from modules.forecast_models import ModelConfig
from test_sequence import sale

# grid search
data = sale.to_numpy().flatten()

n_test = 7 # number of observation used for test 
max_length = len(data) - n_test  # max length used as history

mc = ModelConfig()
# config list for one time series
cfg_list = mc.exp_smoothing_configs(seasonal=[0, 12])

#print(f'{len(cfg_list)} configurations: {cfg_list}')

# grid search
scores = grid_search(data, cfg_list, n_test)

# list top 1 configs
# for cfg, error in scores[:10]:
#     print(cfg, error)
# top score
cfg_selected, error = scores[1]
result = [cfg_selected, error]
breakpoint()

numpy.savetxt("hw_rice.csv", result, delimiter=",")