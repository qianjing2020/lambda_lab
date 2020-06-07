# grid search 
from math import sqrt
import numpy as np
from joblib import Parallel, delayed
from warnings import catch_warnings, filterwarnings
from sklearn.metrics import mean_squared_error
from modules.forecast_models import ForecastModels

fm = ForecastModels()

def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	for i in range(len(test)):
		# step over each time-step in test dataset
		""" change the model here:"""
		yhat = fm.exp_smoothing_forecast(history, cfg)
		predictions.append(yhat)
		# dump the finished test to history
		history.append(test[i])
	error = measure_rmse(test, predictions)
	return error

def measure_rmse(actual, predicted):
	# root mean squared error or rmse
	return sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
	# split a univariate dataset into train/test sets
	return data[:-n_test], data[-n_test:]

def score_model(data, n_test, cfg, debug=False):
	# score a model, return None on failure, else return RMSE
	result = None
	# convert config to a key
	key = str(cfg)
	if debug:
		# show all warnings and fail on exception when debugging
		result = walk_forward_validation(data, n_test, cfg)
	else:		
		try:
			# one failure during model validation suggests an unstable config
			with catch_warnings():
				# never show warnings when grid searching, too noisy
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	if result is not None:
		# check for an interesting result
		print(f'Model %s  %.3f' % (key, result))
	# pair up config and error
	return (key, result)


def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=-1, backend='loky')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results: None if there is an error for scoring RMSE
	scores = [r for r in scores if r[1] != None]
	# sort by error, ascending
	scores.sort(key=lambda tup: tup[1])
	return scores





if __name__ == '__main__':

	data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

	n_test = 10 # number of observation used for test 

	# model configs
	max_length = len(data) - n_test # max length used as history
	cfg_list = simple_configs(max_length)

	# grid search
	scores = grid_search(data, cfg_list, n_test)

	# list top 10 configs
	for cfg, error in scores[:10]:
		print(cfg, error)
	
