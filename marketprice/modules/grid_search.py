# grid search 
from math import sqrt
import numpy as np
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error

def simple_forecast(history, config):
	n, offset, avg_type = config
	""" 
	n: number of observation in history used for forecast
	offset: seasonality
	avg_type: how to average the predictions"""
	if avg_type == 'persist':
		# naive method
		return history[-n] # observation n
	# collect values to average
	values = list()
	if offset == 1:
		# no seasonality
		values = history[-n:] # last n observations
	else:
		if n * offset > len(history):
			# skip bad configs
			raise Exception(f'Config beyond end of data: {n: %d} *{offset: %d} > {len(history)}')
		for i in range(1, n + 1):
			# try and collect n values using offset		
			idx = i * offset
			values.append(history[-idx]) # last n observations spaced by offset
	# check if we can average
	if len(values) < 2:
		raise Exception('Cannot calculate average')
	# mean of last n values
	if avg_type == 'mean':
		return np.mean(values)
	# median of last n values
	return np.median(values)

def measure_rmse(actual, predicted):
	# root mean squared error or rmse
	return sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
	# split a univariate dataset into train/test sets
	return data[:-n_test], data[-n_test:]

def walk_forward_validation(data, n_test, cfg):
	# walk-forward validation for univariate data
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = simple_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

def score_model(data, n_test, cfg, debug=False):
	# score a model, return None on failure, else return rmse 
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

def simple_configs(max_length, offsets=[1, 2]):
	# create a set of configs, offset is seasonality
	configs = list()
	for i in range(1, max_length + 1):
		# number of observation used as history
		for o in offsets:
			for t in ['persist', 'mean', 'median']:
				cfg = [i, o, t]
				configs.append(cfg)
	return configs

if __name__ == '__main__':

	data = sale.to_numpy().flatten()

	n_test = 7 # number of observation used for test 

	# model configs
	max_length = len(data) - n_test # max length used as history
	cfg_list = simple_configs(max_length)

	# grid search
	scores = grid_search(data, cfg_list, n_test)

	# list top 10 configs
	for cfg, error in scores[:10]:
		print(cfg, error)
	
