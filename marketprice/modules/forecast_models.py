import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ForecastModels:
	def __init__(self):
		pass

	def simple_forecast(self, history, config):
		# one-step baseline models
		n, offset, avg_type = config
		"""
		n: number of observation in history used for forecast
		offset: seasonality
		avg_type: how to average the predictions"""
		if avg_type == 'persist':
			# naive method
			return history[n]  # observation n
		# collect values to average
		values = list()
		if offset == 1:
			# no seasonality
			values = history[n:]  # last n observations
		else:
			if n * offset > len(history):
				# skip bad configs
				raise Exception(
				    f'Config beyond end of data: {n: %d} *{offset: %d} > {len(history)}')
			for i in range(1, n + 1):
				# try and collect n values using offset
				idx = i * offset
				values.append(history[idx])  # last n observations spaced by offset
		# check if we can average
		if len(values) < 2:
			raise Exception('Cannot calculate average')
		# mean of last n values
		if avg_type == 'mean':
			return np.mean(values)
		# median of last n values
		return np.median(values)

	def exp_smoothing_forecast(self, history, config):
		# one-step Holt Winter's Exponential Smoothing forecast
		t, d, s, p, b, r = config
		# in config: trend type, dampening type, seasonality type, seasonal period, Box-Cox transform, removal of the bias when fitting

		history = np.array(history)
		model = ExponentialSmoothing(
		    history, trend=t, damped=d, seasonal=s, seasonal_periods=p)

		model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
		# make one step forecast
		yhat = model_fit.predict(len(history), len(history))
		return yhat[0]


class ModelConfig:
    # model configurations, options for hyperparameters
    def __init__(self):
        pass

    def simple_configs(self, max_length, seasonal=[1]):
        # create a set of configs, offset is seasonality
        configs = list()
        for i in range(1, max_length + 1):
            # number of observation used as history
            for o in seasonal:
                for t in ['persist', 'mean', 'median']:
                    cfg = [i, o, t]
                    configs.append(cfg)
        return configs
		
    def exp_smoothing_configs(self, seasonal=[None]):
        configs = list()
        t_params = ['add', 'mul', None]
        d_params = [True, False]
        s_params = ['add', 'mul', None]
        p_params = seasonal
        b_params = [True, False]
        r_params = [True, False]
        # create config instances 
        for t in t_params:
            for d in d_params:
                for s in s_params:
                    for p in p_params:
                        for b in b_params:
                            for r in r_params:
                                cfg = [t, d, s, p, b, r]
                                configs.append(cfg)
        return configs
