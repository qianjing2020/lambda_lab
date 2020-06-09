Based on exploritory model performance, 
statistical method outperform deep learning models. 

So we use grid search for a statistical method: Holt-Winters Exponential Smoothing provided by Statsmodel lilbary. 

When the model optimization set to true, the model automatically tune hyperparameters (smoothing level, smoothing slope, smoothing seasonal, and damping slope). 

The following process describes how to gird search for other hyperparameters (trend, damped, seasonal, seasonal perids, use_boxcox).

The Holter-Winters parameters tested are: trend type, dampening type, seasonality type, seasonal period, Box-Cox transform, removal of the bias when fitting. The RMSE is used to evaluate the model performance. The best performance model is selected for each time series. 

The qc_id, best parameters, and RMSE are save to database table 'hw_params_wholesale' and 'hw_params_retail'
        
