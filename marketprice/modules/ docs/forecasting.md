Based on exploritory model performance, 
statistical method outperform deep learning models. 

Here we use grid search for a statistical method: Holt-Winters Exponential Smoothing provided by Statsmodel lilbary. 

The model tune hyperparameters ( smoothing level, smoothing slope, smoothing seasonal, and damping slope) automatically when set optimization=true. 

The following process describe how to gird search for other hyperparameters (trend, damped, seasonal, seasonal perids, use_boxcox).

