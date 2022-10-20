####################################################################################
# Useful classes and functions for QBUS6840 - Predictive Analytics Group Project
## Group 51
## Last updated 14/10/2022

### The file containts the following:
### Functions:
###    1. MSE: Mean Squared Error
###    2. RMSE: Root Mean Squared Error
###    3. MAPE: Mean Absolute Percentage Error
###    4. ts_cross_val_split: Time Series Cross-Validation Split
###        -> Create cross-validation sets to evaluate, tune and select models
###
### Classes:
###    1. SimpleModel
###        - Create a .fit() and .predict() confidence function for baseline methods

####################################################################################
import numpy as np

def MSE(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true-y_pred)**2)

def RMSE(y_true, y_pred):
    """Root Mean Squared Error"""
    return MSE(y_true, y_pred)**(1/2)

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true-y_pred)/y_true))

def ts_cross_val_split(df, splits=5, forecast_window=24, step=12):
    """Create an expanding n-fold cross-validation training split to evaluate model and tune hyper-parameters.
    Args:
        df (DataFrame): Time series data for train-val split
        splits (int): Number of splits
        forecast_window (int): Number of months for forecasting / length of the validation dataset
        step (int): Number of months to increase after each fold (i.e. expansion window)
    Returns:
        train_folds (dict): Dictionary of n-splits for each training fold
        val_folds (dict): Dictionary of each corresponding validation set
    """
    train_folds, val_folds = dict(), dict()
    for n in range(splits):
        train_folds[n] = df[:-(splits-n)*12 - forecast_window] # Index the expansion to the start of validation
        val_folds[n] = df[-(splits-n)*12 - forecast_window:] # Start from the validation set
        val_folds[n] = val_folds[n][:forecast_window] # Limit the validation set to the window size
        
    print(f'Created {splits} splits for cross-validation. Expanding window = {step} months | Forecast window = {forecast_window} months')
    
    return train_folds, val_folds

class SimpleModel():
    """Baseline models for forecasting allowing to fit, predict and create confidence intervals"""

    def __init__(self, model_type):
        avail_model_types = ['average', 'naive', 'drift', 'growth']
        assert model_type in avail_model_types, f"Model must by in type: {avail_model_types}"
        self.model_type = model_type
        
        # Convert the CI interval to standard deviation assuming normal distribution
        self.conf_intervals = {
            0.50: 0.67,
            0.55: 0.76,
            0.60: 0.84,
            0.65: 0.93,
            0.70: 1.04,
            0.75: 1.15,
            0.80: 1.28,
            0.85: 1.44,
            0.90: 1.64,
            0.95: 1.96,
            0.96: 2.05,
            0.97: 2.17,
            0.98: 2.33,
            0.99: 2.58
        }
 
    def fit(self, X):
        """Fit model depending on type and store key information about the fit dataset"""
        self.train_0 = X.iloc[0].item()
        self.train_t = X.iloc[-1].item()
        self.len_t = len(X)
        self.mean = np.mean(X, axis=None)
        self.std = np.std(X, axis=None)
       
        if self.model_type == 'average':
            self.prediction = self.mean           

        elif self.model_type == 'naive':
            self.prediction = self.train_t           

        elif self.model_type == 'drift':
            self.prediction = (self.train_t - self.train_0) / (self.len_t - 1)           

        elif self.model_type == 'growth':
            #Cum. Growth Rate = (V_final/V_begin)**(1/t) - 1
            cum_growth = (self.train_t / self.train_0) ** (1 / self.len_t) - 1
            self.prediction = (1 + cum_growth)           

        return print('Model fit')

    def predict(self, h_steps):
        """Predict (or forecast) using the model prediction type and h_steps ahead"""

        if self.model_type in ['average', 'naive']:
            return np.array([self.prediction]*h_steps)    

        elif self.model_type == 'drift':
            return np.array([self.train_t + (1+i)*self.prediction for i in range(h_steps)])

        elif self.model_type == 'growth':
            return np.array([self.train_0 * self.prediction**(self.len_t+i) for i in range(1, h_steps+1)])      

    def predict_conf(self, h_steps, CI=0.95):
        """Include Confidence Interval to prediction (default is 95% = +/- 2 * SD)"""
        
        assert CI in self.conf_intervals, "The rate is not included. Please check that it is in self.conf_intervals"
        
        predictions = self.predict(h_steps)
        multiplier = self.conf_intervals[CI]

        if self.model_type in ('average', 'drift'):
            upper = np.array([predictions + multiplier*self.std])
            lower = np.array([predictions - multiplier*self.std])            

        elif self.model_type in ('naive', 'growth'):
            upper = np.sum([predictions, np.array([multiplier*self.std*(i+1) for i in range(h_steps)])], axis=0)
            lower = np.sum([predictions, np.array([-multiplier*self.std*(i+1) for i in range(h_steps)])], axis=0)     

        # Predictions shouldn't go below 0
        lower[lower<0] = 0
        return (predictions, upper, lower)