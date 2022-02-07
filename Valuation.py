from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
from pandas import DataFrame

# Gather Data
boston_dataset = load_boston()
data = DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
features = data.drop(["INDUS", "AGE"], axis=1)

log_prices = np.log(boston_dataset.target)
target = DataFrame(data=log_prices, columns=["PRICE"])

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

property_stats = np.ndarray(shape=(1, 11))
property_stats = features.mean().values.reshape(1, 11)

regr = LinearRegression()
regr.fit(features, target)

# Calculatig the MSE and RMSE
MSE = mean_squared_error(target, regr.predict(features))
RMSE = np.sqrt(mean_squared_error(target, regr.predict(features)))

def get_log_estimate(nr_rooms, student_per_classroom, next_to_river=False,high_confidence=True):

    # Configure Property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = student_per_classroom

    if next_to_river:
        property_stats[0][CHAS_IDX] = 1

    else:
        property_stats[0][CHAS_IDX] = 0


    # Make Predictions
    log_estimate = regr.predict(property_stats)[0][0]

    # Calculate Range
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95

    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68

    return log_estimate, upper_bound, lower_bound, interval

def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):

    """ Estimate a Property Price in Boston.

        KEYWORD ARGUMENTS
        rm -- Number of rooms in the Property.
        ptratio -- Number of Students per Teacher in a classroom  for the School in the Area.
        chas -- True if Property is next to River, otherwise False.
        large_range -- True for 95% prediction interval , False for a 68% interval.

    """

    if rm < 1 or ptratio < 1:
        print("That is UNREALISTIC, Try again !")
        return

    log_est, upper, lower, conf = get_log_estimate(nr_rooms=rm, student_per_classroom=ptratio,
                                                next_to_river=chas, high_confidence=large_range)

    # Converting Log Price to Today's Dollar Price
    dollar_log_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_upper_est = np.e**upper * 1000 * SCALE_FACTOR
    dollar_lower_est = np.e**lower * 1000 * SCALE_FACTOR

    # Rounding Dollar estimate to the Nearest Thousands
    rounded_est = np.around(dollar_log_est, -3)
    rounded_upper_est = np.around(dollar_upper_est, -3)
    rounded_lower_est = np.around(dollar_lower_est, -3)

    print("THE ESTIMATED PROPERTY PRICE: ", rounded_est)
    print(f"At {conf}% Confidence, the valuation Range is:")
    print(f"USD {rounded_upper_est} at the UPPER and USD {rounded_lower_est} at the LOWER")

get_dollar_estimate(10, 30)