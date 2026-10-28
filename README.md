# Flight Delay Predictor

Predicts whether a US flight will arrive 15 or more minutes late, using only information known before departure.

## Problem

About a quarter of flights arrive at least 15 minutes late. The goal is to flag likely late flights before they leave, so every input must be known at prediction time.

## Data

Source: US Bureau of Transportation Statistics, Reporting Carrier On-Time Performance, June 2026.
The raw file has 607,577 flights. After removing 10,019 cancelled and 2,363 diverted flights (which have no arrival delay), 595,195 flights remain.
The data file is not stored in this repo because of its size. Download one month from the BTS site and place it at data/T_ONTIME_REPORTING.csv.

## Approach

1. Cleaned the data and created the target: ARR_DEL15 (1 means 15 or more minutes late).
2. Created features: departure hour, day of week, distance, and airline (one hot encoded).
3. Split by date, not randomly: training on June 1 to 21 (414,427 flights), testing on June 22 to 30 (180,768 flights). This mimics predicting the future.
4. Trained logistic regression, then repeated it with class weighting to handle the imbalance.

## Avoiding data leakage

Departure delay, arrival delay, cancelled, and diverted are excluded as inputs. They are only known after the plane leaves, so using them would make the model look strong while being useless in practice.

## Results (test set)

| Model | Accuracy | Recall | Precision | ROC AUC |
| Always predict on time | 0.761 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.756 | 0.071 | 0.437 | 0.666 |
| Logistic regression, balanced classes | 0.635 | 0.612 | 0.349 | 0.666 |

What this shows:
* Accuracy alone is misleading. Always guessing on time scores 76% and catches no late flights.
* The model learned a real but modest signal (AUC 0.67 versus 0.50 for a coin flip). Weather and aircraft data are not in this dataset.
* Class weighting raises recall from 7% to 61% at the cost of precision, a tradeoff between missing late flights and raising false alarms.

## Patterns found in June 2026

* Sunday had the highest late rate (31.4%) and Tuesday the lowest (19.7%), against 25.7% overall.
* Late rates vary by departure hour. Only one month of data is used, so these patterns may not hold in other months.

## How to run

1. Install Python 3.12.
2. Create and activate a virtual environment, then run pip install -r requirements.txt.
3. Download the BTS data into the data folder as described above.
4. Open notebooks/01_explore.ipynb and run all cells.

## Limitations and next steps

* One month of data only; adding more months would give a stronger test.
* No weather or aircraft features.
* Planned: gradient boosting, logistic regression implemented from scratch in NumPy, and a deployed prediction API.

Author: Maliha Rahman