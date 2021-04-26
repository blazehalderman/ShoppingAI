# ShoppingAI
Python Shopping AI using predictability

An AI to predict whether online shopping customers will complete a purchase.
```
$ python shopping.py shopping.csv
Correct: 4088
Incorrect: 844
True Positive Rate: 41.02%
True Negative Rate: 90.55%
```
Uses csv data in the format(sorted by category) of:
```
Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend, Revenue
```
From the csv data we can predict whether someone will make a purchase or not where Revenue is binary (0 -> no purchase or 1 -> purchase).
