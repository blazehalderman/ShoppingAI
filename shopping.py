import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    months = {month: index-1 for index, month in enumerate(calendar.month_abbr) if index}
    months['June'] = months.pop('Jun')

    labels = []
    evidence = []
    count = 0
    #opens file
    with open(filename) as f:
        #gets the remaining lines in the file to be used for evidence
        lines = f.readlines()[1: ]
    # close file
    f.close()
    # create a list for each line of evidence
    for line in lines:
        evidence.append(line.strip('\n').split(','))
        labels.append(1 if evidence[count][-1] == 'TRUE' else 0)
        evidence[count].pop()
        evidence[count][0] = int(evidence[count][0]) #Administrative
        evidence[count][1] = float(evidence[count][1]) #Administrative_Duration
        evidence[count][2] = int(evidence[count][2]) #Informational
        evidence[count][3] = float(evidence[count][3]) #Informational_Duration
        evidence[count][4] = int(evidence[count][4]) #ProductRelated
        evidence[count][5] = float(evidence[count][5]) #ProductRelated_Duration
        evidence[count][6] = float(evidence[count][6]) #BounceRates
        evidence[count][7] = float(evidence[count][7]) #ExitRates
        evidence[count][8] = float(evidence[count][8]) #PageValues
        evidence[count][9] = float(evidence[count][9]) #SpecialDay
        evidence[count][10] = months[evidence[count][10]] #Months
        evidence[count][11] = int(evidence[count][11]) #OperatingSystems
        evidence[count][12] = int(evidence[count][12]) #Browser
        evidence[count][13] = int(evidence[count][13]) #Region
        evidence[count][14] = int(evidence[count][14]) #Traffic Type
        if evidence[count][15] == 'Returning_Visitor':
            evidence[count][15] = 1
        else:
            evidence[count][15] = 0
        if evidence[count][16] == 'TRUE':
            evidence[count][16] = 1
        else:
            evidence[count][16] = 0
        count += 1

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    return KNeighborsClassifier(n_neighbors=1).fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sens_y, spec_y = float(0), float(0)
    pos, neg = float(0), float(0)
    for label, prediction in zip(labels, predictions):
        if label == 1:
            pos += 1
            if label == prediction:
                sens_y += 1

        if label == 0:
            neg += 1
            if label == prediction:
                spec_y += 1
    sens_y = sens_y / pos
    spec_y = spec_y / neg

    return sens_y, spec_y


if __name__ == "__main__":
    main()
