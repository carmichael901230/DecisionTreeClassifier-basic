# DecisionTreeClassifier-basic
A silly decision tree classifer, similar to sklearn.DecisionTreeClassifier()

## How to Use
**Import Decision Tree Class**
```
import /path_of_decision_tree.py/decision_tree as dt
```
**Define a Decision Tree Instance**
```
myTree = dt.DecisionTree()
```

**Training Data**
- Must be ```list``` type.
- The last column of each row must be the label.
- Don't seperate feature columns and label column

**Train Classifier**
```
dt.fit(training_data)
```

**Test Data**
- Must be ```list``` type
- Must have the same columns and same column sequence as training data
- Except test data doen't contain label column

**Predict Test Data**

```
dt.predict(test_data)
```

```predict()```Function returns a dictionary of predicted labels with their confidence
