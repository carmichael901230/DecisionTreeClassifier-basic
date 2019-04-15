import decision_tree as dt

# header is not necessary
header = ['color', 'size', 'shape', 'label']
train_data = [
    ['Green', 3, 'round', 'Apple'],
    ['Red', 3, 'round', 'Apple'],
    ['Purple', 1, 'round', 'Grape'],
    ['Purple', 1, 'round', 'Grape'],
    ['Yellow', 3, 'round', 'Lemon'],
    ['Yellow', 3, 'long', 'Banana'],
]
# define a decision tree
myTree = dt.DecisionTree()
# train decision tree with training dta
myTree.fit(train_data)

# test data, should has 'grape' as its label
test_data = ['purple',1, 'round']
result = myTree.predict(test_data)
# output predicted result
print(result)