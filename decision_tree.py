# header = ["color", "diameter", "label"]
# data = [
#     ['Green', 3, 'Apple'],
#     ['Red', 4, 'Apple'],
#     ['Purple', 1, 'Grape'],
#     ['Purple', 1, 'Grape'],
#     ['Yellow', 3, 'Lemon'],
#     ['Yellow', 3, 'Banana'],
#     ['Yellow', 5, 'Banana'],
# ]

# from data set Extract possible values in a given column
def unique_val(data, col):
    return set(row[col] for row in data) 

# count number of items of each label
def label_count(data):
    label_cnt={}
    for row in data:
        if row[-1] not in label_cnt.keys():
            label_cnt[row[-1]] = 1
        else:
            label_cnt[row[-1]] += 1
    return label_cnt

# test if the value is numeric
def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


'''Question is defined by giving a column and a feature value
ex: Question(0,"value") => check if [column 0] is "value"
    Question(1, value) => check if [column 1] >= value 
How to use: define a question q = Question(c,value), and a column from data set data[c]
            q.math(data[c]) => if data[c] == value or data[c] >= value, return True'''
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    # Compare the feature value in an example to the
    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val.lower() == self.value.lower()

    # # helper method to print the question 
    # def __repr__(self):
    #     condition = "=="
    #     if is_numeric(self.value):
    #         condition = ">="
    #     return "Is %s %s %s?" % (
    #         header[self.column], condition, str(self.value))


'''Partition dataset
ask [question] on given dataset, partition the dataset into 
True and False subsets.'''
def  partition(data, question):
    true_rows, false_rows = [], []
    for row in data:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


"""Calculate the Gini Impurity for dataset
Gini is the uncertainty of a group of items with respect of their labels.
meaning, there are serval items and their labels. Pick each item at a time 
and randomly pick a label, the probability that the label doesn't match with 
the item is Gini impurity""" 
def gini(data):
    counts = label_count(data)
    if len(counts)==0: return 0     # empty dataset has gini = 0
    impurity = 1
    for lbl in counts.keys():
        prob_of_lbl = counts[lbl] / float(len(data))
        impurity -= prob_of_lbl**2
    return impurity


"""Information Gain describe how good is a question
info_gain = gini(beforeQ) - gini(afterQ)
note: gini(afterQ) is weighted gini impurity
      gini(beforeQ) is [current_uncertainty] as parameter of the function"""
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


"""Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
def find_best_question(data):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(data)
    n_features = len(data[0]) - 1  # number of feature columns

    # iterate each feature column
    for col in range(n_features):
        values = unique_val(data,col)
        # iterate each possible value in the column
        for val in values:
            q = Question(col, val)
            true, false = partition(data, q)
            gain = info_gain(true, false, current_uncertainty)
            if gain > best_gain:
                best_gain, best_question = gain, q
    return best_gain, best_question


"""Leaf is the final data that can't be further partitioned.
Which contains the label of item and the number of the label {label: count}"""
class Leaf:
    def __init__(self, rows):
        self.predictions = label_count(rows)


"""A Decision Node represent each question asked.
Which contains a reference of question, and two child nodes that are
splitted by asking the question"""
class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch  

class DecisionTree:
    def __init__(self):
        self.root = None

    def fit(self, train_data):
        self.root = self.build_tree(train_data)

    def predict(self,testData):
        return self.get_predict(self.classify(testData, self.root))

    """Builds the decision tree, find question to divide dataset and 
    recursively do same thing on children sets
    Base case: info_gain==0, no more partition avaliable, make current data as leaf
    Recursive case: call function on both true and false children"""
    def build_tree(self, data):
        gain, question = find_best_question(data)
        # Base case: no further info gain
        if gain == 0:
            return Leaf(data)

        # Recursive case: divided data into two subsets and keep partition on subsets
        true_rows, false_rows = partition(data, question)
        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)

        # Return a Question node.
        return Decision_Node(question, true_branch, false_branch)

    """Let testData go through decision tree, and get predicted label"""
    def classify(self, testData, node):
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.predictions

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.question.match(testData):
            return self.classify(testData, node.true_branch)
        else:
            return self.classify(testData, node.false_branch)

    '''Format classified label into Probabilities'''
    def get_predict(self,counts):
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs



'''Test'''
# myTree = build_tree(data)
# test = ['red',3]
# print(get_predict(classify(test,myTree)))
# mytree = DecisionTree()
# mytree.fit(data)
# p = mytree.predict(['yellow',3])
# print(p)
