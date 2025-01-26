import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

def main():
    # we see that how increasing the number of training samples affects the accuracy of the decision tree
    # we know that the decision tree divides the feature space into regions that are hyper-rectangles,
    # lower is the value of N_train and less precise the decision tree will be, so we will se the rectangles
    # in the plot of the predictions.
    N_train = [100, 1000, 10000]
    for n in N_train:
        x_train = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)])
        y_train = np.sign(-2 * np.sign(x_train[:, 0]) * np.abs(x_train[:, 0])**(2/3) + 4 * x_train[:, 1]**2)
        
        colors = ['red' if z == -1 else 'blue' for z in y_train]
        plt.figure(figsize=(10,10), num='Training Data')
        plt.scatter([x_1 for x_1, _ in x_train], [x_2 for _, x_2 in x_train], c=colors)
        plt.show()

        # Train a C4.5 decision tree
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(x_train, y_train)

        # Plot the decision tree
        # plt.figure(figsize=(20,10))
        # plot_tree(clf, filled=True, feature_names=['x1', 'x2'], class_names=['-1', '1'])
        # plt.show()

        N_test = 20000
        x_test = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(N_test)])
        y_test = clf.predict(x_test)
        print(f"Accuracy with N_train = {n}: {accuracy_score(y_test, np.sign(-2 * np.sign(x_test[:, 0]) * np.abs(x_test[:, 0])**(2/3) + 4 * x_test[:, 1]**2))}")

        colors = ['red' if z == -1 else 'blue' for z in y_test]
        plt.figure(figsize=(10,10), num='Test Data')
        plt.scatter([x_1 for x_1, _ in x_test], [x_2 for _, x_2 in x_test], c=colors)
        plt.show()

if __name__ == "__main__":
    main()