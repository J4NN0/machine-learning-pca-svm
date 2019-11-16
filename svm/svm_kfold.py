from sklearn import datasets, svm
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt


def main():
    # Import iris data
    iris = datasets.load_iris()
    x = iris.data[:, :2]  # take the first two features.
    y = iris.target  # label

    # Splitting data into train, validation and test set
    # 70% training set, 30% test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

    c = [10 ** (-3), 10 ** (-2), 10 ** (-1), 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    g = [10 ** (-9), 10 ** (-7), 10 ** (-5), 10 ** (-3)]
    best_values = [0.0, 0.0, 0.0]  # respectively best success rate, best C and best gamma

    k_fold = KFold(n_splits=5)

    print('Performing 5-Fold validation')
    for i in c:
        plt.figure(figsize=(40, 20))
        for j in g:
            for id_train, id_test in k_fold.split(x_train):
                svc = svm.SVC(kernel='rbf', C=i, gamma=j)
                score = svc.fit(x_train[id_train], y_train[id_train]).score(x_train[id_test], y_train[id_test])
                print('With C=' + str(i) + ' and gamma=' + str(j) + ' avg=' + str(score))
                if score > best_values[0]:
                    best_values = score, i, j

    print('Best accuracy=' + str(best_values[0]) + ' with C=' + str(best_values[1]) + ' and gamma=' + str(best_values[2]))

    # With the best C ang gamma evaluating k-fold on test set
    print('Evaluating test set')
    svc = svm.SVC(kernel='rbf', C=best_values[1], gamma=best_values[2])
    svc.fit(x_train, y_train)
    print(svc.score(x_test, y_test))


if __name__ == "__main__":
    main()
