import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import StandardScaler

# Función para entrenar y evaluar un clasificador con validación específica
def evaluate_classifier(classifier, X, y, validation_method):
    if validation_method == "hold_out":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return acc, conf_matrix

    elif validation_method == "k_fold":
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        acc_scores = []
        all_conf_matrices = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc_scores.append(accuracy_score(y_test, y_pred))
            all_conf_matrices.append(confusion_matrix(y_test, y_pred))
        avg_acc = np.mean(acc_scores)
        return avg_acc, sum(all_conf_matrices)

    elif validation_method == "leave_one_out":
        loo = LeaveOneOut()
        acc_scores = []
        all_conf_matrices = []
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc_scores.append(accuracy_score(y_test, y_pred))
            all_conf_matrices.append(confusion_matrix(y_test, y_pred, labels=np.unique(y)))
        avg_acc = np.mean(acc_scores)
        return avg_acc, sum(all_conf_matrices)

# Función para probar con los datasets y clasificadores
def main():
    # Cargar datasets
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    # Normalizar los datos
    scaler = StandardScaler()
    X_iris_scaled = scaler.fit_transform(X_iris)

    X_additional1, y_additional1 = make_classification(
        n_samples=150,
        n_features=4,
        n_informative=2,
        n_redundant=1,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    X_additional1_scaled = scaler.fit_transform(X_additional1)

    X_additional2, y_additional2 = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=99
    )
    X_additional2_scaled = scaler.fit_transform(X_additional2)

    datasets = [
        ("Iris Plant", X_iris_scaled, y_iris),
        ("Dataset 1", X_additional1_scaled, y_additional1),
        ("Dataset 2", X_additional2_scaled, y_additional2)
    ]

    # Clasificadores
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42)  # Más iteraciones
    rbf = SVC(kernel="rbf", random_state=42)

    classifiers = [
        ("Perceptrón Multicapa", mlp),
        ("Red Neuronal RBF", rbf),
    ]

    validation_methods = ["hold_out", "k_fold", "leave_one_out"]  # Incluye Leave-One-Out

    for dataset_name, X, y in datasets:
        print(f"\n=== Evaluando en el dataset: {dataset_name} ===")
        for classifier_name, classifier in classifiers:
            print(f"\nClasificador: {classifier_name}")
            for validation_method in validation_methods:
                acc, conf_matrix = evaluate_classifier(classifier, X, y, validation_method)
                print(f"\nMétodo de validación: {validation_method}")
                print(f"Accuracy: {acc:.4f}")
                print(f"Matriz de Confusión:\n{conf_matrix}")

if __name__ == "__main__":
    main()
