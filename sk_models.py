from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import expon, reciprocal
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.exceptions import ConvergenceWarning
import warnings

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

train = pd.read_csv("data/train.csv")
train = train[["Review", "overall"]]
train = train.dropna(subset=["Review"])

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def train_model(model, hparams, representation):
    if representation == "cbow" or representation == "skipgram":
        X = np.load(f"./representations/X_{representation}.npz")["features"]
    else:
        X = load_npz(f"./representations/X_{representation}.npz")
    y = train["overall"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training {model} on {representation} representation")

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=hparams,
        n_iter=10,
        cv=5,
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )

    # Fit RandomizedSearchCV on the training set
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        random_search.fit(X_train, y_train)

    print(f"Training {model} on {representation} representation completed")

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    y_pred = best_model.predict(X_test)
    accuracy = best_model.score(X_test, y_test)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return best_model, best_params, accuracy, class_report, conf_matrix


def svc_model():
    hparams_SVC = {
        "C": reciprocal(0.1, 100),  # Regularization parameter
        "gamma": expon(scale=1.0),  # Kernel coefficient
        "kernel": ["rbf"],  # Assuming we're only interested in the RBF kernel here
        "max_iter": [1000],
    }

    representations = [
        "tf_idf",
        "OneGram",
        "TwoGram",
        "cbow",
        "skipgram",
    ]

    svc_dict = {}

    for representation in tqdm(representations):
        best_model, best_params, accuracy, class_report, conf_matrix = train_model(
            SVC(), hparams_SVC, representation
        )
        svc_dict.update(
            {
                f"{representation}_model": best_model,
                f"{representation}_params": best_params,
                f"{representation}_accuracy": accuracy,
                f"{representation}_class_report": class_report,
                f"{representation}_conf_matrix": conf_matrix,
            }
        )

    with open("svc_dict.pkl", "wb") as f:
        pickle.dump(svc_dict, f)


def rfc_model():
    rfc = RandomForestClassifier()

    hparams_RF = {
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=20)],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [int(x) for x in np.linspace(5, 120, num=12)] + [None],
        "min_samples_split": [2, 5, 10, 15],
        "min_samples_leaf": [1, 2, 4, 6],
        "bootstrap": [True],
        "max_samples": [
            0.5,
            0.75,
            0.25,
        ],  # Consider smaller fractions for large datasets
        "class_weight": ["balanced_subsample"],  # Important for imbalance
        "criterion": ["gini", "entropy"],
    }

    representations = [
        "tf_idf",
        "OneGram",
        "TwoGram",
        "cbow",
        "skipgram",
    ]

    RF_dict = {}

    for representation in tqdm(representations):
        best_model, best_params, y_pred, class_report, conf_matrix = train_model(
            RandomForestClassifier(), hparams_RF, representation
        )
        RF_dict.update(
            {
                f"{representation}_model": best_model,
                f"{representation}_params": best_params,
                f"{representation}_y_pred": y_pred,
                f"{representation}_class_report": class_report,
                f"{representation}_conf_matrix": conf_matrix,
            }
        )

    with open("rfc_dict.pkl", "wb") as f:
        pickle.dump(RF_dict, f)


def mlp_model():
    def create_train_test(representation):
        if representation == "cbow" or representation == "skipgram":
            X = np.load(f"./representations/X_{representation}.npz")["features"]
        else:
            X = load_npz(f"./representations/X_{representation}.npz")
        y = train["overall"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    representations = [
        "tf_idf",
        "OneGram",
        "TwoGram",
        "cbow",
        "skipgram",
    ]

    def sparse_generator(X, y, batch_size=32, epochs=10):
        """
        A generator for batches of sparse data.

        Parameters:
        - X: Sparse feature matrix (scipy.sparse matrix).
        - y: Labels (numpy array).
        - batch_size: Size of batches to generate.
        """
        n_samples = X.shape[0]

        # Shuffle data indices for each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for _ in range(epochs):
            indices = np.arange(n_samples)

            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]

                # Convert the sparse batch to a dense format as TensorFlow expects dense arrays
                X_batch = X[
                    batch_indices
                ].toarray()  # Convert sparse matrix part to dense
                y_batch = y[batch_indices]

                yield X_batch, y_batch

    def sparse_predict_generator(X, batch_size=32):
        n_samples = X.shape[0]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            X_batch = X[start:end].toarray()

            yield X_batch

    mlp_dict = {}

    for representation in tqdm(representations):
        X_train, X_test, y_train, y_test = create_train_test(representation)
        input_dim = X_train.shape[1]

        # Define the MLP model
        model = Sequential()
        model.add(
            Dense(512, input_dim=input_dim, activation="relu")
        )  # input_dim is the size of your input features
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(5, activation="softmax"))  # Output layer

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        y_train_adjusted = y_train - 1
        y_test_adjusted = y_test - 1

        y_train_one_hot = to_categorical(y_train_adjusted)
        y_test_one_hot = to_categorical(y_test_adjusted)

        batch_size = 32
        step_size = X_train.shape[0] // batch_size
        step_size_test = X_test.shape[0] // batch_size
        epochs = 10

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train_adjusted), y=y_train_adjusted
        )
        class_weights_dict = dict(enumerate(class_weights))

        history = model.fit(
            sparse_generator(X_train, y_train_one_hot),
            steps_per_epoch=step_size,
            epochs=epochs,
            class_weight=class_weights_dict,
            validation_data=sparse_generator(X_test, y_test_one_hot),
            validation_steps=step_size_test,
            verbose=1,
        )

        # y_pred = model.predict(sparse_predict_generator(X_test))
        # y_pred = np.argmax(y_pred, axis=1) + 1

        # accuracy = model.evaluate(X_test, y_test_one_hot)[1]
        # class_report = classification_report(y_test, y_pred)
        # conf_matrix = confusion_matrix(y_test, y_pred)

        mlp_dict.update(
            {
                f"{representation}_model": model,
                # f"{representation}_accuracy": accuracy,
                # f"{representation}_class_report": class_report,
                # f"{representation}_conf_matrix": conf_matrix,
            }
        )

    with open("mlp_dict.pkl", "wb") as f:
        pickle.dump(mlp_dict, f)


def main():
    svc_model()
    rfc_model()
    mlp_model()


main()
