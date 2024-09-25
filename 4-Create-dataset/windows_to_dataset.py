from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

@dataclass
class WindowsToDataset:
    """
    Turn windows of behavior events into a dataset which is ready for ML training.
    Perform preprocessing of the data: one-hot encoding for categorical features and normalization to a standard
      normal distribution for numerical features.
    Split the dataset into a "train" and "validation" part.

    Attributes:
        windows_benign:     The file path to the NPZ file with all benign windows.
        windows_malicious:  The file path to the NPZ file with all malicious windows.
        target_dataset:     File path under which the resulting dataset gets stored.
        feature_schema_csv: A CSV file with a list of security features plus their default values.
    """

    windows_benign: str
    windows_malicious: str
    target_dataset: str
    security_features_csv: str

    def _load_windows(self, file_path: str) -> np.ndarray:
        """Load a window dataset.
        
        :param file_path: The dataset file name plus path.
        :return: The windows as a numpy array of shape (num_windows, window_size, num_features).
        """
        load_dataset = np.load(file_path, allow_pickle=True)
        windows = load_dataset["windows"]
        return windows
    
    def _create_labels(self, number_of_windows: int, is_malicious: bool) -> np.ndarray:
        """Create the labels for all benign or malicious samples.
        
        :return: The one-hot encoded labels, e.g. [[1,0],[1,0],[1,0]] for 3 benign labels.
        """
        if is_malicious:
            return np.tile(np.array([0, 1]), (number_of_windows,1))
        else:
            return np.tile(np.array([1, 0]), (number_of_windows,1))
    
    def _fetch_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch the dataset, which consists of behavior event windows plus labels.
        
        :return: A tuple consisting of all windows plus all labels.
        """
        # fetch all windows of behavior events
        benign_windows = self._load_windows(self.windows_benign)
        malicious_windows = self._load_windows(self.windows_malicious)
        x = np.vstack([benign_windows, malicious_windows])
        # create labels
        benign_labels = self._create_labels(len(benign_windows), False)
        malicious_labels = self._create_labels(len(malicious_windows), True)
        y = np.vstack([benign_labels, malicious_labels])
        return x, y

    def _fetch_security_features(self) -> pd.DataFrame:
        """
        Fetch the list of security feature names and their default values, as string

        :return A list of security feature names plus their respective default values as strings
        """
        return pd.read_csv(self.security_features_csv)

    def _get_all_numerical_security_features(self, security_features: pd.DataFrame) -> List[str]:
        """Create a list of all security features which are of numerical nature."""
        return list(security_features[security_features["default_value"].isin(["0", "0.0"])]["security_feature_name"])
    
    def _get_all_boolean_security_features(self, security_features: pd.DataFrame) -> List[str]:
        """Create a list of all security features which are of boolean nature."""
        return list(security_features[security_features["default_value"]=="False"]["security_feature_name"])
    
    def _get_all_categorical_security_features(self, security_features: pd.DataFrame) -> List[str]:
        """Create a list of all security features which are of categorical nature."""
        return list(security_features[security_features["default_value"]=="NOT_PRESENT"]["security_feature_name"])

    def preprocess(self, x) -> Tuple[np.ndarray, List[str]]:
        """Preprocess the dataset.
        
        Perform one-hot encoding for categorical values, plus normalization for numerical values.
        Note that one-hot encoding enlarges the number of features. Each categorical feature gets
          mapped to an array equal in size to the number of possible categories.
        
        :return: A tuple, consisting of the normalized window dataset, plus the feature names after preprocessing.
        """
        # create scikit-learn column transformer for feature preprocessing (one-hot encoding plus normalization of numerical values)
        all_security_features = self._fetch_security_features()
        numerical_features = self._get_all_numerical_security_features(all_security_features)
        boolean_features = self._get_all_boolean_security_features(all_security_features)
        categorical_features = self._get_all_categorical_security_features(all_security_features)
        column_transformer = ColumnTransformer(
            [
                ("numerical", StandardScaler(), numerical_features),
                ("boolean", "passthrough", boolean_features),
                ("categorical", OneHotEncoder(), categorical_features),
            ]
        )
        # stack all windows to a 2-d array
        x_2d = x.reshape((-1, x.shape[2]))
        # apply the column transformer
        security_feature_names = list(all_security_features["security_feature_name"])
        x_df = pd.DataFrame(x_2d, columns=security_feature_names)
        x_preprocessed_2d = column_transformer.fit_transform(x_df)
        features_names_after_preprocessing = column_transformer.get_feature_names_out()
        # convert back to 3d array
        x_preprocessed_3d = x_preprocessed_2d.reshape(x.shape[0], x.shape[1], -1)
        return x_preprocessed_3d, features_names_after_preprocessing

    def _create_dataset_split(self, x, y):
        """Split all data points into a train and a validation dataset.
        
        :return: A tuple, consisting of x_train, x_val, y_train, y_val.
        """
        # TODO: the dataset should be split by graph ID or the endpoint from which
        # the behavior logs originate from!
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.4)
        return x_train, x_val, y_train, y_val


    def call(self) -> None:
        """Turn a list of behavior windows into a dataset which is ready for ML training."""
        print(f"Start dataset creation, using {self.windows_benign} and {self.windows_malicious}.")
        x, y = self._fetch_dataset()
        print(f"Performing preprocessing of dataset for {len(x)} windows ...")
        x_preprocessed, features_names_after_preprocessing = self.preprocess(x)
        x_train, x_val, y_train, y_val = self._create_dataset_split(x_preprocessed, y)
        print(f"Storing final dataset under {self.target_dataset}.")
        np.savez_compressed(
                self.target_dataset,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                features_names_after_preprocessing=features_names_after_preprocessing
        )
