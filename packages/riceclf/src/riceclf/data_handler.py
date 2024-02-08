import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataHandler:
    """
    Handles the loading and preprocessing of rice dataset.

    Attributes:
        file_path (str): The file path to the dataset.
        X_train (DataFrame): The training feature dataset.
        X_test (DataFrame): The testing feature dataset.
        y_train (Series): The training target values.
        y_test (Series): The testing target values.
    """

    def __init__(self, file_path):
        """
        Initializes the DataHandler with the dataset file path.

        Args:
            file_path (str): The file path to the dataset.
        """
        self.file_path = file_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        Loads the data from an ARFF file into a pandas DataFrame.

        Returns:
            DataFrame: The loaded data if successful, None otherwise.
        """
        data, meta = arff.loadarff(self.file_path)
        df = pd.DataFrame(data)
        print(f"Columns in the data: {df.columns}")  # Debug print statement
        if "Class" in df.columns:
            df["Class"] = df["Class"].str.decode(
                "utf-8"
            )  # Convert bytes to string if needed
        else:
            raise ValueError("The expected 'Class' column was not found in the data.")
        return df

    def preprocess(self):
        """
        Preprocesses the loaded data by encoding categorical variables and
        splitting the data into training and testing sets.
        """
        df = self.load_data()
        X = df.drop("Class", axis=1)
        y = df["Class"]

        # Encode labels numerically
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
