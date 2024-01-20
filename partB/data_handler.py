import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        try:
            data, meta = arff.loadarff(self.file_path)
            df = pd.DataFrame(data)
            print(f"Columns in the data: {df.columns}")  # Debug print statement
            if 'Class' in df.columns:
                df['Class'] = df['Class'].str.decode('utf-8')  # Convert bytes to string if needed
            else:
                raise ValueError("The expected 'Class' column was not found in the data.")
            return df
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return None

    def preprocess(self):
        df = self.load_data()
        if df is not None:
            X = df.drop('Class', axis=1)
            y = df['Class']

            # Encode labels numerically
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_encoded, test_size=0.2,
                                                                                    random_state=42)
        else:
            print("Data preprocessing failed due to load error.")
