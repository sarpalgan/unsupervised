import pandas as pd
import numpy as np
import math
import warnings
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
#from IPython.display import display_html 

from data_preprocess import DataPreprocessing

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)


class Prediction:
    """
    Class that clusters a given dataset and predicts the mean and standard deviation of the 'time' column. 
    The class contains three methods: predict, test_preds, and predict_new.

    Attributes:
        df_to_predict (pandas DataFrame): The dataframe to predict on.
        data_clusters_clean (pandas DataFrame): The cleaned dataframe for clustering.
        cols_list (list): The list of columns to use for clustering.
        data_onehot_clusters (pandas DataFrame): The one-hot encoded dataframe for clustering.
        ft_wt (list): The list of columns and weights to use for one-hot encoding.
        kmeans (sklearn KMeans): The KMeans clustering model for the first cluster.
        kmeans2 (sklearn KMeans): The KMeans clustering model for the second cluster.
        cluster2_list (list): The list of clusters for the second cluster model.
        pca (sklearn PCA): The PCA model for feature reduction.
        x_test (pandas DataFrame): The test dataframe for checking the accuracy of predictions.
        x_train (pandas DataFrame): The training dataframe for clustering.
        preds (list): A list of tuples containing the predicted mean and standard deviation of the 'time' column.

    Methods:
        predict(data_clustered):
            Clusters a given dataset and predicts the mean and standard deviation of the 'time' column. 

            Args:
                data_clustered (pandas DataFrame): The input dataset that has already been clustered.

            Returns:
                A tuple containing the predicted mean and standard deviation of the 'time' column.

        test_preds():
            Test the accuracy of predictions and calculate the mean absolute error (MAE) and mean squared error (MSE)
            for a set of test data.

            Args:
                self: An instance of the Prediction class with attributes x_train, x_test, df_to_predict, and preds.

            Returns:
                accuracy (float): The percentage of true predictions within the test data.
                mae (float): The mean absolute error between the actual and predicted values.
                mse (float): The mean squared error between the actual and predicted values.

        predict_new():
            Method to predict new values using the trained model.

            Args:
                self: The object itself.

            Returns:
                None
    """
    
    def __init__(self):
        self.df_to_predict = None
        self.data_clusters_clean = None
        self.cols_list = None
        self.data_onehot_clusters = None
        self.ft_wt = None
        self.kmeans = None
        self.kmeans2 = None
        self.cluster2_list = None
        self.pca = None
        self.x_test = None
        self.x_train = None
        self.preds = []

    def predict(self, data_clustered):
        """
        Clusters a given dataset and predicts the mean and standard deviation of the 'time' column. 

        Args:
        data_clustered (pandas DataFrame): The input dataset that has already been clustered.

        Returns:
        A tuple containing the predicted mean and standard deviation of the 'time' column.
        """

        cols_list = self.cols_list
        df = self.df_to_predict
        df_empty = self.data_onehot_clusters[0:0] 
        ft_wt = self.ft_wt
        model_1 = self.kmeans
        model_2 = self.kmeans2
        cluster2_list = self.cluster2_list 
        pca = self.pca
        time_col = 'time'
    
        #####################################   one hot + pca    

        cols_list = [x for x in cols_list if x in df.columns]
        
        df = df[cols_list]

        for ft in ft_wt:
            if ft[0] in cols_list:
                df = DataPreprocessing.one_hot(df, ft)
    
        test_data_onehot = pd.concat([df_empty, df])
        test_data_onehot = test_data_onehot.fillna(0)
    
        test_data_onehot = pca.transform(test_data_onehot)
    
        #####################################   kmeans-predict
    
        kmeans_clusters1 = model_1.predict(test_data_onehot)
        kmeans_clusters2 = model_2.predict(test_data_onehot) 
    
        print(f'\nCluster1: {kmeans_clusters1}   Cluster2: {kmeans_clusters2}   {kmeans_clusters2 in cluster2_list}')
    
        #####################################   prediction
    
        data_clustered[time_col] = data_clustered[time_col].astype(int)
    
        if kmeans_clusters2 in cluster2_list:
        
            df2 = data_clustered[data_clustered['Cluster2'] == kmeans_clusters2[0]].copy()
        
            if len(df2) <= 2:
                df2 = data_clustered[data_clustered['Cluster1'] == kmeans_clusters1[0]].copy()
            
            t = df2[time_col]
            mean, std = t.mean(), t.std()

        else:
            df2 = data_clustered[data_clustered['Cluster1'] == kmeans_clusters1[0]].copy()
            t = df2[time_col]
            mean, std = t.mean(), t.std()
    
        if np.isnan(std):
            print('\t\t\t\tNaN std converted to 1 ...')
            std = 2

        print(f'prediction: {round(mean)} +- {round(std)}')        
            
        return round(mean), round(std)

    def test_preds(self):
        """Test the accuracy of predictions and calculate the mean absolute error (MAE) and mean squared error (MSE) 
        for a set of test data.
    
        Args:
        - self: An instance of the Prediction class with attributes x_train, x_test, df_to_predict, and preds.
    
        Returns:
        - accuracy: The percentage of true predictions within the test data.
        - mae: The mean absolute error between the actual and predicted values.
        - mse: The mean squared error between the actual and predicted values.
        """
        true_preds = 0
        mae = 0
        mse = 0

        for index in self.x_test.index.to_list():
            self.df_to_predict = self.x_test.loc[[index]]
            y_true = self.df_to_predict['time'].item()
            data_clustered = self.x_train
            prediction, margin = Prediction.predict(self, data_clustered = data_clustered)
            self.preds.append([prediction, margin])
            print(index)

            if (prediction + margin >= y_true) and (prediction - margin <= y_true):
                true_preds += 1

            mae += abs(y_true - prediction)
            mse += (y_true - prediction)**2

        accuracy = 100*(true_preds/len(self.x_test))
        mae = mae / len(self.x_test)
        mse = mse / len(self.x_test)
        
        print(f"\n\nAccuracy: {accuracy:.1f}%")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")

    def predict_new(self):
        """
        Method to predict new values using the trained model.

        Args:
            self: The object itself.
    
        Returns:
            None
        """
        data_clustered = self.data_clusters_clean
        prediction, margin = Prediction.predict(self, data_clustered = data_clustered)
        self.preds.append([prediction, margin])
       
