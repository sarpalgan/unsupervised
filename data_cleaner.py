import numpy as np
import pandas as pd
import math
import warnings
import pickle
import configparser
from typing import List, Tuple

#from IPython.display import display_html 
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)

class DataCleaning:
    """
    Class that performs data cleaning on a DataFrame.

    Attributes
    ----------
    test_size : float
        The size of the test data set.
    zs0 : float
        The threshold value for determining outliers.
    z_score : float
        The z-score value for determining outliers.
    df_coef : NoneType or pandas.DataFrame
        The coefficients of the columns.
    data_clusters : NoneType or pandas.DataFrame
        The clusters from the model.
    data_clusters_clean : NoneType or pandas.DataFrame
        The cleaned clusters from the model.
    drop_list : list
        A list of items to be dropped from the DataFrame.
    cluster2_list : NoneType or list
        A list of clusters.
    data : NoneType or pandas.DataFrame
        The data that has been cleaned.
    x_train : NoneType or pandas.DataFrame
        The training data.
    x_test : NoneType or pandas.DataFrame
        The test data.

    Methods
    -------
    common_member(a: list, b: list) -> list:
        Returns a list of common elements between two lists.
    cleaner(df: pandas.DataFrame, zs: float) -> Tuple[pandas.DataFrame, List[int]]:
        This function removes outliers from the input DataFrame, based on the z-score method.
    drop_displayer(df, indices):
        Updates a dataframe with a 'drops' column to indicate the indices that were dropped.
    clean_noise(df, zs, cluster, x, display_results = display_results, final = False):
        Perform data cleaning to remove noise from a dataframe.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the DataCleaning object.

        Reads settings.ini to get values for the threshold values and test size.
        """
        config = configparser.ConfigParser()
        config.read('settings.ini')

        self.test_size: float = config.getfloat('DATA', 'test_size')
        self.zs0: float = config.getfloat('DATA', 'zs0')
        self.z_score: float = config.getfloat('DATA', 'z_score')
        self.max_clean_perc: float = config.getfloat('DATA', 'max_clean_perc')
        self.displayer = config.get('DATA', 'displayer')

        self.df_coef = None
        self.data_clusters = None
        self.data_clusters_clean = None
        self.drop_list = []
        self.cluster2_list = None
        self.data = None
        self.x_train = None
        self.x_test = None
        

    def common_member(a: list, b: list) -> list:
        """
        Returns a list of common elements between two lists.

        Parameters:
        a (list): A list of elements.
        b (list): A list of elements.

        Returns:
        list: A list of elements common to both input lists. If no common element is found, an empty list is returned.
        """
        a_set = set(a)
        b_set = set(b)

        if (a_set & b_set):
            return list(a_set & b_set)
        else:
            return []

    def cleaner(df: pd.DataFrame, zs: float) -> Tuple[pd.DataFrame, List[int]]:
        """
        This function removes outliers from the input DataFrame, based on the z-score method.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame to clean
        zs : float
            The threshold to determine the outliers. Values outside the range (mean - zs*std, mean + zs*std) are considered outliers.

        Returns:
        --------
        A tuple containing:
        df_cleaned : pandas.DataFrame
            The input DataFrame after cleaning the outliers
        indices : list of int
            A list of the indices of the rows that were removed from the input DataFrame due to outliers.

        Examples:
        ---------
        >>> import pandas as pd
        >>> df = pd.DataFrame({'time': [10, 20, 30, 40, 50]})
        >>> df_cleaned, indices = cleaner(df, 2)
        >>> print(df_cleaned)
           time
        0    10
        1    20
        2    30
        3    40
        4    50
        >>> print(indices)
        []

        """
        indices = []
        if len(df) >= 3:
            t = df[['time']].copy()
            df_zscore_t = (t - t.mean()) / t.std()
            dfz_t = abs(df_zscore_t) > zs
        
            if len(dfz_t[dfz_t['time'].eq(True)].index.tolist()) != 0:
                #display(df2)
                #print(f'dropped indeces: {dfz_t[dfz_t['time'].eq(True)].index.tolist()}')
                indices += dfz_t[dfz_t['time'].eq(True)].index.tolist()
        
            if len(indices) != 0:
                df.drop(indices, axis = 0, inplace=True)
        return df, indices

    def drop_displayer(df, indices):
        """
        Updates a dataframe with a 'drops' column to indicate the indices that were dropped.

        Args:
        - df (pandas.DataFrame): A pandas dataframe to update.
        - indices (list): A list of indices that were dropped.

        Returns:
        - df (pandas.DataFrame): A pandas dataframe with the 'drops' column updated.
        """
        if len(indices) != 0:
            if indices == indices0:
                lst = '1st'
            elif indices == indices1:
                lst = '2nd'
            elif indices == indices2:
                lst = '3rd'
            else:
                raise ValueError(f'could not find indices list {indices}')
            for index in indices:
                df.at[index, 'drops'] = f'{lst} drop'
        return df  

    def clean_noise(df, zs, cluster, x, final = False):
        """
        Perform data cleaning to remove noise from a dataframe.

        Args:
            df (pandas.DataFrame): The dataframe to be cleaned.
            zs (float): A threshold value to determine outliers.
            cluster (str): A column name containing a cluster to be cleaned.
            x (float): A multiplier to determine the threshold value for extreme outliers.
            display_results (bool): A flag to display the results of the cleaning process.
            final (bool): A flag to indicate the final cleaning process.

        Returns:
            list: A list of indices that were dropped.

        Raises:
            ValueError: If indices list cannot be found.

        """
        # Display or not
        display_results = self.displayer

        # Get the list of column names from self.df_coef['columns_name'] and append 'time'
        cols = self.df_coef['columns_name'].to_list()
        cols.append('time')

        # Initialize the index_drop_list
        index_drop_list = []

        # Convert 'time' column to integer
        df['time'] = df['time'].astype(int)

        # Loop through unique values in the cluster column
        for item in df[cluster].unique():
            # Get a copy of the dataframe with the current cluster value
            df2 = df[df[cluster] == item].copy()
            
            # Clean the dataframe using the threshold values
            df2, indices0 = DataCleaning.cleaner(df2, self.zs0)
            df2, indices1 = DataCleaning.cleaner(df2, zs)
            df2, indices2 = DataCleaning.cleaner(df2, zs * x)
            
            # Append indices to the index_drop_list
            index_drop_list += indices0
            index_drop_list += indices1
            index_drop_list += indices2
            
            # If final flag is True, set 'drops' column to empty string
            if final:
                df['drops'] = ''                    

            # If there are indices to drop and final flag is True, display the results of the cleaning process
            if ((len(indices0) != 0) or (len(indices1) != 0) or (len(indices2) != 0)) and (final):
                df = DataCleaning.drop_displayer(df, indices0)
                df = DataCleaning.drop_displayer(df, indices1)
                df = DataCleaning.drop_displayer(df, indices2)
                
                # If the cluster is Cluster1, set 'common' column to True for common indices
                if cluster == 'Cluster1':
                    list1 = indices0 + indices1 + indices2
                    list1 = DataCleaning.common_member(list1, self.data_clusters[self.data_clusters['Cluster2'].isnull() == True].index.tolist())
                    df['common'] = ''
                    for index in list1:
                        df.at[index, 'common'] = 'True'

                    # If there are common indices, display the results of the cleaning process
                    if len(df[(df[cluster] == item) & (df['common'] == 'True')]) != 0:
                        print(f'\n\ncleaning in {cluster}')
                        print(df[df[cluster] == item][['Cluster1', 'Cluster2'] + cols + ['drops', 'common']]) # display
                        per = 100 * (len(indices0) + len(indices1) + len(indices2)) / len(df[(df[cluster] == item) & (df['common'] == 'True')])
                        print(f'total cleaning percentage = {int(per)}%')
                        print('\n\n---------------------------------------------------------------------------------------')

                # If the cluster is not Cluster1, display the results of the cleaning process    
                else:
                    print(f'\n\ncleaning in {cluster}')
                    print(df[df[cluster] == item][[cluster] + cols + ['drops']]) #display
                    per = 100 * (len(indices0) + len(indices1) + len(indices2)) / len(df[df[cluster] == item])
                    print(f'cleaning percentage = {int(per)}%')
                    print('\n\n---------------------------------------------------------------------------------------')
                                    
                #print(f'first cleaning: {indices0}   second cleaning: {indices1}   third cleaning: {indices2}')

        return index_drop_list

    def cleaning(self):
        """
        Clean data by removing noisy points and clusters with insufficient data.
        
        Args:
            display_results (bool, optional): If True, display intermediate and final results of the cleaning process.
                Defaults to False.
        """
        display_results = self.displayer
        z_score_initial = self.z_score     
        x = 0.45
        control = False
        while control == False:
            x += 0.05
            self.drop_list = []

            # Clean cluster 2
            clean = self.data_clusters[self.data_clusters['Cluster2'].isnull() == False].copy()
            self.drop_list += DataCleaning.clean_noise(clean, zs = self.z_score, cluster = 'Cluster2', x = x)

            # Clean cluster 1
            drop_list2 = []
            clean = self.data_clusters.copy()
            drop_list2 += DataCleaning.clean_noise(clean, zs = self.z_score, cluster = 'Cluster1', x = x)
            
            # Remove clusters with insufficient data
            self.drop_list += DataCleaning.common_member(drop_list2, self.data_clusters[self.data_clusters['Cluster2'].isnull() == True].index.tolist())
            for item in self.data_clusters['Cluster1'].unique():
                if len(self.data_clusters[self.data_clusters['Cluster1'] == item]) < 3:
                    #display(self.data_clusters[self.data_clusters['Cluster1'] == item])
                    self.drop_list += self.data_clusters[self.data_clusters['Cluster1'] == item].index.tolist()
                    #print(f'Cleaned Cluster1 = {item} due to insufficient data')
            
            # Remove noisy points and save cleaned data
            self.data_clusters_clean = self.data_clusters.copy()
            self.data_clusters_clean.drop(self.drop_list, inplace=True)
            self.cluster2_list = [x for x in sorted(set(self.data_clusters_clean['Cluster2'].unique())) if ~np.isnan(x)] # to be used in prediction
            percent = round(len(self.drop_list)/len(self.data), 2)
        
            # Check cleaning progress and update parameters
            if percent < self.max_clean_perc:
                print(f'\n\n\tcleaned {100*percent:.1f}% of the data with zs0 = {self.zs0}, zs1 = {self.z_score}, zs2 = {round(self.z_score * x, 3)}\n\n')
                
                # Display final results if requested
                if display_results:  
                    print('\n\n\tDisplaying results\n\n')
                    final = True
                
                    self.drop_list = []
                    clean = self.data_clusters[self.data_clusters['Cluster2'].isnull() == False].copy()
                    self.drop_list += DataCleaning.clean_noise(clean, zs = self.z_score, cluster = 'Cluster2', x = x, final = final)

                    
                    drop_list2 = []
                    clean = self.data_clusters.copy()
                    drop_list2 += DataCleaning.clean_noise(clean, zs = self.z_score, cluster = 'Cluster1', x = x, final = final)

                    self.drop_list += DataCleaning.common_member(drop_list2, self.data_clusters[self.data_clusters['Cluster2'].isnull() == True].index.tolist())

                    for item in self.data_clusters['Cluster1'].unique():
                        if len(self.data_clusters[self.data_clusters['Cluster1'] == item]) < 3:
                            print(self.data_clusters[self.data_clusters['Cluster1'] == item]) #display
                            self.drop_list += self.data_clusters[self.data_clusters['Cluster1'] == item].index.tolist()
                            print(f'Cleaned Cluster1 = {item} due to insufficient data')

                    self.data_clusters_clean = self.data_clusters.copy()
                    self.data_clusters_clean.drop(self.drop_list, inplace=True)

                    self.cluster2_list = [x for x in sorted(set(self.data_clusters_clean['Cluster2'].unique())) if ~np.isnan(x)] # to be used in prediction                
                    print(f'\n\n\tcleaned {100*percent:.1f}% of the data with zs0 = {self.zs0}, zs1 = {self.z_score}, zs2 = {round(self.z_score * x, 3)}\n\n')
                
                control = True              
            # Increase zs1 by 0.05 and reset zs2 = zs1 / 2 if zs2 = zs1 and percentage of cleaned data is greater than 30%
            elif (round(x, 2) == 1) and (percent > self.max_clean_perc):
                print(f'\n\nreached zs2 = zs1 increasing zs1 from {self.z_score} to {round((self.z_score + 0.05), 2)}\n')
                self.z_score += 0.05
                self.z_score = round(self.z_score, 2)
                x = 0.45

            # Increase zs2 if zs2 < zs1 and percentage of cleaned data is greater than 30%
            else:
                print(f'\ncleaned {100*percent:.1f}% of the data with zs0 = {self.zs0}, zs1 = {self.z_score}, zs2 = {round(self.z_score * x, 3)}, x: {round(x, 2)}')
                print(f'increasing zs2 to {round((self.z_score * (x + 0.05)), 3)}')
            
            # Increase zs0 by 0.1 and set zs1 to its initial value if zs1 reaches or passes zs0
            if self.z_score >= self.zs0:
                print(f'\n\tincreasing zs0 from {self.zs0} to {round((self.zs0 + 0.1), 2)}\n')
                self.zs0 += 0.10
                self.zs0 = round(self.zs0, 2)
                self.z_score = z_score_initial
                self.z_score = round(self.z_score, 2)
                x = 0.45

    def tr_te_split(self):
        """
        Prepares the train_test splits of the data.

        Returns:
        None
        """
        print('\nPreparing train_test splits...', end='')
        X = self.data_clusters_clean.copy()
        self.x_train, self.x_test = train_test_split(X, test_size = self.test_size, random_state = 42)
        print('Done!\n')

    def save_results(self):
        """
        Save the current state of the object in a binary file using the Pickle module.
    
        Args:
        self: An instance of the class.
    
        Returns:
        None
        """
        with open('data_cleaning.pickle', 'wb') as f:
            pickle.dump(self, f)
