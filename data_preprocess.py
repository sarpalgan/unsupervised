import pandas as pd
import os
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import pickle
import configparser

from datetime import date, timedelta
from typing import List
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
from scipy.optimize import curve_fit
#from IPython.display import display_html 

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)

class DataPreprocessing:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('ayarlar.ini')

        self.data_path: str = config.get('DUZENLEME', 'data_path') 
        self.data = pd.read_csv(self.data_path + '.csv', dtype=str, sep=';', encoding='latin').fillna('BOS')
        self.qtv_fts: List[str] = config.get('DUZENLEME', 'qtv_fts').split(', ') #['SIPARISKILO', 'SIPARISMETRE']
        self.morethan: int = config.getint('GENEL', 'MaxTerminSuresi')
        self.pca_ratio: float = config.getfloat('GENEL', 'PCAOrani')

        #self.time_cols = ['PARTIACILISTARIHI', 'PARTIILKSEVKTARIHI', 'time']  # Will be deleted
        #self.time_cols = ['SIPARISTARIHI', 'GELDIGITARIH', 'time']  # Will be deleted
        self.drop_clmns = ['ICTAMIRVAR', 'LYCRATIPIID']                       # Will be deleted
        self.ini_len = len(self.data)
        self.ft_wt = []
        self.cat_fts = None
        self.column_name = None
        self.df_coef = None
        self.data_onehot = None
        self.data_onehot_clusters = None
        self.data_onehot_clusters_pca = None
        self.pca = None
        self.kmeans = None
        self.kmeans2 = None        
        self.K_final = None
        self.K = None
        self.data_clusters = None
        self.cols_list = None
        
    def drop_columns(self):                                                   # Will be deleted
        for item in set(self.drop_clmns) & set(self.data.columns.to_list()):
          self.data = self.data.drop(columns=item)
    
    def tedarik_suresi_hesapla(self, days = [], max_time_limit = 0):

        columns = ['SIPARISTARIHI', 'GELDIGITARIH'] #self.time_cols
        data = self.data.copy()

        for i in range(len(data)):
            data.at[i, columns[0]] = datetime.datetime.strptime(data[columns[0]][i], '%d.%m.%Y')
            data.at[i, columns[1]] = datetime.datetime.strptime(data[columns[1]][i], '%d.%m.%Y')

        def hafta_sonlarini_bul(start: date, end: date, days = []) -> List[date]:
            total_days: int = (end - start).days + 1
            week_ends = []
            if 'Cumartesi' in days:
                week_ends.append(5)
            if 'Pazar' in days:
                week_ends.append(6)
            all_days = [start + timedelta(days=day) for day in range(total_days)]
            return [day.strftime("%c") for day in all_days if day.weekday() in week_ends]
  
        data.insert(0,"day", None)
        data = data.reindex(columns = [col for col in data.columns if col != "day"] + ["day"])
        data.insert(0,"month", None)
        data = data.reindex(columns = [col for col in data.columns if col != "month"] + ["month"])
        data.insert(0,"week", None)
        data = data.reindex(columns = [col for col in data.columns if col != "week"] + ["week"])
        data.insert(0,"week_day", None)
        data = data.reindex(columns = [col for col in data.columns if col != "week_day"] + ["week_day"])

        if "time" not in data.columns:
            data.insert(0,"time", None)

        data = data.reindex(columns = [col for col in data.columns if col != "time"] + ["time"])

        for i in range(len(data)):
            date_start: date = data[columns[0]][i]
            date_end: date = data[columns[1]][i]
            week_day_code = date_start.weekday()
            data.at[i, "month"] = date_start.month
            data.at[i, "day"] = date_start.day
        
            if date_start.day <= 7:
                data.at[i, "week"] = 1
            elif date_start.day > 7 and date_start.day <= 14:
                data.at[i, "week"] = 2
            elif date_start.day > 14 and date_start.day <= 21:
                data.at[i, "week"] = 3
            elif date_start.day > 21:
                data.at[i, "week"] = 4

            hafta_sonlarini = len(hafta_sonlarini_bul(date_start, date_end, days = days))

            if ((date_end - date_start).days - hafta_sonlarini) < 1:
                data.at[i, "time"] = (date_end - date_start).days
            else:
                if max_time_limit > 0 and ((date_end - date_start).days - hafta_sonlarini) > max_time_limit:
                    data.at[i, "time"] = max_time_limit
                else:
                    data.at[i, "time"] = (date_end - date_start).days - hafta_sonlarini

            if week_day_code == 0:
                data.at[i, "week_day"] = '0'
            elif week_day_code == 1:
                data.at[i, "week_day"] = '1'
            elif week_day_code == 2:
                data.at[i, "week_day"] = '2'
            elif week_day_code == 3:
                data.at[i, "week_day"] = '3'
            elif week_day_code == 4:
                data.at[i, "week_day"] = '4'
            elif week_day_code == 5:
                data.at[i, "week_day"] = '5'
            else:
                data.at[i, "week_day"] = '6'

        data.drop([columns[0], columns[1], 'TEDARIKSURESI', 'week'], inplace=True, axis=1)    
        self.data = data

    def drop_long_time(self):
        self.data['time'] = self.data['time'].astype(int)
        self.data = self.data[self.data['time']<=self.morethan].reset_index(drop=True)

    def find_cat_fts(self):
        self.cat_fts = self.data.columns.to_list()
        for item in set(self.qtv_fts) & set(self.data.columns.to_list()):
            self.cat_fts.remove(item)
        self.cat_fts.remove('time')
        self.column_name = self.cat_fts

    def correlation_ratio(categories, values):
        categories = np.array(categories)
        values = np.array(values)
    
        ssw = 0
        ssb = 0
        for category in set(categories):
            subgroup = values[np.where(categories == category)[0]]
            ssw += sum((subgroup - np.mean(subgroup)) ** 2)
            ssb += len(subgroup) * (np.mean(subgroup) - np.mean(values)) ** 2

        return math.sqrt(ssb / ssw)

    def one_hot(df, ft):      
        #print("one hot encoding ", ft[0], "...")
        df[ft[0]] = df[ft[0]].astype(str)
        df[ft[0]] = str(ft[0]) + '_' + df[ft[0]] 
        dum = pd.get_dummies(df[ft[0]]) * ft[1]
        df = df.drop(ft[0], axis = 1)
        df = df.join(dum)
        #print(ft[0], "encoded.")
        return df

    def display_correlation_and_encode_data(self):
        coef_list = []

        for i in range(len(self.cat_fts)):
            coef = DataPreprocessing.correlation_ratio(self.data[self.cat_fts[i]], self.data['time'].astype(float))
            coef_list.append(coef)
    
        self.df_coef = pd.DataFrame(coef_list, columns = ['coef'])
        self.df_coef["columns_name"] = self.cat_fts

        self.df_coef = self.df_coef.sort_values('coef', ascending = False)
        self.df_coef = self.df_coef.reset_index(drop=True)

        self.df_coef['percentage'] = ''
        for i in range(len(self.df_coef)):
            self.df_coef['percentage'][i] = self.df_coef['coef'][i] / sum(self.df_coef['coef'])
        print(self.df_coef) #display

        for i in range(len(self.df_coef)):
            if self.df_coef['percentage'][i] < 0.005:
                ft = self.df_coef['columns_name'][i]
                self.data.drop([ft], inplace=True, axis=1)
                self.cat_fts.remove(ft)
                self.df_coef.drop([i], inplace=True)
                print(f'dropped {ft}\t column from data due to low percentage')

        sum_of_perc = 0
        for i in range(len(self.df_coef)):
            sum_of_perc += self.df_coef['percentage'][i]
            if sum_of_perc >= 0.7:
                print(f'\ntotal percentage = {round(sum_of_perc, 2)}%')
                self.cols_list = self.df_coef['columns_name'].to_list()[:i+1]
                break
        
        print(f'\ncolumns: {self.cols_list}')
        self.K_final = len(self.data[self.cols_list].drop_duplicates())
        print(f'\n\tK_final = {self.K_final}')

        self.data_onehot = self.data.copy()
        self.data_onehot = self.data_onehot[self.cat_fts]

        self.data_onehot_clusters = self.data_onehot.copy()
        self.data_onehot_clusters = self.data_onehot_clusters[self.cols_list]

        ct = self.df_coef['columns_name'].to_list()
        wts = self.df_coef['coef'].to_list()

        for i in range(len(wts)):
            self.ft_wt.append([ct[i], wts[i]])

        for ft in self.ft_wt:
            self.data_onehot  = DataPreprocessing.one_hot(self.data_onehot, ft)
    
    
        for ft in self.ft_wt[:len(self.cols_list)]:
            self.data_onehot_clusters  = DataPreprocessing.one_hot(self.data_onehot_clusters, ft)

    def cluster_pca(self):
        xl = len(self.data_onehot_clusters.columns)
        pca = PCA(self.pca_ratio)
        pca.fit(self.data_onehot_clusters)
        print(f"\ndoing PCA...number of features dropped from {xl} to {pca.n_components_}\n") 
        #print("variance ratio: ", pca.explained_variance_ratio_) 

        self.data_onehot_clusters_pca = pca.transform(self.data_onehot_clusters)
        self.pca = pca
        
    def cluster(self):
        K = np.arange(2, self.K_final, round(self.K_final/20))
        K1 = [] 
        Sum_of_squared_distances = []

        for num_clusters in K :
            print(f'Clustering with k = {num_clusters}')         
            
            if len(Sum_of_squared_distances) > 2:
                kmeans2_old = self.kmeans2
                kmeans_clusters2_old = kmeans_clusters2
            
            self.kmeans2 = KMeans(n_clusters = num_clusters, random_state=42).fit(self.data_onehot_clusters_pca)
            kmeans_clusters2 = self.kmeans2.predict(self.data_onehot_clusters_pca)
            
            Sum_of_squared_distances.append(self.kmeans2.inertia_)
            K1.append(num_clusters)

            self.data_clusters = self.data.copy()
            self.data_clusters.insert(0, "Cluster2", kmeans_clusters2, True)

            for i in sorted(self.data_clusters['Cluster2'].unique()):
                if len(self.data_clusters[self.data_clusters['Cluster2'] == i]) < 3:
                    self.data_clusters['Cluster2'] = self.data_clusters['Cluster2'].replace(i,np.nan)
            
            if len(Sum_of_squared_distances) > 2:
        
                perc = Sum_of_squared_distances[-1] / Sum_of_squared_distances[-2]
                limit = round(self.data_clusters['Cluster2'].isna().sum() / len(self.data_clusters), 3)

                if (limit > 0.05) or (0.95 <= round(perc, 2)):
            
                    if (limit > 0.05):
                        Sum_of_squared_distances = Sum_of_squared_distances[:-1]
                        K1 = K1[:-1]
                        K2 = K1[-1]
                        print(f'\n\tNaN percentage of cluster2 with k = {num_clusters} is {round(100 * limit, 1)}% therefore k = {K2} for Cluster2')
                        #print(f'\nClustering with k = {K2}')
                        self.data_clusters.drop(['Cluster2'], inplace = True, axis = 1) 

                        self.kmeans2 = kmeans2_old
                        kmeans_clusters2 = kmeans_clusters2_old

                        self.data_clusters.insert(0, "Cluster2", kmeans_clusters2, True)
                        break
                
                    else:
                        K2 = K1[-1]
                        print(f'\n\tloss is decreased by {round(perc, 2)}% therefore k = {K2} for Cluster2')
                        break

                else:
                    self.data_clusters.drop(['Cluster2'], inplace = True, axis = 1) 
                                    
                #######################################
    
        #plt.figure(figsize=(15,10))

        x = K1
        y = Sum_of_squared_distances
        #plt.plot(x, y, 'bx')

        #x.append(K_final)
        #y.append(0)
        #plt.plot(x[-1], y[-1], 'bo')

        def func(x, a, b, c):
            return a * np.exp(-b * np.array(x)) + c

        popt, pcov = curve_fit(func, x, y, bounds=(0, [np.inf, np.inf, np.inf]))

        K2_range = np.arange(1, x[-1], 1)


        kn = KneeLocator(K2_range, func(K2_range, *popt), curve = 'convex', direction = 'decreasing')
        #plt.vlines(int(kn.knee), plt.ylim()[0], plt.ylim()[1], colors = 'g', linestyles = 'solid')
        print('solid: ', int(kn.knee))
        kn_exp = int(kn.knee)
        #plt.plot(K2_range, func(K2_range, *popt), 'g-')
                
                #######################################
            
        #x = K1
        #y = Sum_of_squared_distances
        #plt.plot(x, y, 'bx')
        knees_list = []

        for i in range(5,15):
            z = np.polyfit(x, y, i)
            x_fit = np.arange(1, x[-1], 1)
            y_fit = np.polyval(z, x_fit)
    
            kn = KneeLocator(x_fit, y_fit, curve = 'convex', direction = 'decreasing', polynomial_degree = i)
            knees_list.append(int(kn.knee))

        a = [sum([abs(x - item) for x in knees_list]) for item in knees_list]
        index = a.index(min(a))
        knee = knees_list[index]

        z = np.polyfit(x, y, range(5,15)[index])
        x_fit = np.arange(1, x[-1], 1)
        y_fit = np.polyval(z, x_fit)
        
        #plt.plot(x_fit, y_fit, 'r--')  

                #######################################              
                        
        self.kmeans = KMeans(n_clusters = kn_exp, random_state=42).fit(self.data_onehot_clusters_pca)
        kmeans_clusters = self.kmeans.predict(self.data_onehot_clusters_pca)
        self.data_clusters.insert(0, "Cluster1", kmeans_clusters, True)

        #plt.plot(K1, Sum_of_squared_distances, 'x')
        #plt.xlabel('Values of k')
        #plt.ylabel('Sum of squared distances/Inertia')
        #plt.title('Elbow Method For Optimal k')
        #plt.vlines(knee, plt.ylim()[0], plt.ylim()[1], colors = 'r', linestyles = 'dashed')
        #plt.show()
        print(f'degree of polyfit = {range(5,15)[index]}')
        print(f'\n\tCluster1 = {kn_exp}')
        print(f'\n\tCluster2 = {K2}')
        print(f'dashed = {knee}')

        self.data_clusters.head()

    def save_results(self):
        with open(f'{self.data_path}_data_preprocessing.pickle', 'wb') as f:
            pickle.dump(self, f)