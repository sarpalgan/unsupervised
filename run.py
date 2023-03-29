import pickle

from data_preprocess import DataPreprocessing
from data_cleaner import DataCleaning
from prediction import Prediction

############################################## Preprocessing
def Preprocess():

    data_preprocessing = DataPreprocessing()

    data_preprocessing.drop_columns()
    data_preprocessing.tedarik_suresi_hesapla()
    data_preprocessing.drop_long_time()
    data_preprocessing.find_cat_fts()
    data_preprocessing.display_correlation_and_encode_data()
    data_preprocessing.cluster_pca()
    data_preprocessing.cluster()
    data_preprocessing.save_results()

############################################## Cleaning
def Clean():

    with open('data_preprocessing.pickle', 'rb') as f:
        data_preprocessing = pickle.load(f)

    data_cleaning = DataCleaning()

    data_cleaning.df_coef = data_preprocessing.df_coef
    data_cleaning.data_clusters = data_preprocessing.data_clusters
    data_cleaning.data =  data_preprocessing.data

    data_cleaning.cleaning()
    data_cleaning.tr_te_split()
    data_cleaning.save_results()

############################################## Prediction
def Predict(mode='predict'):

    with open('data_cleaning.pickle', 'rb') as f:
        data_cleaning = pickle.load(f)

    with open('data_preprocessing.pickle', 'rb') as f:
        data_preprocessing = pickle.load(f)

    data_prediction = Prediction()

    data_prediction.data_clusters_clean = data_cleaning.data_clusters_clean
    data_prediction.cols_list = data_preprocessing.cols_list
    data_prediction.data_onehot_clusters = data_preprocessing.data_onehot_clusters
    data_prediction.ft_wt = data_preprocessing.ft_wt
    data_prediction.kmeans = data_preprocessing.kmeans
    data_prediction.kmeans2 = data_preprocessing.kmeans2
    data_prediction.cluster2_list = data_cleaning.cluster2_list
    data_prediction.pca = data_preprocessing.pca
    data_prediction.x_test = data_cleaning.x_test
    data_prediction.x_train = data_cleaning.x_train

    if mode == 'test':
        data_prediction.test_preds()

    if mode == 'predict':

        ###   TODO   ###

        df_to_predict = data_prediction.x_test.loc[[8743], ['BASKITIPIID', 'month', 'ISLEMGRUBUID', 'RECETETURUID', 'USTASAMAID', 'PROSESID', 'KALINLIKID', 'KUMASCINSIID', 'SIPARISPROSESID']]
        
        ###   TODO   ###

        data_prediction.df_to_predict = df_to_predict
        data_prediction.predict_new()
