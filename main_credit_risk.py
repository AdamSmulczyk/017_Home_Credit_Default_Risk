#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
from models_credit_risk import *
from data_processor_credit_risk import preprocess_data, handle_outliers, convert_types_2, convert_bool,dataset_stabilizer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder


def main():
    # -- loading_data --------------------  
    print('-' * 80)
    print('train')   
    train_file = 'application_train.csv'
    train = preprocess_data(train_file)

    print('-' * 80)
    print('test')
    test_file = 'application_test.csv'
    test = preprocess_data(test_file)

    print('-' * 80)
    print('bureau')
    bureau_file = 'bureau.csv'
    bureau = preprocess_data(bureau_file)

    print('-' * 80)
    print('bureau_balance')
    bureau_balance_file = 'bureau_balance.csv'
    bureau_balance = preprocess_data(bureau_balance_file)

    print('-' * 80)
    print('credit_card_balance')
    credit_card_balance_file = 'credit_card_balance.csv'
    credit_card_balance = preprocess_data(credit_card_balance_file)

    print('-' * 80)
    print('installments_payments')
    installments_payments_file = 'installments_payments.csv'
    installments_payments = preprocess_data(installments_payments_file)

    print('-' * 80)
    print('POS_CASH_balance')
    POS_CASH_balance_file = 'POS_CASH_balance.csv'
    POS_CASH_balance = preprocess_data(POS_CASH_balance_file)

    print('-' * 80)
    print('previous_application')
    previous_application_file = 'previous_application.csv'
    previous_application = preprocess_data(previous_application_file)
    
    # -- preprocess_data -------------------- 
    print("Pre-processing data...")
    
    lista = [
    (train),
    (bureau),
    (bureau_balance),
    (credit_card_balance),
    (installments_payments),
    (POS_CASH_balance),
    (previous_application)]

    # changing columns with 2 values in bool type   
    for df in lista:    
        object_to_boolen = [i for i in df.columns if 
                        df[i].nunique() < 3]        
        for j in object_to_boolen:
             df[j] = df[j].astype('bool')

    # changing columns with  float16->float32  to fill nulls by mean         
    for df in lista: 
        for col in df.select_dtypes(include=['float16']).columns:
            df[col] = df[col].astype('float32')

    # drop columns with missing values > 85%
    for i in lista:   
        Missing = i.isna().mean()*100
        colums_to_drop = i.columns[Missing>60] # columns with the missing values more than 60% of Data
        i.drop(columns = colums_to_drop, inplace=True)  


    for i in lista:
        i = dataset_stabilizer(i)        

    #Pre-processing buro_balance
    print('Pre-processing buro_balance...')
    buro_grouped_size = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
    buro_grouped_max = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
    buro_grouped_min = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()
    buro_counts = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)
    buro_counts_unstacked = buro_counts.unstack('STATUS')
    buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size
    buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min
    buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max
    bureau = bureau.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')


    #Pre-processing buro
    bureau = bureau.drop(columns=['CREDIT_TYPE'])
    bureau = pd.get_dummies(bureau, columns=['CREDIT_ACTIVE', 'CREDIT_CURRENCY'])

    print('Pre-processing buro...')
    #One-hot encoding of categorical features in buro data set
    buro_cat_features = [bcol for bcol in bureau.columns if bureau[bcol].dtype == 'object']
    buro = pd.get_dummies(bureau, columns=buro_cat_features)
    avg_buro = bureau.groupby('SK_ID_CURR').mean()
    avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
    del avg_buro['SK_ID_BUREAU']

    #Pre-processing previous_application
    previous_application = previous_application.drop(columns=['NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START',
           'FLAG_LAST_APPL_PER_CONTRACT', 'NAME_CASH_LOAN_PURPOSE',
           'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON',
           'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY',
           'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE',
           'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION','DAYS_DECISION', 'SELLERPLACE_AREA',
           'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
           'DAYS_LAST_DUE', 'DAYS_TERMINATION'])


    print('Pre-processing previous_application...')
    #One-hot encoding of categorical features in previous application data set
    prev_cat_features = [pcol for pcol in previous_application.columns if previous_application[pcol].dtype == 'category']
    prev = pd.get_dummies(previous_application, columns=prev_cat_features)
    avg_prev = previous_application.groupby('SK_ID_CURR').mean()
    cnt_prev = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
    del avg_prev['SK_ID_PREV']


    #Pre-processing POS_CASH
    from sklearn.preprocessing import  LabelEncoder
    print('Pre-processing POS_CASH...')
    le = LabelEncoder()
    POS_CASH_balance['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH_balance['NAME_CONTRACT_STATUS'].astype(str))
    nunique_status = POS_CASH_balance[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    nunique_status2 = POS_CASH_balance[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
    POS_CASH_balance['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    POS_CASH_balance['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
    POS_CASH_balance.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)


    #Pre-processing credit_card
    print('Pre-processing credit_card...')
    credit_card_balance['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card_balance['NAME_CONTRACT_STATUS'].astype(str))
    nunique_status = credit_card_balance[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    nunique_status2 = credit_card_balance[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
    credit_card_balance['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    credit_card_balance['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
    credit_card_balance.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)


    #Pre-processing payments
    print('Pre-processing payments...')
    avg_payments = installments_payments.groupby('SK_ID_CURR').mean()
    avg_payments2 = installments_payments.groupby('SK_ID_CURR').max()
    avg_payments3 = installments_payments.groupby('SK_ID_CURR').min()
    del avg_payments['SK_ID_PREV']



    #Join data bases
    print('Joining databases...')
    train = train.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(POS_CASH_balance.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(credit_card_balance.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
    
    
    
    
    # -- process_data --------------------
    print("Processing data...")
    X = train.drop(columns=['SK_ID_CURR','TARGET']).copy()
    y = train['TARGET']
       
    # Feature engineering
    X['app EXT_SOURCE mean'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
    X['app EXT_SOURCE std'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis = 1)
    X['app EXT_SOURCE prod'] = X['EXT_SOURCE_1'] * X['EXT_SOURCE_2'] * X['EXT_SOURCE_3']
    X['app EXT_SOURCE_1 * EXT_SOURCE_2'] = X['EXT_SOURCE_1'] * X['EXT_SOURCE_2']
    X['app EXT_SOURCE_1 * EXT_SOURCE_3'] = X['EXT_SOURCE_1'] * X['EXT_SOURCE_3']
    X['app EXT_SOURCE_2 * EXT_SOURCE_3'] = X['EXT_SOURCE_2'] * X['EXT_SOURCE_3']
    X['app EXT_SOURCE_1 * DAYS_EMPLOYED'] = X['EXT_SOURCE_1'] * X['DAYS_EMPLOYED']
    X['app EXT_SOURCE_2 * DAYS_EMPLOYED'] = X['EXT_SOURCE_2'] * X['DAYS_EMPLOYED']
    X['app EXT_SOURCE_3 * DAYS_EMPLOYED'] = X['EXT_SOURCE_3'] * X['DAYS_EMPLOYED']
    X['app EXT_SOURCE_1 / DAYS_BIRTH'] = X['EXT_SOURCE_1'] / X['DAYS_BIRTH']
    X['app EXT_SOURCE_2 / DAYS_BIRTH'] = X['EXT_SOURCE_2'] / X['DAYS_BIRTH']
    X['app EXT_SOURCE_3 / DAYS_BIRTH'] = X['EXT_SOURCE_3'] / X['DAYS_BIRTH']
    X['app AMT_INCOME_TOTAL / CNT_FAM_MEMBERS'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
    X['app AMT_INCOME_TOTAL / CNT_CHILDREN'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
    X['app DAYS_LAST_PHONE_CHANGE / DAYS_BIRTH'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH']
    X['app DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_EMPLOYED']
    X['app DAYS_EMPLOYED - DAYS_BIRTH'] = X['DAYS_EMPLOYED'] - X['DAYS_BIRTH']
    X['app DAYS_EMPLOYED / DAYS_BIRTH'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
    X['app CNTa_CHILDREN / CNT_FAM_MEMBERS'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']

    print("Handle outliers...")
    X = handle_outliers(X)
    print("Convert types...")
    X = convert_types_2(X)
    X = convert_bool(X)

    SEED = 42
    XGB_Params =    { 'max_depth': 13, 
                      'min_child_weight': 5, 
                      'learning_rate': 0.02,
                      'colsample_bytree': 0.6, 
                      'max_bin': 3000, 
                      'n_estimators': 1500,
                      'random_state': SEED}
    
    XGB_Params_2 = { 'subsample': 0.65,
                     'penalty': 'l1',
                     'n_estimators': 150,
                     'min_child_weight': 11,
                     'max_depth': 4,
                     'learning_rate': 0.016729945602749863,
                     'gamma': 0.08,
                     'random_state': SEED}

    print("Initializing workflow...")
#     RFC_1 = RandomForestModel(n_estimators=50, random_state=SEED)
#     XGB_2 = XGBoostModel(**XGB_Params)

#     voting_estimators = [('RandomForest', RFC_1.model), ('XGBoost', XGB_2.model)]
#     workflow_voting = Workflow_9()
#     workflow_voting.run_workflow(
#         model_name='VotingModel',
#         model_kwargs={'estimators': voting_estimators, 'voting': 'soft', 'weights': [1.0, 2.0]},
#         X=X,
#         y=y,
#         test_size=0.2,
#         random_state=42,
#         scoring='r2'
#     )
    

    RFC_1 = Workflow_9()
    RFC_1.run_workflow(
        model_name='XGBoostModel',
        model_kwargs=XGB_Params_2,
        X=X,
        y=y,
        test_size=0.2,
        random_state=42,
        scoring='accuracy'
    )

if __name__ == "__main__":
    main()

