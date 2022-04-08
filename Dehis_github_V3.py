# -*- coding: utf-8 -*-
"""
Last modified 20 DEC 2020

Greg Booth, M.D.
Naval Biotechnology Group
Naval Medical Center Portsmouth
Portsmouth, VA 23323
in collaboration with:
    Jacob Cole, M.D.
    Scott Hughey, M.D.
    Phil Geiger, M.D.
"""


from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from numpy import array
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import shap


def focus_cpt(data3,cpt_code):
    data=data3[(data3['CPT']==cpt_code)|(data3['OTHERCPT1']==cpt_code)|
                (data3['OTHERCPT2']==cpt_code)|(data3['OTHERCPT3']==cpt_code)|
                (data3['OTHERCPT4']==cpt_code)|(data3['OTHERCPT5']==cpt_code)|
                (data3['OTHERCPT6']==cpt_code)|(data3['OTHERCPT7']==cpt_code)|
                (data3['OTHERCPT8']==cpt_code)|(data3['OTHERCPT9']==cpt_code)|
                (data3['OTHERCPT10']==cpt_code)]
    return data

#CPT code for ex-lap is 49000

def process_lap_data(cpt_code,Train=True):
    cpt_list = ['CPT','OTHERCPT1','OTHERCPT2','OTHERCPT3','OTHERCPT4','OTHERCPT5',
                'OTHERCPT6','OTHERCPT7','OTHERCPT8','OTHERCPT9','OTHERCPT10']

    cpt_code=str(cpt_code)
    #define which NSQIP variables you want to extract so you don't load unnecessary data
    cols3 = ['CPT','OTHERCPT1','OTHERCPT2','OTHERCPT3','OTHERCPT4','OTHERCPT5','OTHERCPT6','OTHERCPT7','OTHERCPT8','OTHERCPT9','OTHERCPT10','EMERGNCY','AGE','SEX','FNSTATUS2','WNDINF','EMERGNCY','PRSEPIS','DIABETES','DYSPNEA','ASACLAS','STEROID','ASCITES','VENTILAT','DISCANCR','HYPERMED','HXCHF','SMOKE','HXCOPD','DIALYSIS','RENAFAIL','HEIGHT','WEIGHT','PRSODM','PRWBC','PRHCT','PRPLATE','PRCREAT','PRBUN','PNAPATOS','OSSIPATOS','DSSIPATOS','SSSIPATOS','TRANSFUS','WNDCLAS','OPTIME','DISCHDEST','DEHIS']
    cols4 = ['CPT','OTHERCPT1','OTHERCPT2','OTHERCPT3','OTHERCPT4','OTHERCPT5','OTHERCPT6','OTHERCPT7','OTHERCPT8','OTHERCPT9','OTHERCPT10','EMERGNCY','Age','SEX','FNSTATUS2','WNDINF','EMERGNCY','PRSEPIS','DIABETES','DYSPNEA','ASACLAS','STEROID','ASCITES','VENTILAT','DISCANCR','HYPERMED','HXCHF','SMOKE','HXCOPD','DIALYSIS','RENAFAIL','HEIGHT','WEIGHT','PRSODM','PRWBC','PRHCT','PRPLATE','PRCREAT','PRBUN','PNAPATOS','OSSIPATOS','DSSIPATOS','SSSIPATOS','TRANSFUS','WNDCLAS','OPTIME','DISCHDEST','DEHIS']

    if Train:
        #load all the datasets from disk (need to request NSQIP datasets first) as data2

    else: 
        #load all the datasets from disk (need to request NSQIP datasets first) as data2

    
    #combine each year into one large dataframe
    data2 = shuffle(data2,random_state = 444)
    data2 = data2.reset_index(drop=True)

    print('Total cpt {:s} ='.format(cpt_code),data2.shape[0])

    
    pos = ['Wound Disruption']
    neg = ['No Complication']
    data2['DEHIS']=data2['DEHIS'].replace(to_replace=neg,value='0')
    data2['DEHIS']=data2['DEHIS'].replace(to_replace=pos,value='1')
    data2['DEHIS']=data2['DEHIS'].astype(int)
    targets_data = data2['DEHIS']
    targets_data=array(targets_data)
    targets_data=targets_data.reshape(-1,1)
    
    #now process all the inputs and handle missing data
    #process BMI
    BMI=[]
    weights1=data2['WEIGHT'].to_numpy()
    heights1=data2['HEIGHT'].to_numpy()
    for i in range(len(data2)):
        if (weights1[i]!=-99) and (heights1[i]!=-99): 
            #convert height and weight to BMI if both are known
            BMI.append((703*weights1[i])/((heights1[i])**2))
        else: 
            BMI.append(-99)
            
    for i in range(len(BMI)):
        if BMI[i]>=70:
            BMI[i]=70
        if BMI[i] < 15 and BMI[i]>0:
            BMI[i]=15
        if (BMI[i]==-99):
            BMI[i]=np.nan 
    
    BMI=array(BMI).reshape(-1,1)
    
    #process age
    data2['Age'] = data2['Age'] .astype(str).replace('\.0', '', regex=True)
    x00=data2['Age']
    x0=x00.copy()
    for i in range(len(x00)):
        if x00.iloc[i]=='90+':
            x0.iloc[i]='90'
        elif x00.iloc[i]=='-99':
            x0.iloc[i]='nan'
    
    x0=x0.replace({'nan':'10'})
    x0=x0.astype(float)
    x0=x0.replace({10:np.nan})
    x0=x0.to_numpy().reshape(-1,1)
    
    
    x1=data2['SEX']
    x1=x1.replace({'NULL':np.nan})
    x1=x1.replace({'male':0,'female':1})
    x1=x1.to_numpy().reshape(-1,1)
    
    x22 = data2['FNSTATUS2']
    x22=x22.replace({'Partially D':'Partially Dependent'})
    x22=x22.replace({'Totally Dep':'Totally Dependent'})
    x22=x22.replace({'Independent':0,'Partially Dependent':1,'Totally Dependent':2,'Unknown':np.nan})
    x2=x22.to_numpy().reshape(-1,1)
    
    x4=data2['ASACLAS']
    x4=x4.replace({'NULL':np.nan})
    x4=x4.replace({'Null':np.nan})
    x4=x4.replace({'None assigned':np.nan})
    x4=x4.replace({'1-No Disturb':1,'2-Mild Disturb':2,'3-Severe Disturb':3,'4-Life Threat':4,'5-Moribund':5})
    x4=x4.to_numpy().reshape(-1,1)
    
    x5=data2['STEROID']
    x5=x5.replace({'NULL':np.nan})
    x5=x5.replace({'NUL':np.nan})
    x5=x5.replace({'No':0,'Yes':1})
    x5=x5.to_numpy().reshape(-1,1)
    
    x6=data2['ASCITES']
    x6=x6.replace({'NULL':np.nan})
    x6=x6.replace({'NUL':np.nan})
    x6=x6.replace({'Ye':'Yes'})
    x6=x6.replace({'No':0,'Yes':1})
    x6=x6.to_numpy().reshape(-1,1)
    
    x77 = data2['PRSEPIS']
    x77=x77.replace({'NULL':np.nan})
    x77=x77.replace({'None':0,'SIRS':0,'Septic':1,'Sepsis':1,'Septic Shock':2})
    x7=x77.to_numpy().reshape(-1,1)
    
    x8=data2['VENTILAT']
    x8=x8.replace({'NULL':np.nan})
    x8=x8.replace({'NUL':np.nan})
    x8=x8.replace({'No':0,'Yes':1})
    x8=x8.to_numpy().reshape(-1,1)
    
    x9=data2['DISCANCR']
    x9=x9.replace({'NULL':np.nan})
    x9=x9.replace({'NUL':np.nan})
    x9=x9.replace({'No':0,'Yes':1})
    x9=x9.to_numpy().reshape(-1,1)
    
    x101 = data2['DIABETES']
    x101=x101.replace({'NULL':np.nan})
    x101=x101.replace({'NON-INSULIN':'ORAL'})
    x101=x101.replace({'NO':0,'ORAL':1,'INSULIN':1,})
    x10=x101.to_numpy().reshape(-1,1)
    
    x11=data2['HYPERMED']
    x11=x11.replace({'NULL':np.nan})
    x11=x11.replace({'NUL':np.nan})
    x11=x11.replace({'No':0,'Yes':1})
    x11=x11.to_numpy().reshape(-1,1)
    
    x13=data2['HXCHF']
    x13=x13.replace({'NULL':np.nan})
    x13=x13.replace({'NUL':np.nan})
    x13=x13.replace({'Ye':'Yes'})
    x13=x13.replace({'No':0,'Yes':1})
    x13=x13.to_numpy().reshape(-1,1)
    
    x14= data2['DYSPNEA']
    x14=x14.replace({'NULL':np.nan})
    x14=x14.replace({'No':0,'MODERATE EXERTION':1,'AT REST':1})
    x14=x14.to_numpy().reshape(-1,1)
    
    x15=data2['SMOKE']
    x15=x15.replace({'NULL':np.nan})
    x15=x15.replace({'NUL':np.nan})
    x15=x15.replace({'No':0,'Yes':1})
    x15=x15.to_numpy().reshape(-1,1)
    
    x16=data2['HXCOPD']
    x16=x16.replace({'NULL':np.nan})
    x16=x16.replace({'NUL':np.nan})
    x16=x16.replace({'No':0,'Yes':1})
    x16=x16.to_numpy().reshape(-1,1)
    
    x17=data2['DIALYSIS']
    x17=x17.replace({'NULL':np.nan})
    x17=x17.replace({'NUL':np.nan})
    x17=x17.replace({'Ye':'Yes'})
    x17=x17.replace({'No':0,'Yes':1})
    x17=x17.to_numpy().reshape(-1,1)
    
    x18=data2['RENAFAIL']
    x18=x18.replace({'NULL':np.nan})
    x18=x18.replace({'NU':np.nan})
    x18=x18.replace({'Ye':'Yes'})
    x18=x18.replace({'No':0,'Yes':1})
    x18=x18.to_numpy().reshape(-1,1)
    
    x19 = BMI.reshape(-1,1)
    
    x20=data2['EMERGNCY']
    x20=x20.replace({'NUL':np.nan,'NULL':np.nan,'No':0,'Yes':1})
    x20=x20.to_numpy().reshape(-1,1)
    
    x21 = data2['WNDINF']
    x21=x21.replace({'No':0,'NULL':np.nan,'Yes':1})
    x21=x21.to_numpy().reshape(-1,1)
    
    x22=data2['PRSODM']
    x22=x22.replace({-99:np.nan})
    x22=x22.to_numpy().reshape(-1,1)
    
    x23=data2['PRWBC']
    x23=x23.replace({-99:np.nan})
    x23=x23.to_numpy().reshape(-1,1)
    
    x24 = data2['PRHCT']
    x24=x24.replace({-99:np.nan})
    x24=x24.to_numpy().reshape(-1,1)
    
    x25 = data2['PRPLATE']
    x25=x25.replace({-99:np.nan})
    x25=x25.to_numpy().reshape(-1,1)
    
    x26 = data2['PRCREAT']
    x26=x26.replace({-99:np.nan})
    x26=x26.to_numpy().reshape(-1,1)
    
    x26b = data2['PRBUN']
    x26b=x26b.replace({-99:np.nan})
    x26b=x26b.to_numpy().reshape(-1,1)
    
    x27=data2['PNAPATOS']
    x27=x27.replace({'No':0,'NULL':0,'NU':0,'Yes':1,'Ye':1})
    x27=x27.to_numpy().reshape(-1,1)
    
    x28=data2['OSSIPATOS']
    x28=x28.replace({'No':0,'NULL':0,'NUL':0,'Ye':1,'Yes':1})
    x28=x28.to_numpy().reshape(-1,1)
    
    x29=data2['DSSIPATOS']
    x29=x29.replace({'No':0,'NULL':0,'NU':0,'Yes':1,'Ye':1})
    x29=x29.to_numpy().reshape(-1,1)
    
    x30=data2['SSSIPATOS']
    x30=x30.replace({'No':0,'NULL':0,'NUL':0,'Yes':1})
    x30=x30.to_numpy().reshape(-1,1)
    
    hold35=[]
    for i in range(len(data2)):
        if (x21[i] ==1 or x28[i]==1 or x29[i]==1 or x30[i]==1):
            hold35.append(1)
        else:
            hold35.append(0)
    x35=array(hold35).reshape(-1,1)
        
    
    x31 = data2['TRANSFUS']
    x31=x31.replace({'Yes':1,'NULL':np.nan,'No':0})
    x31 = x31.to_numpy().reshape(-1,1)
    
    x32=data2['WNDCLAS']
    x32=x32.replace({'NULL':np.nan,'1-Clean':1,'2-Clean/Contaminated':2,'3-Contaminated':3,'4-Dirty/Infected':4})
    x32=x32.to_numpy().reshape(-1,1)
    
    x33=data2['OPTIME']
    x33=x33.replace({-99:np.nan})
    x33=x33.to_numpy().reshape(-1,1)
    x33b=x33[:]
    for i in range(len(x33)):
        if x33[i]<=30:
            x33b[i]=30
        elif x33[i]>=300:
            x33b[i]=300
        
    #put all inputs together into one array
    inputs_aggregate = np.concatenate([x0,x1,x2,x4,x5,x6,x7,x8,x9,x10,x11,x13,x14,x15,x16,x17,x18,x19,x20,x22,x23,x24,x25,x26,x27,x31,x32,x33b,x35],axis=1)

    columns_D = ['AGE','SEX','FXN STATUS', 'ASA','STEROID','ASCITES','PRSEPIS',
                 'VENT','CANCER','DIABETES','HTN','CHF','DYPSNEA','SMOKER','COPD',
                 'DIALYSIS','RENAL FAIL','BMI','EMERGENCY','Na','WBC','HCT','PLT',
                 'CREAT','PNA','TRANSFUS','WND CLASS','OP TIME','AGG INFXN']

    #drop nans
    data3 = inputs_aggregate.copy()
    data4 = targets_data.copy()
    data5 = np.concatenate([data3,data4],axis=1)
    data5=data5[~np.isnan(data5).any(axis=1)]
    print('final size of data for CPT {:s} ='.format(cpt_code),data5.shape)
    
    inputs_aggregate2 = data5[:,:-1]
    targets_data2 = data5[:,-1].reshape(-1,1)
    data_final = pd.DataFrame(inputs_aggregate2,columns = columns_D)
    data_final['TARGETS'] = targets_data2.copy()
    
    #final data pre-processing
    data_final['Na']=data_final['Na'].clip(125,150)
    data_final['WBC']=data_final['WBC'].clip(2,20)
    data_final['HCT']=data_final['HCT'].clip(21,50)
    data_final['PLT']=data_final['PLT'].clip(50,500)
    data_final['CREAT']=data_final['CREAT'].clip(0.5,4)

    print('train data = ',data_final.shape[0])
    
    return data_final.iloc[:,:-1], data_final.iloc[:,-1]


#now for the fun part
#feel free to adjust the hyperparameters within the function
#this will return optimal hyperparameters after bayesian optimization
#This optimization section is modified from some code originally appearing here: https://ayguno.github.io/curious/portfolio/bayesian_optimization.html
def optimized_data(rand_points,search_number,inputs_NSQIP,targets_train):

    def xgboost_bayesian(max_depth,learning_rate,colsample_bytree, min_child_weight,reg_alpha,gamma):
        
        optimizer = xgb.XGBClassifier(max_depth=int(max_depth),
                                               learning_rate= learning_rate,
                                               n_estimators= 500,
                                               reg_alpha = reg_alpha,
                                               gamma = gamma,
                                               nthread = -1,
                                               colsample_bytree = colsample_bytree,
                                               min_child_weight = min_child_weight,
                                               objective='binary:logistic',
                                               seed = 444,
                                               scale_pos_weight = 1)
        roc_auc_holder=[]
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,random_state=444)
        for train_index, test_index in rskf.split(inputs_NSQIP, targets_train):
            x_train, x_test = inputs_NSQIP.iloc[train_index],inputs_NSQIP.iloc[test_index]
            y_train, y_test = targets_train.iloc[train_index], targets_train.iloc[test_index]
            
            optimizer.fit(x_train,y_train.ravel(),eval_set =  [(x_test,y_test.ravel())], eval_metric = 'logloss',early_stopping_rounds = 10)
            probs = optimizer.predict_proba(x_test)
            probs = probs[:,1]
            roc1 = roc_auc_score(y_test,probs)
            roc_auc_holder.append(roc1) 

        return sum(roc_auc_holder)/len(roc_auc_holder)
    
    hyperparameters = {
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.1),
        'reg_alpha': (0, 0.3),
        'gamma': (0, 0.5),
        'min_child_weight': (5,30),
        'colsample_bytree': (0.1, 1)
    }
    
    bayesian_object = BayesianOptimization(f = xgboost_bayesian, 
                                 pbounds =  hyperparameters,
                                 verbose = 2)
    
    bayesian_object.maximize(init_points=rand_points,n_iter=search_number,
                             acq='ucb', kappa= .01, alpha = 1e-7)
    
    #now we have optimal parameters
    Dehis1 = xgb.XGBClassifier(max_depth=int(bayesian_object.max['params']['max_depth']),
                                           learning_rate= bayesian_object.max['params']['learning_rate'],
                                           n_estimators= 500,
                                           reg_alpha = bayesian_object.max['params']['reg_alpha'],
                                           gamma = bayesian_object.max['params']['gamma'],
                                           nthread = -1,
                                           colsample_bytree = bayesian_object.max['params']['colsample_bytree'],
                                           min_child_weight = bayesian_object.max['params']['min_child_weight'],
                                           objective='binary:logistic',
                                           seed = 444,
                                           scale_pos_weight = 1)

    #now refit all data an determine optimal n_estimators via cross-val
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,random_state=444)
    best_it_hold = []
    for train_index, test_index in rskf.split(inputs_NSQIP, targets_train):
        x_train, x_test = inputs_NSQIP.iloc[train_index],inputs_NSQIP.iloc[test_index]
        y_train, y_test = targets_train.iloc[train_index], targets_train.iloc[test_index]
        Dehis1.fit(x_train,y_train.ravel(), eval_set=[(x_test,y_test.ravel())],eval_metric = 'logloss',early_stopping_rounds=10)
        best_it_hold.append(Dehis1.best_iteration)
        probs = Dehis1.predict_proba(x_test)
        probs = probs[:,1]
        roc1 = roc_auc_score(y_test,probs)
        
    best_training_iteration = int(round(sum(best_it_hold)/len(best_it_hold)))

    optimized_params =            {'max_depthV1':int(round(bayesian_object.max['params']['max_depth'])),
                                   'colsample_bytreeV1':bayesian_object.max['params']['colsample_bytree'],
                                   'gammaV1':bayesian_object.max['params']['gamma'],
                                   'learning_rateV1': bayesian_object.max['params']['learning_rate'],
                                   'min_child_weightV1':bayesian_object.max['params']['min_child_weight'],
                                   'reg_alphaV1':bayesian_object.max['params']['reg_alpha'],
                                   'scale_pos_weightV1':1,
                                   'best_training_iteration':best_training_iteration,
                                   'roc':bayesian_object.max['target']}
    
    return optimized_params


def model_to_train(optimized_params):
    Dehis1 = xgb.XGBClassifier(max_depth = optimized_params['max_depthV1'],
                    learning_rate= optimized_params['learning_rateV1'],
                    n_estimators= optimized_params['best_training_iteration'],
                    reg_alpha = optimized_params['reg_alphaV1'],
                    gamma = optimized_params['gammaV1'],
                    nthread = -1,
                    colsample_bytree = optimized_params['colsample_bytreeV1'],
                    min_child_weight = optimized_params['min_child_weightV1'],
                    objective='binary:logistic',
                    seed = 444,
                    scale_pos_weight = 1)
    return Dehis1


def top_features(inputs_NSQIP,targets_train,optimized_params):

    cols = inputs_NSQIP.columns
    Dehis1 = model_to_train(optimized_params)
    Dehis1.fit(inputs_NSQIP,targets_train.ravel(),eval_metric = 'logloss')
    explainer=shap.TreeExplainer(Dehis1)
    shap_values=explainer.shap_values(inputs_NSQIP)

    values = np.abs(shap_values).mean(axis=0)
    features = pd.DataFrame([inputs_NSQIP.columns.tolist(), values.tolist()]).T
    features.columns = ['predictor', 'importance']
    features = features.sort_values('importance', ascending=False)
    final_features = features['predictor'][0:15]

    shap_values2=pd.DataFrame(shap_values,columns=cols)
    shap_values22=shap_values2[final_features].to_numpy()
    shap.summary_plot(shap_values22,inputs_NSQIP[final_features],show=False)

    return final_features


def bootstrap_test(inputs_NSQIP,targets_train,inputs_test,targets_test,optimized_params,final_features):

    Dehis1 = xgb.XGBClassifier(max_depth = optimized_params['max_depthV1'],
                            learning_rate= optimized_params['learning_rateV1'],
                            n_estimators= optimized_params['best_training_iteration'],
                            reg_alpha = optimized_params['reg_alphaV1'],
                            gamma = optimized_params['gammaV1'],
                            nthread = -1,
                            colsample_bytree = optimized_params['colsample_bytreeV1'],
                            min_child_weight = optimized_params['min_child_weightV1'],
                            objective='binary:logistic',
                            seed = 444,
                            scale_pos_weight = 1)
    
    Dehis1.fit(inputs_NSQIP[final_features],targets_train.ravel(), eval_metric = 'logloss')
    preds= Dehis1.predict_proba(inputs_test[final_features])
    preds=preds[:,1]
    
    roc_hold=[]
    brier_hold=[]
    index_holder=range(0,len(targets_test))
    j=0
    y_test=[]
    y_preds=[]
    for i in range(1000):
        y_test=[]
        y_preds=[]
        boot = np.random.choice(index_holder,size=(len(index_holder)),replace=True)
        for k in range(len(boot)):
            y_test.append(targets_test[boot[k]])
            y_preds.append(preds[boot[k]])
            
        y_test=np.array(y_test)
        y_preds=np.array(y_preds)
    
        auc_roc = roc_auc_score(y_test,y_preds)
        b_loss = brier_score_loss(y_test, y_preds)
        print('be patient, iteration',j)
        j=j+1
        roc_hold.append(auc_roc)
        brier_hold.append(b_loss)
    
    av_brier = sum(brier_hold)/len(brier_hold)
    av_roc = sum(roc_hold)/len(roc_hold)
    roc_hold=sorted(roc_hold)
    brier_hold=sorted(brier_hold)
    print('ROC AUC = ',av_roc,', 95% CI = ',roc_hold[25],'to ',roc_hold[975])
    print('Brier = ',av_brier,', 95% CI = ',brier_hold[25],'to ',brier_hold[975])



#calibration
def plot_cal(inputs_new,targets_train,test_new,targets_test,optimized_params,final_features):
    Dehis1 = model_to_train(optimized_params)
    Dehis1.fit(inputs_new[final_features],targets_train.ravel(), eval_metric = 'logloss')
    probs = Dehis1.predict_proba(test_new[final_features])
    probs = probs[:,1]
    
    #brier score
    b_loss = brier_score_loss(targets_test,probs)
    
    #calibration curve
    fop, mpv = calibration_curve(targets_test,probs,n_bins = 10,strategy='quantile')
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    ax.plot(mpv,fop,'b',label='Brier Score = {:.3f}'.format(b_loss))
    ax.plot([0,.06],[0,.06],'k--',label='Perfect Calibration')
    ax.legend(loc = 'lower right')
    ax.plot(mpv,fop,'b.')
    ax.set_title('Calibration Curve for Holdout Data')
    ax.set_xlabel('Mean Predicted Value')
    ax.set_ylabel('Fraction of Positives')
    plt.show()


#SHAP
def shap_plots(x_train, y_train, x_test, y_test,optimized_params,final_features):
    
    #shap!
    Dehis1 = model_to_train(optimized_params)
    Dehis1.fit(x_train[final_features],y_train,eval_metric='logloss')
    probs=Dehis1.predict_proba(x_test[final_features])[:,1].reshape(-1,1)
    explainer=shap.TreeExplainer(Dehis1)
    shap_values=explainer.shap_values(x_train[final_features])

    #shap dependency
    for i in final_features:
            shap.dependence_plot(i,shap_values,inputs_df,interaction_index=None,show=False)

    #now let's do some sensitivity analyses
    #find 2 most confident and correct
    #probs=preds
    y_data = y_data_test.to_numpy().reshape(-1,1)
    test_new=x_test[final_features].copy()
    shap_values=explainer.shap_values(test_new)
    j=0
    k=1
    sens_true_pos_hold =[]
    sens_false_neg_hold = []
    least_conf=[]
    preds_hold = np.concatenate([probs,y_data],axis=1)
    preds_hold_sorted=preds_hold.copy()
    preds_hold_sorted=preds_hold_sorted[preds_hold_sorted[:,0].argsort()]
    #first true positives
    while j<3:
        #c = np.where(probs==probs.max())
        c = preds_hold_sorted[-k,0]
        if preds_hold_sorted[-k,1]==1:
            sens_true_pos_hold.append(np.where(probs==c))
            j=j+1
        k=k+1
    #now false negatives
    j=0
    k=1
    while j<3:
        #c = np.where(probs==probs.max())
        c = preds_hold_sorted[-k,0]
        if preds_hold_sorted[-k,1]==0:
            sens_false_neg_hold.append(np.where(probs==c))
            j=j+1
        k=k+1
        
    #now find the least confident predictors
    j=0
    k=3;
    least_conf=[]
    while len(least_conf)<3:
        #c = np.where(probs==probs.max())
        c = preds_hold_sorted[j,0]
        hold1=np.where(probs==c)
        hold1=array(hold1)
        if hold1.shape[1]>1:
            for i in range((hold1.shape[1])):
                least_conf.append(hold1[0,i])
            k=k-hold1.shape[1]
        else:
            least_conf.append(hold1)
        j=j+1
        print("j =",j)
        print('k=',k)
    least_conf=least_conf[0:3]
    least_conf=array(least_conf)
    least_conf=least_conf.astype(int)
        
    #account for any duplicates
    sens_false_neg_hold = np.concatenate(sens_false_neg_hold,axis=1)
    sens_false_neg_hold = sens_false_neg_hold[0,0:2]
    sens_true_pos_hold = np.concatenate(sens_true_pos_hold,axis=1)
    sens_true_pos_hold = sens_true_pos_hold[0,0:2]
    
    sens_true_pos_hold=np.squeeze(sens_true_pos_hold)
    sens_false_neg_hold=np.squeeze(sens_false_neg_hold)
    least_conf=np.squeeze(least_conf)
    ##now we have indices of most confident correct and most confident but incorrect
    sens_true_pos_hold=array(sens_true_pos_hold)
    data_true_pos = test_new.iloc[sens_true_pos_hold,:]
    
    sens_false_neg_hold=array(sens_false_neg_hold)
    data_false_neg = test_new.iloc[sens_false_neg_hold,:]
    
    #plot all of the force_plots
    #true positives
    #basic formatting for display purposes only
    inputs_test2=test_new.copy()
    
    inputs_test2['BMI']=inputs_test2['BMI'].round(1)
    
    for i in range(0,2):
        shap_display=shap.force_plot(explainer.expected_value,shap_values[sens_true_pos_hold[i],:],inputs_test2.iloc[sens_true_pos_hold[i],:],matplotlib=True,show=False,text_rotation=60)
        print(preds_hold[sens_true_pos_hold[i],0],preds_hold[sens_true_pos_hold[i],1])

    #least confident
    for i in range(0,2):
        shap_display=shap.force_plot(explainer.expected_value,shap_values[least_conf[i,0],:],inputs_test2.iloc[least_conf[i,0],:],matplotlib=True,show=False,text_rotation=60)
        print(preds_hold[least_conf[i],0],preds_hold[least_conf[i],1])


def clinical_impact(x_train,y_train,x_test,targets_test,optimized_params,final_features):
    #clinical impact
    Dehis1 = model_to_train(optimized_params)
    Dehis1.fit(x_train[final_features],y_train,eval_metric='logloss')
    preds=Dehis1.predict_proba(x_test[final_features])[:,1].reshape(-1,1)
    Thresholds = np.linspace(0.001, .10, 100, endpoint=True)
    sens_XGB = []
    spec_XGB = []
    ppv_XGB=[]
    num_tp = []
    num_fn = []
    num_fp = []
    dca = []
    all_treat = []
    no_treat = []
    prevalence = Counter(targets_test)[1]/targets_test.shape[0]
    for j in range(len(Thresholds)):
        y_pred_XGB = [1 if i>Thresholds[j] else 0 for i in preds]
        CM_XGB = confusion_matrix(targets_test, y_pred_XGB)
        #sens and ppv
        tp_XGB = CM_XGB[1,1]
        fp_XGB = CM_XGB[0,1]
        fn_XGB = CM_XGB[1,0]
        tn_XGB = CM_XGB[0,0]
        pr_XGB = tp_XGB/[tp_XGB+fp_XGB]
        rec_XGB = tp_XGB/[tp_XGB+fn_XGB]
        spec_XGB_hold = tn_XGB/[tn_XGB+fp_XGB]
        sens_XGB.append(rec_XGB)
        spec_XGB.append(spec_XGB_hold)
        ppv_XGB.append(pr_XGB)
        num_tp.append(tp_XGB)
        num_fn.append(fn_XGB)
        num_fp.append(fp_XGB)
        dca.append((tp_XGB/(preds.shape[0]))-(fp_XGB/(preds.shape[0]))*(Thresholds[j]/(1-Thresholds[j])))
        no_treat.append(0)
        all_treat.append((prevalence)-(1-prevalence)*(Thresholds[j]/(1-Thresholds[j])))
        
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    ax.plot(Thresholds,no_treat,'k',label='No Treatment')
    ax.plot(Thresholds,all_treat,'b--',label='Treat All')
    ax.plot(Thresholds,dca,'r--',label='Model')
    ax.legend(loc = 'upper right')
    ax.set_title('Decision Curve Analysis')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Net Benefit')
    plt.xlim([0,0.06])
    plt.ylim([-0.005, .02])
    plt.show()
