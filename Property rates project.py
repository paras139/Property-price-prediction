## bangalore property rates project
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') 
data=pd.read_csv('/Users/paras/Downloads/Bengaluru_House_Data.csv')
data.drop(['availability','area_type','balcony','society'],inplace=True,axis=1)
print(data.head(5))
#data.info()
#print(data.columns)
print(data.isna().sum())
data['bath']=data['bath'].fillna(1)
#print(data.isna().sum())
data=data.dropna(how='any')
print(data['size'].unique())
data['BHK']=data['size'].apply(lambda x:int(x.split(' ')[0]))
print(data['BHK'].unique())
print(data['total_sqft'].unique())
# def isfloat(x):
#     try: float(x)
#     except:
#          return False
#     return True  
#print(data[~data['total_sqft'].apply(isfloat)].head(5))
def convert_sqft_to_num(x):
    a=x.split('-')
    if len(a)==2: return (float(a[0])+float(a[1]))/2
    try: return float(x) 
    except: return None    
data['total_sqft']=data['total_sqft'].apply(convert_sqft_to_num)
data.drop(['size'],inplace=True,axis=1)
#print(data.head(15))
data1=data.copy()
data1['price_per_sqft']=data1['price']*100000/data1['total_sqft']
print(data1.head(5))
print(len(data1.location.unique()))
data1.location=data1.location.apply(lambda x:x.strip()) #remove all front and back spaces
location_area=data1.groupby('location')['location'].agg("count").sort_values(ascending=False)
print(location_area)
print(len(location_area[location_area<=10]))
locationlessthan10=location_area[location_area<=10]
#print(locationlessthan10)
print(len(data1.location.unique()))
data1.location=data1.location.apply(lambda x:'others' if x in locationlessthan10 else x )
print(len(data1.location.unique()))
print(data1[data1.total_sqft/data1.BHK<300].head(5))
data2=data1[~(data1.total_sqft/data1.BHK<300)]
print(data2.head(5))
#print(data2.shape)
print(data2["price_per_sqft"].describe().apply(lambda x:format(x,'f')))
def remove_outliners(df):
    df_out=pd.DataFrame()
    for  i ,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
data3=remove_outliners(data2)
print(data3.shape)
import matplotlib.pyplot as plt
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location)&(df.BHK==2)]
    bhk3=df[(df.location==location)&(df.BHK==3)]
    plt.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='Blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color='Green',marker='+',label='3 BHK',s=50)
    plt.xlabel('Total Square Foot')
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
#plot_scatter_chart(data3,'Rajaji Nagar')
#plt.show()
def remove_bkh_outliers(df):
    exclude_indices=np.array([])
    for i, j in df.groupby('location'):
        bhk_sats={}
        for i,BHK_df in j.groupby('BHK'):
            bhk_sats[i]={
                'mean':np.mean(BHK_df.price_per_sqft),
                'count':BHK_df.shape[0]
            }
        for i,BHK_df in j.groupby('BHK'):
            stats=bhk_sats.get(i-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index)
    return df.drop(exclude_indices,axis=0)
data4=remove_bkh_outliers(data3)
print(data4.shape)
#plot_scatter_chart(data4,"Rajaji Nagar")
#plt.hist(data4.price_per_sqft,rwidth=0.6)
#plt.xlabel("Price Per Square Foor")
#plt.ylabel("frequency")
#plt.title("Distribution of Price per Square Foot")
#plt.show()
print(pd.Series(data4.bath.unique()).sort_values(ascending=True).to_numpy())
#print(data4[data4.bath>10])
#print(data4[data4.bath>data4.BHK+2])
data5=data4[data4.bath<data4.BHK+2]
print(data5.shape)
print(data5.head)
data6=data5.drop(['price_per_sqft'],axis=1)
print(data6.head())
dummies=pd.get_dummies(data6.location)
#print(dummies.head(5))
data6=pd.concat([data6,dummies.drop('others',axis=1)],axis=1)
#print(data6.head())
data7=data6.drop('location',axis='columns')
# print(data7.head())
# print(data7.shape)
X=data7.drop('price',axis='columns')
print(X.head())
y=data6.price
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
model=LinearRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
print(cross_val_score(LinearRegression(), X, y, cv=cv))
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
#def find_best_model_using_gridsearchcv(X,y):
#     algos = {
#         'decision_tree': {
#             'model': DecisionTreeRegressor(),
#             'params': {
#                 'criterion' : ['mse','friedman_mse'],
#                 'splitter': ['best','random']
#             }
#         },
#         # 'linear_regression' : {
#         #     'model': LinearRegression(),
#         #     'params': {
#         #         'normalize': [True, False]
#         #     }
#         # },
#         'lasso': {
#             'model': Lasso(),
#             'params': {
#                 'alpha': [1,2],
#                 'selection': ['random', 'cyclic']
#             }
#         },
#     }
#     scores = []
#     cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#     for i, j in algos.items():
#         gs =  GridSearchCV(j['model'], j['params'], cv=cv, return_train_score=False)
#         gs.fit(X,y)
#         scores.append({
#             'model': i,
#             'best_score': gs.best_score_,
#             'best_params': gs.best_params_
#         })
#         return pd.DataFrame(scores,columns=['model','best_score','best_params'])
# print(find_best_model_using_gridsearchcv(X,y))
# print(X.columns)
# print(len(X.columns))
# #print(np.zeros(244))
# print(np.where(X.columns=='5th Block Hbr Layout'))
def price_predict(location,sqft,bath,bhk):
    loc_index=np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index >=0:
        x[loc_index]=1
    return model.predict([x])[0]
print(price_predict('Whitefield',1000,2,2))
#print(price_predict('1st Phase JP Nagar',1000,2,3))
import pickle
import json
with open('Propery_rates_project.pickle','wb') as f:
    pickle.dump(model,f) 
columns={
    'data_columns':[col.lower() for col in X.columns]
}
with open('columns.json','w') as f:
    f.write(json.dumps(columns))
