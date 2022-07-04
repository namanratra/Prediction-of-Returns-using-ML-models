import pandas as pd 
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
import os

data = pd.read_csv('ReturnsPrediction.csv')
data.info()

def datahandling(data):


    # converting to datetime object
    data['dateOfBirth'] = pd.to_datetime(data['dateOfBirth'], errors ='coerce')
    data['orderDate'] = pd.to_datetime(data['orderDate'])
    data['deliveryDate'] = pd.to_datetime(data['deliveryDate'])
    data['creationDate'] = pd.to_datetime(data['creationDate'])
    data['age'] =  (data['orderDate'] - data['dateOfBirth'] ).astype('<m8[Y]') # extracting feature 'age'
    print(data.info())

    # groupby state to replace the nan values with state median value
    df = data[['customerID','age','state']]
    df = df.drop_duplicates(subset=['customerID'])
    d1 = df.groupby('state')['age'].median ()
    d1 = d1.reset_index()
    d2 = data.set_index("state").age.fillna(d1.set_index("state").age).reset_index()
    data['age'] = d2['age']

    # new feature deliverytime and also replace nan with median 
    data['deliveryTime'] = data['deliveryDate'] - data['orderDate']
    data['deliveryTime'] = data['deliveryTime'].apply(lambda x: x.days)
    data['deliveryTime'].fillna(data['deliveryTime'].median(), inplace = True)

    # day of when order is placed, (0-6)
    data['orderDay']  = data['orderDate'].dt.dayofweek
    data.head()

    # new feature ordermonthday ( 1-31 )
    data['ordermonthday'] = data['orderDate'].dt.day

    # creating feature which captures orderhistory
    data['timeAsCustomer'] = data.loc[:,'orderDate'] - data.loc[:,'creationDate']
    sorted_ID_Time = data.sort_values(['customerID', 'timeAsCustomer'], ascending=[True, True])
    sorted_ID_Time['timeAsCustomer'] = (sorted_ID_Time['timeAsCustomer'] / np.timedelta64(1, 'D')).astype(int)
    sorted_ID_Time.reset_index(drop=True, inplace=True)
    sorted_ID_Time.info()


    order_number = []
    counter = 1
    number = 1
    for r in range(0, len(sorted_ID_Time) - 1):
        if (sorted_ID_Time.iloc[r,8] == sorted_ID_Time.iloc[(r + 1),8]) and (sorted_ID_Time.iloc[r,18] == sorted_ID_Time.iloc[(r + 1),18]) and r <= len(sorted_ID_Time) - 2:
            counter += 1
        elif (sorted_ID_Time.iloc[r,8] == sorted_ID_Time.iloc[(r + 1),8]) and r <= len(sorted_ID_Time) - 2:
            for i in range(0, counter):
                order_number.append(number)
            counter = 1
            number += 1
        else:
            for i in range(0, counter):
                order_number.append(number)
            counter = 1
            number = 1
            

    for i in range(0, counter):   # for adding last element in dataframe
            order_number.append(number)
            
    

    sorted_ID_Time['Orderhistory'] = order_number
    data = sorted_ID_Time.sort_values(['orderItemID'], ascending=[True])
    data.reset_index(drop=True, inplace=True)
    data.info()

    # replacing nan values in 'color' feature
    data['color'].fillna('No Color', inplace = True)

    # extracting total number of articles ordered by customer in a single order 
    # 'orderDate-customerID' is same means its same order
    
    
    data['item'] = 1  #dummy feature
    data['OrderCustomerItem'] = data['customerID'].map(str) + data['orderDate'].map(str) + data['itemID'].map(str) #dummy feature
    data['OrderCustomer'] = data['customerID'].map(str) + data['orderDate'].map(str)  #dummy feature

    # gives total number of articles in cart within an order
    data['ordertotalcount'] = data.groupby('OrderCustomer')['item'].cumsum()   
    d1 = data.groupby('OrderCustomer')['ordertotalcount'].max()
    d1 = d1.reset_index()
    d1= pd.DataFrame(d1)
    data['ordertotalcount'] = np.nan
    d2 = data.set_index("OrderCustomer").ordertotalcount.fillna(d1.set_index("OrderCustomer").ordertotalcount).reset_index()
    data['ordertotalcount'] = d2['ordertotalcount']
    data.head()

    # gives total number of articles of simialr ItemID within order
    data['orderItemcount'] = data.groupby('OrderCustomerItem')['item'].cumsum()
    d1 = data.groupby('OrderCustomerItem')['orderItemcount'].max()
    d1 = d1.reset_index()
    d1= pd.DataFrame(d1)
    data['orderItemcount'] = np.nan
    d2 = data.set_index("OrderCustomerItem").orderItemcount.fillna(d1.set_index("OrderCustomerItem").orderItemcount).reset_index()
    data['orderItemcount'] = d2['orderItemcount']
    data.head()

    # within one order if itemID repeats it means its is size or color of diff types 
    #incorporatng feature for the count of same
    
    # item color count within an order
    d1 = data.groupby('OrderCustomerItem')['color'].nunique()
    d1 = d1.reset_index()
    d1= pd.DataFrame(d1)
    data['orderItemcolorcount'] = np.nan
    d2 = data.set_index("OrderCustomerItem").orderItemcolorcount.fillna(d1.set_index("OrderCustomerItem").color).reset_index()
    
    data['orderItemcolorcount'] = d2['orderItemcolorcount']
    data.head()

    # item color count within an order
    d1 = data.groupby('OrderCustomerItem')['size'].nunique()
    d1 = d1.reset_index()
    d1= pd.DataFrame(d1)
    d1.rename(columns = {'size': 'sizecount'}, inplace= True)
    d1.head()
    data['orderItemsizecount'] = np.nan
    d2 = data.set_index("OrderCustomerItem").orderItemsizecount.fillna(d1.set_index("OrderCustomerItem").sizecount).reset_index()
    data['orderItemsizecount'] = d2['orderItemsizecount']
    data.head()

    # boxplot for age
    plt.figure(figsize=(10, 10), dpi=80)
    plt.boxplot(x=data['age'])
    plt.show()

    # dropping outliers in age
    data.drop( data[ (data['age']<18) | (data['age'] >100) ].index, inplace =True)
    data.info()

    # dropping outliers in deliverytime
    # delivery time cannot be negative
    data.drop( data[ data['deliveryTime'] < 0 ].index, inplace = True)
    # boxplot for age
    plt.figure(figsize=(8, 8), dpi=80)
    plt.boxplot(x=data['deliveryTime'])
    plt.show()

    data.drop( data[ data['deliveryTime'] > 30 ].index, inplace = True)
    data.reset_index(drop=True, inplace=True)
    data.info()

    # deliverytime plot (values range from 0 - 30 )
    df_filtered = data.loc[data['deliveryTime']<15]
    plt.figure(figsize=(60, 60), dpi=80)
    sns.catplot(x="deliveryTime", kind="count", palette="ch:.25", data=df_filtered)
    plt.show()

    plt.figure(figsize=(60, 60), dpi=80)
    sns.catplot(x="deliveryTime", kind="count", hue = "returnShipment", palette="ch:.25", data=df_filtered)
    plt.show()

    # taking care of 'size' categorical labels for similar values
    print(data['size'].unique())
    data.loc[data['size']  == 'm', 'size'] = 'M'
    data.loc[data['size']  == 'l', 'size'] = 'L'
    data.loc[data['size']  == 'xl', 'size'] = 'XL'
    data.loc[data['size']  == 'xxl', 'size'] = 'XXL'
    data.loc[data['size']  == 'xxxl', 'size'] = 'XXXL'

    # count plot of size
    plt.figure(figsize=(30, 30), dpi=80)
    X= data
    sns.catplot(x="size", kind="count", palette="ch:.25", data=X)
    plt.show()
    
    # adding sizetype based on count
    d1 = data['size'].value_counts()
    d1.reset_index()
    d1= pd.DataFrame(d1)
    d1.rename(columns = {'size':'count'}, inplace = True)

    d1['sizetype'] = d1['count'].apply(lambda x: 'average' if x > 30000
                       else ('almost average' if x > 10000 else ('rare' if x > 1000 else 'Veryrare')))
    data['sizetype'] = np.nan
    d2 = data.set_index("size").sizetype.fillna(d1.sizetype).reset_index()
    data['sizetype'] = d2['sizetype']

    X= data
    plt.figure(figsize=(30, 30), dpi=80)
    sns.catplot(x="sizetype", kind="count",hue = "returnShipment", palette="ch:.25", data=X)
    plt.show()

    # adding sizetype based on count
    d1 = data['color'].value_counts()
    d1.reset_index()
    d1= pd.DataFrame(d1)
    d1.rename(columns = {'color':'count'}, inplace = True)

    d1['colortype'] = d1['count'].apply(lambda x: 'High' if x > 40000
                        else ('Moderate' if x > 10000 else ('Low' if x > 1000 else 'Verylow')))
    data['colortype'] = np.nan
    d2 = data.set_index("color").colortype.fillna(d1.colortype).reset_index()
    data['colortype'] = d2['colortype']

    X = data
    plt.figure(figsize=(30, 30), dpi=80)
    sns.catplot(x="colortype", kind="count",hue = "returnShipment", palette="ch:.25", data=X)
    plt.show()

    # dropping non relvant columns as they have been already used for feature extraction
    data = data.drop(['orderDate','deliveryDate','creationDate','dateOfBirth','item','OrderCustomerItem','OrderCustomer','timeAsCustomer'], axis =1)
    data.info()
    return data


def applying_kmeans(X):

    # building data set to be used for product classification item Id count is 2997...feature sapce is going huge

    # getting mean price of each itemID (there could be diff due to size and color)

    d1 = X[['itemID','price','manufacturerID']]
    d1['itemid + price'] = X['itemID'].map(str) + X['price'].map(str)
    d1 = d1.drop_duplicates(subset=['itemid + price'])
    d1 = d1.groupby(['itemID'])['price'].mean()

    d1 = d1.reset_index()
    d1.reset_index(drop=True, inplace=True)
    d1.info()

    # getting size count for each ItemID

    d2 = X.groupby(['itemID'])['size'].nunique()
    d2 = d2.reset_index()
    d2.reset_index(drop=True, inplace=True)
    d2.info()

    d2['meanprice'] = d1['price']   # appending meanprice info from prevous dataframe
    d2.rename(columns = {'size':'sizecount'}, inplace = True) 

    # getting color count for each ItemID

    d1 = X.groupby(['itemID'])['color'].nunique()
    d1 = d1.reset_index()
    d2['colorcount'] = d1['color']  #appending colorcount info

    # getting manufacturerID for each ItemID
    d1 = X[['itemID','manufacturerID']]
    d1 = d1.drop_duplicates(subset=['itemID'])
    d1.reset_index(drop=True, inplace=True)
    d1 = d1.sort_values(['itemID'], ascending=[True])
    d1.reset_index()
    d2['manufacturerID'] = d1['manufacturerID'] # appending manufacturerID info
    d2.info()

    cTransform = LabelBinarizer()
    # label encoding of categorical data
    manufacturerID_enc = cTransform.fit_transform(d2['manufacturerID'])
    manufacturerID_enc = np.delete(manufacturerID_enc, -1, 1) #to get rid off multi-collinearity

    d2_ordinal = np.array(d2[['sizecount','colorcount']])

    price_ = np.array(d2['meanprice'])
    price_ = price_.reshape(-1, 1)

    product_enc = np.concatenate((manufacturerID_enc,d2_ordinal,price_),axis =1)

    from sklearn.cluster import KMeans

    #checking for elbow
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(product_enc)
        distortions.append(km.inertia_)

    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()
    
    # running kmeans with clusters = 6

    km = KMeans(n_clusters =6, init ='k-means++', n_init =30, max_iter = 500,
                tol = 1e-05, random_state= 0)
    y_km = km.fit_predict(product_enc)



    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)
    n = 20
    cmap = get_cmap(n)
    plt.figure(figsize=(30, 30), dpi=80)


    for i in range(0,6):

        plt.scatter(
        product_enc[y_km == i, 1], product_enc[y_km == i, 2],
        s=50, c=cmap(i),
        marker='o', edgecolor='black',
        label=i
        )

    product_cluster = pd.DataFrame(y_km, columns = ['cluster'])
    product_cluster.to_csv('D:\product_cluster.csv')

    # appending cluster value to data
    d2['cluster'] = product_cluster['cluster']
    d2.info()

    X['cluster'] = np.nan
    d1 = X.set_index("itemID").cluster.fillna(d2.set_index("itemID").cluster).reset_index()
    X['cluster'] = d1['cluster']

    import seaborn as sns
    plt.figure(figsize=(30, 30), dpi=80)
    sns.catplot(x="cluster", kind="count", hue = "returnShipment", palette="ch:.25", data=X)
    plt.show()

    return X

def chi_test(X):
    X = X.astype(str)
    
    oe = OrdinalEncoder()
    X = oe.fit_transform(X)


    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)


    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)

    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()

Data_ = datahandling(data)
Data_ = pd.DataFrame(Data_)
Data_.info()
print("beforekmeans")
Data_ = applying_kmeans(Data_)
print("afterkemans")
Data_.info()

data = Data_

data['deliveryTime'] = data['deliveryTime'].apply(lambda x: 'HighTime' if x > 7 else x)
y=data['returnShipment']
data.info()

data = data.drop(['orderItemID','returnShipment','itemID','customerID','orderDay','orderItemID','ordermonthday'], axis = 1)
data.info()
X= data.drop(['price','age'],  axis = 1)
chi_test(X)

data = data.drop(['state','salutation','Orderhistory','orderItemcolorcount','sizetype','colortype'], axis = 1)
data.info()
X= data
#### starting models

# encoding categorical data

cTransform = LabelBinarizer()
x_size = cTransform.fit_transform(X['size'])
x_size = np.delete(x_size, -1, 1)
x_color = cTransform.fit_transform(X['color'])
x_color = np.delete(x_color, -1, 1)
x_manufacture = cTransform.fit_transform(X['manufacturerID'])
x_manufacture = np.delete(x_manufacture, -1, 1)
x_cluster = cTransform.fit_transform(X['cluster'])
x_cluster = np.delete(x_cluster, -1, 1)
X['deliveryTime'] = X['deliveryTime'].astype(str)
x_deliverytime = cTransform.fit_transform(X['deliveryTime'])
x_deliverytime = np.delete(x_deliverytime, -1, 1)

X_enc = np.concatenate((x_size,x_color,x_manufacture,x_cluster,x_deliverytime), axis=1)

# encoding ordinal data
X_ordinal = X[['orderItemcount','ordertotalcount','orderItemsizecount','age']]
oe = OrdinalEncoder()
X_ordinal = oe.fit_transform(X_ordinal)

#normalizing contonous variable
x_maxmin = np.array(X['price'])
x_maxmin = x_maxmin.reshape(-1, 1)

nscaler = MinMaxScaler()
x_maxmin = nscaler.fit_transform(x_maxmin)

X_enc = np.concatenate((X_enc,x_maxmin,X_ordinal),axis =1)

# running PCA


pca = PCA(n_components=385)
pca.fit(X_enc)

var = pca.explained_variance_ratio_
cum_var = np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)

# visualizing PCA

plt.plot(cum_var)
plt.show()

from sklearn.decomposition import PCA
# compressing data to 30 features
# k-means also acts as data compressor therefore running with 3206 features can also be done.

pca = PCA(n_components=30)
X_enc = pca.fit_transform(X_enc)

# data split

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_enc, y, test_size=0.25, random_state=1)

# running knn model


accuracies = []


for i in range(2,21):
    knnmodel = KNeighborsClassifier(n_neighbors=i)
    knnmodel.fit(X_train_pca, y_train)
    y_test_pred = knnmodel.predict(X_test_pca)
    accte = accuracy_score(y_test, y_test_pred)
    print(i, accte)
    accuracies.append(accte)

plt.plot(range(2, 21), accuracies)
plt.xlim(1,20)
plt.xticks(range(2, 21))
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies')
plt.show()




k_accr = []

# running knn with n_neighbors=19

knnmodel = KNeighborsClassifier(n_neighbors=19)
knnmodel.fit(X_train_pca, y_train)
y_train_pred = knnmodel.predict(X_train_pca)

acctr = accuracy_score(y_train, y_train_pred)
k_accr.append((i,acctr))
print("Accurray Training:",i, acctr)
cmtr = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix Training:\n", cmtr)

y_test_pred = knnmodel.predict(X_test_pca)
cmte = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(y_test, y_test_pred)
print("Accurray Test:", accte)


# decision tree model for max depth search

accuracies = np.zeros((2,15), float)

for k in range(0, 15):
    etmodel = DecisionTreeClassifier(criterion='entropy',random_state=0, max_depth=k+1)
    etmodel.fit(X_train_pca, y_train)
    y_train_pred = etmodel.predict(X_train_pca)
    acctr = accuracy_score(y_train, y_train_pred)
    accuracies[0,k] = acctr
    y_test_pred = etmodel.predict(X_test_pca)
    accte = accuracy_score(y_test, y_test_pred)
    accuracies[1,k] = accte

plt.plot(range(1, 16), accuracies[0,:])
plt.plot(range(1, 16), accuracies[1,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies (Entropy)')
plt.show()

# decision tree model with maxdepth = 9

etmodel = DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=9)
etmodel.fit(X_train_pca, y_train)
y_train_pred = etmodel.predict(X_train_pca)
cmtr = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(y_train, y_train_pred)
print("Accurray Training:", acctr)

y_test_pred = etmodel.predict(X_test_pca)
cmte = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix Testing:\n", cmte)

accte = accuracy_score(y_test, y_test_pred)
print("Accurray Test:", accte)


#random forest model for depth search

accuracies = np.zeros((2,20), float)

for k in range(0, 20):
    # etmodel = DecisionTreeClassifier(criterion='entropy',random_state=0, max_depth=k+1)
    rfmodel = RandomForestClassifier(random_state=0, max_depth = k+1)

    rfmodel.fit(X_train_pca, y_train)
    y_train_pred = rfmodel.predict(X_train_pca)
    acctr = accuracy_score(y_train, y_train_pred)
    accuracies[0,k] = acctr
    y_test_pred = rfmodel.predict(X_test_pca)
    accte = accuracy_score(y_test, y_test_pred)
    accuracies[1,k] = accte

plt.plot(range(1, 21), accuracies[0,:])
plt.plot(range(1, 21), accuracies[1,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Random forest')
plt.show()

# search for nestimator fpr Random forest

accuracies = np.zeros((2,20), float)
ntrees = (np.arange(20)+1)*10
for k in range(0, 20):
    # etmodel = DecisionTreeClassifier(criterion='entropy',random_state=0, max_depth=k+1)
    rfmodel = RandomForestClassifier(random_state=0, n_estimators=ntrees[k])

    rfmodel.fit(X_train_pca, y_train)
    y_train_pred = rfmodel.predict(X_train_pca)
    acctr = accuracy_score(y_train, y_train_pred)
    accuracies[0,k] = acctr
    y_test_pred = rfmodel.predict(X_test_pca)
    accte = accuracy_score(y_test, y_test_pred)
    accuracies[1,k] = accte

plt.plot(ntrees, accuracies[0,:])
plt.plot(ntrees, accuracies[1,:])
plt.xticks(ntrees, rotation=90)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest')
plt.show()

# random Forest grid search


mdepth = np.array([12,13,14,15,16])
ntrees = np.array([160,170,180,190,200])
row = 0
accuracies = np.zeros((4,25), float)

for k in range(0, 5):
    for l in range(0, 5):
        rfmodel = RandomForestClassifier(random_state=0,max_depth=mdepth[k], n_estimators=ntrees[l])
        rfmodel.fit(X_train_pca, y_train)
        y_train_pred = rfmodel.predict(X_train_pca)
        acctr = accuracy_score(y_train, y_train_pred)
        accuracies[2,row] = acctr
        y_test_pred = rfmodel.predict(X_test_pca)
        accte = accuracy_score(y_test, y_test_pred)
        print(accte)
        accuracies[3,row] = accte
        accuracies[0,row] = mdepth[k]
        accuracies[1,row] = ntrees[l]
        row = row + 1


headers = ["Max_Depth", "n_Estimators", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain",floatfmt=".3f")
print("\n",table)


# adaboost classfier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15), n_estimators=180, algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X_train_pca, y_train)



ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9), 
                             n_estimators=180, algorithm="SAMME.R", 
                             learning_rate=0.5)
ada_clf.fit(X_train_pca, y_train)

y_train_pred = ada_clf.predict(X_train_pca)
cmtr = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(y_train, y_train_pred)
print("Accurray Training:", acctr)

y_test_pred = ada_clf.predict(X_test_pca)
cmte = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix Testing:\n", cmte)

accte = accuracy_score(y_test, y_test_pred)
print("Accurray Test:", accte)

# running gradient boosting

gbmodel = GradientBoostingClassifier(random_state=0,max_depth=12, 
                                     learning_rate=0.13)
gbmodel.fit(X_train_pca, y_train)

y_train_pred = gbmodel.predict(X_train_pca)
cmtr = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(y_train, y_train_pred)
print("Accurray Training:", acctr)
y_test_pred = gbmodel.predict(X_test_pca)
cmte = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(y_test, y_test_pred)
print("Accurray Test:", accte)

accuracies = np.zeros((4,21*10), float)
lr = np.linspace(0, 0.4, 21)
lr[0] = 0.01
row = 0
for k in range(0, 10):
    for l in range(0, 21):
        gbmodel = GradientBoostingClassifier(random_state=0,
        max_depth=k+1, learning_rate=lr[l])
        gbmodel.fit(X_train_pca, y_train)
        Y_train_pred = gbmodel.predict(X_train_pca)
        acctr = accuracy_score(y_train, y_train_pred)
        accuracies[2,row] = acctr
        y_test_pred = gbmodel.predict(X_test_pca)
        accte = accuracy_score(y_test, y_test_pred)
        accuracies[3,row] = accte
        accuracies[0,row] = k+1
        accuracies[1,row] = lr[l]
        row = row + 1
            

print(accuracies[3].max())
maxi = np.array(np.where(accuracies==accuracies[3].max()))
print(maxi[0,:], maxi[1,:])
headers = ["Max_depth", "Learning_rate", "acctr", "accte"]
table = tabulate(accuracies[:,maxi[1,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n",table)

#running logistic regression

lrmodel = LogisticRegression()
lrmodel.fit(X_train_pca, y_train)
y_train_pred = lrmodel.predict(X_train_pca)
y_test_pred = lrmodel.predict(X_test_pca)
cmtr = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix Training:\n", cmtr) 
acctr = accuracy_score(y_train, y_train_pred)
print("Accurray Training:", acctr) 
cmte = confusion_matrix(y_test, y_test_pred) 
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(y_test, y_test_pred)
print("Accurray Test:", accte)

# running neural network

accuracies = np.zeros((2,20), float)
for k in range(0, 20):
    nnetmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(k+1,), random_state=0)
    nnetmodel.fit(X_train_pca, y_train)
    y_train_pred = nnetmodel.predict(X_train_pca)
    acctr = accuracy_score(y_train, y_train_pred)
    accuracies[0,k] = acctr
    y_test_pred = nnetmodel.predict(X_test_pca)
    accte = accuracy_score(y_test, y_test_pred)
    accuracies[1,k] = accte
plt.plot(range(1, 21), accuracies[0,:])
plt.plot(range(1, 21), accuracies[1,:])
plt.xlim(1,20)
plt.xticks(range(1, 21))
plt.xlabel('Hidden Neurons')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies (Neural Network)')
plt.show()

# running cross validation


X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_enc, y, test_size = 0.25)
report = pd.DataFrame(columns=['Model','Mean Acc. Training','Standard Deviation','Acc. Test'])


rfmodel = RandomForestClassifier(random_state=0, max_depth = 15, n_estimators= 180)

accuracies = cross_val_score(rfmodel, X_train_pca, y_train,
                              scoring='accuracy', cv = 10)

print(accuracies)
acc_mean = accuracies.mean()
acc_std = accuracies.std()
print("accuracy mean in Cross Validation: ", acc_mean)

print("accuracy std in Cross Validation: ",acc_std)

# ensemble stacking

rfmodel = RandomForestClassifier(random_state=0)
lr_ensemble = LogisticRegression()
knnmodel = KNeighborsClassifier(n_neighbors=10)

stens1model = StackingClassifier(classifiers=[knnmodel,rfmodel], use_probas=True,average_probas=False,
                                  meta_classifier=lr_ensemble)

accuracies = cross_val_score(stens1model, X_train_pca, y_train, scoring='accuracy', cv=3)

stens1model.fit(X_train_pca, y_train)
y_test_pred = stens1model.predict(X_test_pca)
accte = accuracy_score(y_test, y_test_pred)
report.loc[len(report)] = ['Stacking Ensemble', accuracies.mean(),
accuracies.std(), accte]
print(report.loc[len(report)-1])