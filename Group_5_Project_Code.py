
# coding: utf-8

# In[176]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# In[177]:


#Read csv file, displaying original number of features
df = pd.read_csv("Alzheimers_Proj_DataSet_Original.csv",low_memory=False)
print("Original number of columns",len(df.columns))


# In[178]:

#Dropping blank columns and unwanted columns
df=df.dropna(axis=1,how='all')
print("After dropping unwanted columns",len(df.columns))
df = df.drop(['LOGIDAY','LOGIMO','LOGIYR','CVDIMAG1','MOMOPARK','NACCIMCI','NACCNORM','NACCADGC','NACCID','NACCIDEM','NACCDIMP','NACCMAGE','NACCAGE','NACCFDYS','NACCWNDW','NACCAHTN','NACCHTNC','NACCACEI','NACCAAAS','NACCBETA','NACCCCBS','NACCDIUR','NACCVASD','NACCANGI','NACCLIPL','NACCNSD','NACCAC','NACCADEP','NACCAPSY','NACCAANX','NACCADMD','NACCPDMD','NACCAMD','NACCEMD','NACCEPMD','NACCDBMD','NACCBMI','NACCABBP','NACCLEVA','NACCLEVB','NACCC1','NACCZMMS','NACCZLMI','NACCZLMD','NACCZDFT','NACCZDFL','NACCZDBT','NACCZDBL','NACCZANI','NACCZVEG','NACCZTRA','NACCZTRB','NACCZWAI','NACCZBOS','NACCUDSD','NACCMCIT','NACCHIV','NACCMND','NACCPCA','NACCCANC','NACCPRAD','NACCPOAD','NACCLBD','NACCPRVD','NACCPOVD','NACCARD','NACCUND','NACCFTDD','NACCPPAD','NACCPSPD','NACCCBDD','NACCHNTD','NACCPRID','NACCMEDD','NACCMID','NACCDEPD','NACCPSYD','NACCDSD','NACCPDD','NACCSTKD','NACCHYDD','NACCTBID','NACCCNSD','NACCOTHD','NACCPRAM','NACCPOAM','NACCLBM','NACCPRVM','NACCPOVM','NACCARM','NACCUNM','NACCFTDM','NACCPPAM','NACCPSPM','NACCCBDM','NACCHNTM','NACCPRIM','NACCMEDM','NACCMIM','NACCDEPM','NACCPSYM','NACCDSM','NACCPDM','NACCSTKM','NACCHYDM','NACCTBIM','NACCCNSM','NACCOTHM','NACCMDSS','NACCAGEB','NACCMAGE','NACCNIHR','NACCAVST','NACCNVST','NACCDAYS','NACCSTAT','NACCNURS','NACCDIED','NACCAGED','NACCHDIS','NACCFAMH','NACCMOMD','NACCDADD','NACCMDSD','NACCIMCI','NACCIDEM','NACCNORM','NACCDIMP','NACCMAD','NACCAPOE','NACCNE4S','NACCADGC','NACCDAGE','NACCINT','NACCFTD','NACCPAFF','NACCMRI','NACCNMRI','NACCADNI','TRAILARR','TRAILALI','TRAILBRR','TRAILBLI','COGOTH2','COGOTH3','COGFLUC','BEVWELL','BEREM','FOCLSIGN','FOCLSYM','LOGIPREV','CVDCOG','SMOKYRS'],axis=1)


# In[181]:


print("Calculating correlation matrix and plotting it")
corr = df.corr().iloc[0:50,0:50]
import numpy as np 
from pandas import DataFrame
from matplotlib import pyplot
import seaborn as sns

a4_dims = (30, 30)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.set(font_scale=1)
sns.heatmap(corr, annot=True,ax=ax,annot_kws={"size": 1})

plt.show()
# In[182]:


#Dropping all descriptive(string) columns except for NACCID
print("Dropping all descriptive(string) columns except for NACCID")
df1=df.select_dtypes(include=['object'])
col_list=[x for x in df1.loc[:, df1.columns != 'NACCID']]
df=df.drop(col_list,axis=1)
print("After dropping string columns",len(df.columns))


# In[183]:


#Dropping NPIQINF(NPI Informant)as this feature is not really needed.
print("Dropping NPIQINF(NPI Informant)as this feature is not really needed.")
df=df.drop(['NPIQINF'],axis=1)
print("After dropping NPIQINF",len(df.columns))


# In[184]:


#Dropping columns which have the same value for all records
print("Dropping columns which have the same value for all records")
for col in df.columns:
    if len(df[col].unique()) == 1:
        df = df.drop(col,axis=1)
print("After dropping columns with same value",len(df.columns))


# In[186]:


print("Rounding of year columns to nearest year")
df2=df.filter(regex='^(?!LOGIYR).*YR$')
print("Imputing missing years by 0, this imputation was done in considering the constraints from the data dictionary")
df2=df2.round(0)
df2=df2.fillna(0)
df.update(df2)


# In[187]:


#Setting values of variables that depend on some other variables
print("Imputing the data with multi level contraints form the data dictionary and setting values of variables that depend on some other variables")
df.loc[df.CBSTROKE == 0, ['STROK1YR','STROK2YR','STROK3YR']] = 0
df.loc[df.CBTIA == 0, ['TIA1YR','TIA2YR','TIA3YR','TIA4YR']] = 0
df.loc[df.PD == 0, 'PDYR'] = 0
df.loc[df.PDOTHR == 0, 'PDOTHRYR'] = 0
df.loc[df.DEL == 0, 'DELSEV'] = 0
df.loc[df.HALL == 0, 'HALLSEV'] = 0
df.loc[df.DEPD == 0, 'DEPDSEV'] = 0
df.loc[df.ANX == 0, 'ANXSEV'] = 0
df.loc[df.ELAT == 0, 'ELATSEV'] = 0
df.loc[df.APA == 0, 'APASEV'] = 0
df.loc[df.DISN == 0, 'DISNSEV'] = 0
df.loc[df.IRR == 0, 'IRRSEV'] = 0
df.loc[df.MOT == 0, 'MOTSEV'] = 0
df.loc[df.NITE == 0, 'NITESEV'] = 0
df.loc[df.APP == 0, 'APPSEV'] = 0
cols =['DECAGE', 'COGMEM', 'COGJUDG', 'COGLANG', 'COGVIS', 'COGATTN', 'COGOTHR', 'COGFRST', 'COGMODE', 'BEAPATHY', 'BEDEP',
       'BEVHALL', 'BEAHALL', 'BEDEL', 'BEDISIN', 'BEIRRIT',
       'BEAGIT', 'BEPERCH',  'BEOTHR', 'BEFRST', 'BEMODE', 'MOGAIT',
       'MOFALLS', 'MOTREM', 'MOSLOW', 'MOFRST', 'MOMODE', 'COURSE',
       'FRSTCHG']
df.loc[df.DECCLIN == 0, cols] = 0
df.loc[df.VISCORR == 0, 'VISWCORR'] = 9
df.loc[df.HEARAID == 0, 'HEARWAID'] = 9
cols1=['SPEECH', 'FACEXP', 'TRESTFAC', 'TRESTRHD', 'TRESTLHD', 'TRESTRFT',
       'TRESTLFT', 'TRACTRHD', 'TRACTLHD', 'RIGDNECK', 'RIGDUPRT', 'RIGDUPLF',
       'RIGDLORT', 'RIGDLOLF', 'TAPSRT', 'TAPSLF', 'HANDMOVR', 'HANDMOVL',
       'HANDALTR', 'HANDALTL', 'LEGRT', 'LEGLF', 'ARISING', 'POSTURE', 'GAIT',
       'POSSTAB', 'BRADYKIN']
df.loc[df.PDNORMAL == 1, cols1] = 0


# In[188]:


#Rounding up all fractional values to integers as these are categorical features.
print("Rounding up all fractional values to integers as these are categorical features.")
df3=df[['CVHATT','CVAFIB','CVANGIO','CVBYPASS','CVPACE','CVCHF','CVOTHR','CBSTROKE','CBTIA','CBOTHR','PD','PDOTHR','SEIZURES','TRAUMBRF','TRAUMCHR','TRAUMEXT','NCOTHR','HYPERTEN','HYPERCHO','DIABETES','B12DEF','THYROID','INCONTU','INCONTF','DEP2YRS','DEPOTHR','ALCOHOL','TOBAC30','ABUSOTHR','PSYCDIS','HEARWAID']]
df3=df3.round(0)
df.update(df3)


# In[189]:


#Imputing all remaining categorical and ordinal features with 0
print("Imputing all remaining categorical and ordinal features with 0")
df.VISWCORR = pd.to_numeric(df.VISWCORR, errors="coerce")
list=df.columns[df.isnull().any()].tolist()
df[list] = df[list].fillna(value=0)
df.to_csv("clean.csv")
df=df.apply(preprocessing.LabelEncoder().fit_transform)


# In[190]:


X = df.loc[:,df.columns!='class']
y = df.loc[:,df.columns =='class']


# In[191]:


df_for_plots = pd.read_csv("Alzheimers_Proj_DataSet_Original.csv",low_memory=False)


# In[192]:


# check if there are continue valued columns in opur dataset spo that we can plot the 
print("check if there are continue valued columns in opur dataset spo that we can plot the ")
continuous_column = []
l = ['MMSE', 'FORMVER','MOMONSET','MOMAGE','VEG','ANIMALS','CDRSUM','MOMDAGE','DADONSET','DADAGE','DADDAGE','SIBS','SIBSDEM','SIB1ONS','SIB1AGE','SIB2ONS','DECAGE','SIB2AGE','SIB3ONS','SIB3AGE','SIB4ONS','SIB4AGE','SIB5ONS','SIB5AGE','SIB6ONS','SIB6AGE','KIDS','KIDSDEM','KID1ONS','KID1AGE','KID2ONS','KID2AGE','KID3ONS','KID3AGE','KID4ONS','KID4AGE','KID5ONS','KID5AGE','KID6ONS','KID6AGE','PMAS','PMAF','PMAPF','PMBS','PMBF','PMBPF','PMCS','PMCF','PMCPF','PMDS','HEIGHT','WEIGHT','BPSYS','BPDIAS','HRATE','TRAILB','TRAILA','WAIS','MEMUNITS']
for i in l: 
    for j in df_for_plots.columns:
        if i == j:
            continuous_column.append(j)


# In[193]:


df_for_plots[continuous_column].to_csv("continuous_column.csv")


# In[194]:


continuous_column.append("class")


# In[195]:


# Scatter plots for continuous data
print("Scatter plots for continuous data and the target variable")
import seaborn as sns; sns.set(style="ticks", color_codes=True)
g = sns.pairplot(df_for_plots[continuous_column])

plt.show()
# In[196]:


# Scaling/Normalization
print("Performing Scaling/Normalization of the dataset")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df[continuous_column])
df[continuous_column] = scaler.transform(df[continuous_column])


# In[222]:


# Probability Distribution of all the continuous values columns
print("Probability Distribution of all the continuous values columns")
df[continuous_column].hist()

plt.show()
# # Correlation based independence check for columns (drop columns > 0.3)

# In[202]:


print("Checking if there are any columns with correlation greater than 0.4")
corr = df.corr().iloc[:,:3]

l = corr['class']
for i in range(len(corr.index)):
    if abs(l[i])>0.3:
        print(corr.index[i], l[i])


# In[203]:


def model(X,y,clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    tre = clf
    tre.fit(X_train, y_train)
    y_test_pred = tre.predict(X_test)
    y_train_pred = tre.predict(X_train)
    #y_score_train = tre.predict(X_test)
    #accuracy = clf.score(X_test,y_test)
    AUC_ROC = roc_auc_score(y_test, y_test_pred)
    #print("Accuracy is: "+ str(accuracy))

    print('Training Accuracy is : %.2f' %accuracy_score(y_train, y_train_pred))
    print('Test Accuracy is : %.2f' %accuracy_score(y_test, y_test_pred))
    print("Area under the curve for ROC is: "+ str(AUC_ROC))


# In[204]:


# Distribution of class/label/target
print("visualizing the correlation of Distribution of class/label/target")
l = pd.DataFrame(l)
l.hist()
plt.show()

# In[205]:


def k_best_features_names(X_new):
    mask = X_new.get_support() #list of booleans
    new_features = [] # The list of your K best features
    feature_names = X.columns
    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)
    return new_features


# In[206]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import mutual_info_classif


# In[207]:


#CHI BASED
#----------------------------------------------------
print("Getting CHI BASED features")
X_chi_20 = SelectKBest(chi2, k=20).fit(X, y)
X_chi_20_feature_names =  k_best_features_names(X_chi_20)
X_chi_20 = SelectKBest(chi2, k=20).fit_transform(X, y)

X_chi_30 = SelectKBest(chi2, k=30).fit(X, y)
X_chi_30_feature_names =  k_best_features_names(X_chi_30)
X_chi_30 = SelectKBest(chi2, k=30).fit_transform(X, y)


# In[208]:


#f_classif
#----------------------------------------------------
print("getting f_classif based featureset")
X_f_classif_20 = SelectKBest(f_classif, k=20).fit(X, y)
X_f_classif_20_feature_names =  k_best_features_names(X_f_classif_20)
X_f_classif_20 = SelectKBest(f_classif, k=20).fit_transform(X, y)

X_f_classif_30 = SelectKBest(f_classif, k=30).fit(X, y)
X_f_classif_30_feature_names =  k_best_features_names(X_f_classif_30)
X_f_classif_30 = SelectKBest(f_classif, k=30).fit_transform(X, y)


# In[209]:


#mutual_info_classif
#----------------------------------------------------
print("getting mutual_info_classif based featureset")
X_mutual_info_classif_20 = SelectKBest(mutual_info_classif, k=20).fit(X, y)
X_mutual_info_classif_20_feature_names =  k_best_features_names(X_mutual_info_classif_20)
X_mutual_info_classif_20 = SelectKBest(mutual_info_classif, k=20).fit_transform(X, y)

X_mutual_info_classif_30 = SelectKBest(mutual_info_classif, k=30).fit(X, y)
X_mutual_info_classif_30_feature_names =  k_best_features_names(X_mutual_info_classif_30)
X_mutual_info_classif_30 = SelectKBest(mutual_info_classif, k=30).fit_transform(X, y)


# In[210]:


#PCA
#----------------------------------------------------
print("getting PCA based featureset")

from sklearn.decomposition import PCA
pca = PCA(n_components=20)
X_pca_20 = pca.fit_transform(X)
print(X_pca_20.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=30)
X_pca_30 = pca.fit_transform(X)
print(X_pca_30.shape)


# In[211]:


#check if final features are independent or not
print("Using Heatmap check if final features are independent or not.")
X_chi_20 = pd.DataFrame(X_mutual_info_classif_20,columns=X_mutual_info_classif_20_feature_names)
corr = X_chi_20.corr()
import numpy as np 
from pandas import DataFrame
from matplotlib import pyplot
import seaborn as sns

a4_dims = (30, 30)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.set(font_scale=1)
sns.heatmap(corr, annot=True,ax=ax,annot_kws={"size": 20})
plt.show()
# In[217]:


print("Model building starts:")

from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

lr = linear_model.LogisticRegression(penalty='l2', C=0.1)
dt = DecisionTreeClassifier()
svm = svm.SVC()
nb = naive_bayes.BernoulliNB()

print("\n\n__________Linear Regression______________")

model(X_chi_20,y,lr)
model(X_chi_30,y,lr)
model(X_f_classif_20,y,lr)
model(X_f_classif_30,y,lr)
model(X_mutual_info_classif_20,y,lr)
model(X_mutual_info_classif_30,y,lr)
model(X_pca_20,y,lr)
model(X_pca_30,y,lr)

print("\n\n__________Decision Trees______________")


model(X_chi_20,y,dt)
model(X_chi_30,y,dt)
model(X_f_classif_20,y,dt)
model(X_f_classif_30,y,dt)
model(X_mutual_info_classif_20,y,dt)
model(X_mutual_info_classif_30,y,dt)
model(X_pca_20,y,dt)
model(X_pca_30,y,dt)


print("\n\n__________SVM______________")


model(X_chi_20,y,svm)
model(X_chi_30,y,svm)
model(X_f_classif_20,y,svm)
model(X_f_classif_30,y,svm)
model(X_mutual_info_classif_20,y,svm)
model(X_mutual_info_classif_30,y,svm)
model(X_pca_20,y,svm)
model(X_pca_30,y,svm)

print("\n\n__________Naive Bayes______________")


model(X_chi_20,y,nb)
model(X_chi_30,y,nb)
model(X_f_classif_20,y,nb)
model(X_f_classif_30,y,nb)
model(X_mutual_info_classif_20,y,nb)
model(X_mutual_info_classif_30,y,nb)
model(X_pca_20,y,nb)
model(X_pca_30,y,nb)


# In[221]:


print("Plotting the ROC curve for the best fit model")

X = X_f_classif_30
clf = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
tre = clf
tre.fit(X_train, y_train)
y_test_pred = tre.predict(X_test)
y_train_pred = tre.predict(X_train)
AUC_ROC = roc_auc_score(y_test, y_test_pred)
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Decision Tree using f_class_if, AUC="+str(auc))
plt.legend(loc=4)
plt.show()


