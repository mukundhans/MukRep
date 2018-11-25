import pandas as pd
import numpy as np
import sklearn.preprocessing as pre
import sklearn.linear_model as linmod
import sklearn.neighbors as nb
import sklearn.tree as tree
import sklearn.ensemble as ens
import sklearn.cross_validation as cv
import sklearn.metrics as met
import sklearn.neighbors as n
import catboost as cb
import sklearn.svm as svm
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.losses import categorical_crossentropy

df_train = pd.read_csv("/home/mukundhan/02_Tech/Python/Inceptez_Hackathon/traindata.csv")
df_test = pd.read_csv("/home/mukundhan/02_Tech/Python/Inceptez_Hackathon/testdata.csv")
df_test_results = pd.read_csv("/home/mukundhan/02_Tech/Python/Inceptez_Hackathon/Result Evaluation.csv")

#Introduce new classifier column and drop target regression column
for r in range(0, df_train.shape[0]):
    if df_train.loc[r, "TARGET"] > 0:
        df_train.loc[r, "TARGET_C"] = 1
    else:
        df_train.loc[r, "TARGET_C"] = 0
df_train.drop("TARGET", axis="columns", inplace=True)

for r in range(0, df_test_results.shape[0]):
    if df_test_results.loc[r, "TARGET"] > 0:
        df_test_results.loc[r, "TARGET_C"] = 1
    else:
        df_test_results.loc[r, "TARGET_C"] = 0
df_test_results.drop("TARGET", axis="columns", inplace=True)

ycol = "TARGET_C"
id_cols = "ID"
drop_cols = ["ind_var13_medio_0", "ind_var18_0", "ind_var27_0", "ind_var46_0", "ind_var46", "num_var13_medio", "num_var34_0", "num_var34", 
         "num_var41", "num_var46_0", "saldo_var18", "saldo_var28", "saldo_var41", "delta_imp_reemb_var33_1y3", "delta_num_reemb_var17_1y3", 
         "imp_amort_var18_hace3", "imp_reemb_var17_hace3", "imp_reemb_var17_ult1", "imp_reemb_var33_hace3", "imp_reemb_var33_ult1", 
         "imp_trasp_var17_in_hace3", "imp_trasp_var17_out_hace3", "imp_trasp_var17_out_ult1", "imp_trasp_var33_out_ult1", 
         "num_var2_0_ult1", "num_var2_ult1", "num_trasp_var17_in_hace3", "num_trasp_var17_out_hace3", "num_trasp_var17_out_ult1", 
         "num_trasp_var33_out_hace3", "saldo_var2_ult1", "saldo_medio_var13_medio_hace3", "saldo_medio_var13_medio_ult1",
         "var15", "imp_sal_var16_ult1", "ind_var12_0", "ind_var17_0", "ind_var17", "ind_var39_0", "ind_var41_0", "num_op_var41_ult1",
         "num_op_var41_ult3", "num_op_var39_ult3", "num_var30", "num_var33_0", "num_var33_0", "num_var42_0", "num_var42",
         "ind_var43_emit_ult1", "ind_var43_recib_ult1", "num_var7_emit_ult1", "num_med_var45_ult3", "num_meses_var39_vig_ult3",
         "num_op_var40_efect_ult1", "num_op_var41_comer_ult3", "num_trasp_var17_in_ult1", "num_trasp_var33_in_hace3", 
         "num_trasp_var33_in_ult1", "num_venta_var44_hace3", "num_var45_ult3", "num_var45_ult1", "saldo_medio_var5_hace3", 
         "saldo_medio_var5_ult1", "saldo_medio_var8_ult1", "saldo_medio_var12_hace3", "saldo_medio_var13_corto_hace2",
         "saldo_medio_var13_corto_hace3", "saldo_medio_var29_hace2", "saldo_medio_var29_hace3", "saldo_medio_var29_ult1",
         "num_var12_0"
         ]
xcols = [x for x in df_train.columns if x not in [ycol, id_cols, drop_cols]]

# Remove rows where var3 < 0
df_train = df_train[df_train.var3 > 0]

# Combine train & test
df_train["src"] = "train"
df_test["src"] = "test" 
df = pd.concat([df_train, df_test], axis=0, sort=False)

#df.drop(drop_cols, axis="columns", inplace=True)
        
# Min-Max Scaling
rs = pre.MinMaxScaler()
df[xcols] = pd.DataFrame(data=rs.fit_transform(df[xcols]))

# Check correlation
for c in df.columns:
    corr = abs(df.loc[:, c].corr(df.loc[:, "TARGET_C"]))
    print(c, "= ", corr)
    #if 0.0001 <= corr <= 0.009:
        #TODO: Drop columns whose corr falls in this range
    

# Fit algorithm
dfx_train = df.loc[df.src == "train", xcols]
dfy_train = df.loc[df.src == "train", ycol]
dfx_test = df.loc[df.src == "test", xcols]
print(dfx_train.shape)

x_train, x_test, y_train, y_test = cv.train_test_split(dfx_train, dfy_train, test_size=0.3, random_state=0)

# ------------------------------------ Logistic Regression -------------------------------------------
print("---------------------------------------------------------------------")
print("Logictic Regression ...")
logreg = linmod.LogisticRegression()
logreg.fit(x_train, y_train)

# Predict within train
print("*** TRAIN- Accuracy metrics...")
y_pred_train = logreg.predict(x_test)
cm = met.confusion_matrix(y_test, y_pred_train)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(y_test, y_pred_train))

# Predict within test
print("*** TEST- Accuracy metrics...")
y_pred_test = logreg.predict(dfx_test)
cm = met.confusion_matrix(df_test_results.TARGET_C, y_pred_test)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(df_test_results.TARGET_C, y_pred_test))

# ------------------------------ Random Forrest ----------------------------------
print("---------------------------------------------------------------------")
print("Random Forrest ...")
rf = ens.RandomForestClassifier(n_estimators=250, max_features="auto", max_depth = 50)
rf.fit(dfx_train, dfy_train)

# Predict within train
print("*** TRAIN- Accuracy metrics...")
y_pred_train = rf.predict(x_test)
cm = met.confusion_matrix(y_test, y_pred_train)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(y_test, y_pred_train))

# Predict within test
print("*** TEST- Accuracy metrics...")
y_pred_test = rf.predict(dfx_test)
cm = met.confusion_matrix(df_test_results.TARGET_C, y_pred_test)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(df_test_results.TARGET_C, y_pred_test))

# ------------------------------ KNN ----------------------------------
print("---------------------------------------------------------------------")
print("KNN ...")
knn = n.KNeighborsClassifier(n_neighbors=100, metric = 'manhattan', weights = 'uniform')
knn.fit(dfx_train, dfy_train)

# Predict within train
print("*** TRAIN- Accuracy metrics...")
y_pred_train = knn.predict(x_test)
cm = met.confusion_matrix(y_test, y_pred_train)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(y_test, y_pred_train))

# Predict within test
print("*** TEST- Accuracy metrics...")
y_pred_test = knn.predict(dfx_test)
cm = met.confusion_matrix(df_test_results.TARGET_C, y_pred_test)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(df_test_results.TARGET_C, y_pred_test))

# ------------------------------ CatBoost ----------------------------------
print("---------------------------------------------------------------------")
print("CatBoost ...")
catb = cb.CatBoostClassifier(eval_metric="AUC",depth=10, iterations=50, l2_leaf_reg=0.2, learning_rate=0.01)
catb.fit(dfx_train, dfy_train, verbose=25)

# Predict within train
print("*** TRAIN- Accuracy metrics...")
y_pred_train = catb.predict(x_test)
cm = met.confusion_matrix(y_test, y_pred_train)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(y_test, y_pred_train))

# Predict within test
print("*** TEST- Accuracy metrics...")
y_pred_test = catb.predict(dfx_test)
cm = met.confusion_matrix(df_test_results.TARGET_C, y_pred_test)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(df_test_results.TARGET_C, y_pred_test))

# ------------------------------ SVM ----------------------------------
print("---------------------------------------------------------------------")
print("SVM ...")
clf = svm.SVC(kernel='rbf')
clf.fit(dfx_train, dfy_train)

# Predict within train
print("*** TRAIN- Accuracy metrics...")
y_pred_train = clf.predict(x_test)
cm = met.confusion_matrix(y_test, y_pred_train)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(y_test, y_pred_train))

# Predict within test
print("*** TEST- Accuracy metrics...")
y_pred_test = clf.predict(dfx_test)
cm = met.confusion_matrix(df_test_results.TARGET_C, y_pred_test)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(df_test_results.TARGET_C, y_pred_test))

# ------------------------------ Tensor flow ----------------------------------
print("---------------------------------------------------------------------")
print("Tensor flow ...")
tf = Sequential()

tf.add(Dense(200, input_dim=213, activation='relu'))
tf.add(Dropout(0.3))

tf.add(Dense(108, activation='relu'))
tf.add(Dropout(0.5))

tf.add(Dense(150, activation='relu'))
tf.add(Dropout(0.5))

tf.add(Dense(1, activation='sigmoid'))
tf.add(Dense(2, activation='softmax'))

tf.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

tf.fit(dfx_train, dfy_train, epochs=100, batch_size=30000, shuffle=True)

# Predict within train
print("*** TRAIN- Accuracy metrics...")
y_pred_train = tf.predict(x_test)
df_y = pd.DataFrame(y_pred_train)

for r in range(0, df_y.shape[0]):
    if df_y.loc[r, 0] > 0:
        df_y.loc[r, "OP"] = 1
    else:
        df_y.loc[r, "OP"] = 0
        
cm = met.confusion_matrix(y_test, df_y.OP)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(y_test, df_y.OP))

# Predict within test
print("*** TEST- Accuracy metrics...")
y_pred_test = tf.predict(dfx_test)
df_y = pd.DataFrame(y_pred_test)

for r in range(0, df_y.shape[0]):
    if df_y.loc[r, 0] > 0:
        df_y.loc[r, "OP"] = 1
    else:
        df_y.loc[r, "OP"] = 0
        
cm = met.confusion_matrix(df_test_results.TARGET_C, df_y.OP)
print("Confusion Matrix ...")
print(cm)
print("F1 Score= ", met.f1_score(df_test_results.TARGET_C, df_y.OP))