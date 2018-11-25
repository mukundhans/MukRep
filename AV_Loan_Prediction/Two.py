import pandas as pd
import numpy as np
import sklearn.preprocessing as pre
import sklearn.linear_model as linmod
import sklearn.neighbors as nb
import sklearn.tree as tree
import sklearn.ensemble as ens

df_train = pd.read_csv("/home/mukundhan/02_Tech/Python/AV_Loan_Prediction/train.csv")
df_test = pd.read_csv("/home/mukundhan/02_Tech/Python/AV_Loan_Prediction/test.csv")

df_train["src"] = "train"
df_test["src"] = "test"

df_train.dropna(subset=['LoanAmount'], inplace=True)
df_train.dropna(subset=['Credit_History'], inplace=True)

df = pd.concat([df_train, df_test], axis=0, sort=False)

df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)

df.loc[(df.Loan_Status == "Y") & (df.Credit_History.isna()), "Credit_History"] = 1
df.loc[(df.Loan_Status == "N") & (df.Credit_History.isna()), "Credit_History"] = 0

df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].mean(), inplace=True)

df["Total_Income"] = df.ApplicantIncome + df.CoapplicantIncome

df.loc[df.Dependents == "3+", "Dependents"] = 1
df.loc[df.Dependents == "2", "Dependents"] = 2
df.loc[df.Dependents == "1", "Dependents"] = 3
df.loc[df.Dependents == "0", "Dependents"] = 4

df.loc[df.Loan_Amount_Term <= 60, "Loan_Amount_Term"] = 1
df.loc[(df.Loan_Amount_Term > 60) & (df.Loan_Amount_Term <= 200), "Loan_Amount_Term"] = 2
df.loc[(df.Loan_Amount_Term > 200) & (df.Loan_Amount_Term <= 300), "Loan_Amount_Term"] = 3
df.loc[(df.Loan_Amount_Term > 300), "Loan_Amount_Term"] = 4

df.loc[df["Gender"] == "Male", "Is_Male"] = 1
df.loc[df["Gender"] == "Female", "Is_Male"] = 0

df.loc[df["Gender"] == "Female", "Is_Female"] = 1
df.loc[df["Gender"] == "Male", "Is_Female"] = 0

df.loc[df["Married"] == "Yes", "Is_Married"] = 1
df.loc[df["Married"] == "No", "Is_Married"] = 0

df.loc[df["Married"] == "No", "Is_Not_Married"] = 1
df.loc[df["Married"] == "Yes", "Is_Not_Married"] = 0

df.loc[df["Education"] == "Graduate", "Is_Graduate"] = 1
df.loc[df["Education"] == "Not Graduate", "Is_Graduate"] = 0

df.loc[df["Education"] == "Not Graduate", "Is_Not_Graduate"] = 1
df.loc[df["Education"] == "Graduate", "Is_Not_Graduate"] = 0

df.loc[df["Self_Employed"] == "Yes", "Is_Self_Employed"] = 1
df.loc[df["Self_Employed"] == "No", "Is_Self_Employed"] = 0

df.loc[df["Self_Employed"] == "No", "Is_Not_Self_Employed"] = 1
df.loc[df["Self_Employed"] == "Yes", "Is_Not_Self_Employed"] = 0

df.loc[df["Property_Area"] == "Semiurban", "Is_PA_Semiurban"] = 1
df.loc[df["Property_Area"].isin(["Urban", "Rural"]), "Is_PA_Semiurban"] = 0

df.loc[df["Property_Area"] == "Urban", "Is_PA_Urban"] = 1
df.loc[df["Property_Area"].isin(["Semiurban", "Rural"]), "Is_PA_Urban"] = 0

df.loc[df["Property_Area"] == "Rural", "Is_PA_Rural"] = 1
df.loc[df["Property_Area"].isin(["Semiurban", "Urban"]), "Is_PA_Rural"] = 0

df.loc[df["Loan_Status"] == "Y", "Loan_Status"] = 1
df.loc[df["Loan_Status"] == "N", "Loan_Status"] = 0

df.Dependents = df.Dependents.astype(float)
df.Loan_Status = df.Loan_Status.astype(float)

df.loc[df["Credit_History"] == 1, "Credit_History"] = 10
df.loc[df["Credit_History"] == 0, "Credit_History"] = 1

df.drop(["Gender", "Married", "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome", "Property_Area"], 
        axis='columns', inplace=True)

# Separate x & y in train
xcols = ["Dependents", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Total_Income",
         "Is_Male", "Is_Female", "Is_Married", "Is_Not_Married", "Is_Graduate", "Is_Not_Graduate", "Is_Self_Employed", 
         "Is_Not_Self_Employed", "Is_PA_Semiurban", "Is_PA_Urban", "Is_PA_Rural"]
ycol = ["Loan_Status"]

dfx_train = df.loc[df.src == "train", xcols]
dfy_train = df.loc[df.src == "train", ycol]
dfx_test = df.loc[df.src == "test", xcols]

# Scale x data
#rs = pre.MinMaxScaler()
#dfx_train[["LoanAmount", "Loan_Amount_Term", "Total_Income"]] = rs.fit_transform(dfx_train[["LoanAmount", "Loan_Amount_Term", "Total_Income"]])

######################## Logistic Regression ######################
logreg = linmod.LogisticRegression()
logreg.fit(dfx_train, dfy_train)


dfy_submission = pd.DataFrame()
dfy_submission["Loan_ID"] = df.loc[df.src == "test", "Loan_ID"]
dfy_submission["Loan_Status"] = logreg.predict(dfx_test)

dfy_submission.loc[dfy_submission["Loan_Status"] == 0.0, "Loan_Status"] = "N"
dfy_submission.loc[dfy_submission["Loan_Status"] == 1.0, "Loan_Status"] = "Y"

dfy_submission.to_csv("/home/mukundhan/02_Tech/Python/AV_Loan_Prediction/linreg.csv", index=False)

######################## KNN ######################
for k in range(25):
    knn = nb.KNeighborsClassifier(n_neighbors = k+1, weights='uniform', algorithm='auto')
    knn.fit(dfx_train, dfy_train) 
    y_pred = knn.predict(dfx_test)

dfy_submission1 = pd.DataFrame()
dfy_submission1["Loan_ID"] = df.loc[df.src == "test", "Loan_ID"]
dfy_submission1["Loan_Status"] = y_pred    

dfy_submission1.loc[dfy_submission1["Loan_Status"] == 0.0, "Loan_Status"] = "N"
dfy_submission1.loc[dfy_submission1["Loan_Status"] == 1.0, "Loan_Status"] = "Y"

dfy_submission1.to_csv("/home/mukundhan/02_Tech/Python/AV_Loan_Prediction/knn.csv", index=False)
######################## Decision Tree ######################
t = tree.DecisionTreeClassifier(max_depth=10, random_state=0)
t.fit(dfx_train, dfy_train)

dfy_submission2 = pd.DataFrame()
dfy_submission2["Loan_ID"] = df.loc[df.src == "test", "Loan_ID"]
dfy_submission2["Loan_Status"] = t.predict(dfx_test)    

dfy_submission2.loc[dfy_submission2["Loan_Status"] == 0.0, "Loan_Status"] = "N"
dfy_submission2.loc[dfy_submission2["Loan_Status"] == 1.0, "Loan_Status"] = "Y"

dfy_submission2.to_csv("/home/mukundhan/02_Tech/Python/AV_Loan_Prediction/dtree.csv", index=False)

######################## Random Forest ######################
rf = ens.RandomForestClassifier()
rf.fit(dfx_train, dfy_train)

dfy_submission3 = pd.DataFrame()
dfy_submission3["Loan_ID"] = df.loc[df.src == "test", "Loan_ID"]
dfy_submission3["Loan_Status"] = rf.predict(dfx_test)    

dfy_submission3.loc[dfy_submission3["Loan_Status"] == 0.0, "Loan_Status"] = "N"
dfy_submission3.loc[dfy_submission3["Loan_Status"] == 1.0, "Loan_Status"] = "Y"

dfy_submission3.to_csv("/home/mukundhan/02_Tech/Python/AV_Loan_Prediction/rforest.csv", index=False)
######################## Tensor flow ######################
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

# fix random seed for reproducibility
np.random.seed(7)

# Build model
model = Sequential()
model.add(Dense(12, input_dim=16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='tensor_simple', histogram_freq=0,
                          write_graph=True, write_images=True)
# Fit the model
model.fit(dfx_train, dfy_train, epochs=250, batch_size=50, shuffle=True, callbacks=[tensorboard])

dfy_submission4 = pd.DataFrame()
dfy_submission4["Loan_ID"] = df.loc[df.src == "test", "Loan_ID"]
dfy_submission4["Loan_Status"] = model.predict(dfx_test)
dfy_submission4["Loan_Status"] = dfy_submission4["Loan_Status"].astype("float64")
dfy_submission4.loc[dfy_submission4["Loan_Status"] <= 0.5, "Loan_Status1"] = "N"
dfy_submission4.loc[dfy_submission4["Loan_Status"] > 0.5, "Loan_Status1"] = "Y"
dfy_submission4.drop("Loan_Status", axis="columns", inplace=True)

dfy_submission4.to_csv("/home/mukundhan/02_Tech/Python/AV_Loan_Prediction/tf.csv", index=False)

