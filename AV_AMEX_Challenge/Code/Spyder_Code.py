import pandas as pd
import numpy as np
import sklearn.preprocessing as pre
import sklearn.linear_model as linmod
import sklearn.neighbors as nb
import sklearn.tree as tree
import sklearn.ensemble as ens
import sklearn.metrics as met
import sklearn.cross_validation as cv

df_train = pd.read_csv("/home/mukundhan/02_Tech/Python/AV_AMEX_Challenge/PreProcessed_Files/train.csv")
df_test = pd.read_csv("/home/mukundhan/02_Tech/Python/AV_AMEX_Challenge/PreProcessed_Files/test.csv")

df_train["src"] = "train"
df_test["src"] = "test"

df = pd.concat([df_train, df_test], axis=0, sort=False)

# One hot encoding
dfprod_dummies = pd.get_dummies(df["PRODUCT"], prefix = "PROD_")
dfprodcat_dummies = pd.get_dummies(df["PRODUCT_CATEGORY_1"], prefix = "PROD_CAT1_")
dfgender_dummies = pd.get_dummies(df["GENDER"], prefix = "GENDER_")

df = pd.concat([df, dfprod_dummies], axis=1)
df = pd.concat([df, dfprodcat_dummies], axis=1)
df = pd.concat([df, dfgender_dummies], axis=1)

# Scaling
df_click_cnt = pd.DataFrame()
df_click_cnt["CLICK_CNT"] = df["CLICK_CNT"]
rs = pre.MinMaxScaler()
df_click_cnt = pd.DataFrame(data=rs.fit_transform(df_click_cnt), columns=df_click_cnt.columns)
df.drop(["CLICK_CNT"], axis="columns", inplace=True)
df["CLICK_CNT"] = df_click_cnt["CLICK_CNT"] 

# Give more importance to highly correlaletd columns
#df["USER_DEPTH"] = df["USER_DEPTH"] * 5
#df["CLICK_DURATION"] = df["CLICK_DURATION"] * 5
#df["CLICK_CNT"] = df["CLICK_CNT"] * 5

# Drop unwanted columns
df.drop(["PRODUCT", "PRODUCT_CATEGORY_1", "GENDER", "PRODUCT_CATEGORY_2", "DATETIME", "USER_ID", "CAMPAIGN_ID", 
         "WEBPAGE_ID", "VAR_1", "CITY_DEVELOPMENT_INDEX"], 
        axis="columns", inplace=True)


# Fix x & y columns
xcols = ["USER_GROUP_ID", "AGE_LEVEL", "USER_DEPTH", "CLICK_DURATION", "CLICK_CNT",
         "PROD__A", "PROD__B", "PROD__C", "PROD__D", "PROD__E", "PROD__F", "PROD__G",
       "PROD__H", "PROD__I", "PROD__J", "PROD_CAT1__1", "PROD_CAT1__2", "PROD_CAT1__3", 
       "PROD_CAT1__4", "PROD_CAT1__5", "GENDER__Female", "GENDER__Male"]

ycol = ["IS_CLICK"] 

dfx_train = df.loc[df.src == "train", xcols]
dfy_train = df.loc[df.src == "train", ycol]
dfx_test = df.loc[df.src == "test", xcols]

# Check correlation
df1 = df.loc[df.src == "train"]
print("USER_GROUP_ID= ", df1["USER_GROUP_ID"].corr(df1["IS_CLICK"]))
print("USER_DEPTH= ", df1["USER_DEPTH"].corr(df1["IS_CLICK"]))
print("CLICK_DURATION= ", df1["CLICK_DURATION"].corr(df1["IS_CLICK"]))
print("CLICK_CNT= ", df1["CLICK_CNT"].corr(df1["IS_CLICK"]))


# Convert the target from float to int, because o/p is expected is like that
dfy_train.loc[dfy_train["IS_CLICK"] == 0.0, "IS_CLICK"] = 0
dfy_train.loc[dfy_train["IS_CLICK"] == 1.0, "IS_CLICK"] = 1
dfy_train.IS_CLICK = dfy_train.IS_CLICK.astype(int)

# Test within train data
x_train, x_test, y_train, y_test = cv.train_test_split(dfx_train, dfy_train, test_size=0.3, random_state=0)
logreg = linmod.LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
#print('Logistic Regression accuracy= ', logreg.score(x_test, y_test))

cm = met.confusion_matrix(y_test, y_pred)
#print("Confusion Matrix= ", cm)
######################## Logistic Regression ######################
logreg = linmod.LogisticRegression()
logreg.fit(dfx_train, dfy_train)

# print(logreg.predict(dfx_test))
df_prob1 = pd.DataFrame(logreg.predict_proba(dfx_test))
#print(df_prob.shape)
#print(df_prob.loc[:, 1])

dfy_submission1 = pd.DataFrame()
dfy_submission1["session_id"] = df.loc[df.src == "test", "SESSION_ID"]
dfy_submission1["is_click"] = df_prob1.loc[:, 1]

dfy_submission1.to_csv("/home/mukundhan/02_Tech/Python/AV_AMEX_Challenge/output/lr.csv", index=False)
######################## Random forrest ######################
rf = ens.RandomForestClassifier(n_estimators=100, max_features="auto", max_depth = 30)
rf.fit(dfx_train, dfy_train)
df_prob2 = pd.DataFrame(rf.predict_proba(dfx_test))

dfy_submission2 = pd.DataFrame()
dfy_submission2["session_id"] = df.loc[df.src == "test", "SESSION_ID"]
dfy_submission2["is_click"] = df_prob2.loc[:, 1]

dfy_submission2.to_csv("/home/mukundhan/02_Tech/Python/AV_AMEX_Challenge/output/rf.csv", index=False)
