import pandas as pd
import numpy as np
import sklearn.preprocessing as pre
import sklearn.linear_model as linmod

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


print("Is_Male= ", df.Is_Male.corr(df.Loan_Status))
print("Is_Female= ", df.Is_Female.corr(df.Loan_Status))
print("Dependents= ", df.Dependents.corr(df.Loan_Status))
print("Credit_History= ", df.Credit_History.corr(df.Loan_Status))
print("Total_Income= ", df.Total_Income.corr(df.Loan_Status))
print("Loan_Amount_Term= ", df.Loan_Amount_Term.corr(df.Loan_Status))
print("Is_Married= ", df.Is_Married.corr(df.Loan_Status))
print("Is_Not_Married= ", df.Is_Not_Married.corr(df.Loan_Status))
print("Is_Graduate= ", df.Is_Graduate.corr(df.Loan_Status))
print("Is_Not_Graduate= ", df.Is_Not_Graduate.corr(df.Loan_Status))
print("Is_Self_Employed= ", df.Is_Self_Employed.corr(df.Loan_Status))
print("Is_Not_Self_Employed= ", df.Is_Not_Self_Employed.corr(df.Loan_Status))
print("Is_PA_Semiurban= ", df.Is_PA_Semiurban.corr(df.Loan_Status))
print("Is_PA_Urban= ", df.Is_PA_Urban.corr(df.Loan_Status))
print("Is_PA_Rural= ", df.Is_PA_Rural.corr(df.Loan_Status))
