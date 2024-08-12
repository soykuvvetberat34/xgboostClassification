from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,mean_absolute_error,r2_score
import pandas as pd
import numpy as np

df_=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\sınıflandırma\\diabetes.csv")
y=df_["Outcome"]
df=df_.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=200)


xgb=XGBClassifier()
xgb_params={
    "n_estimator":[100,500,1000],
    "subsample":[0.6,0.8,1],#göz önünde bulundurulacak örneklem sayısı
    "max_depth":[3,5,7],
    "learning_rate":[0.1,0.01,0.001]
}
xgb_cv=GridSearchCV(xgb,xgb_params,cv=5,n_jobs=-1)
xgb_cv.fit(x_train,y_train)
n_estimator=xgb_cv.best_params_["n_estimator"]
subsample=xgb_cv.best_params_["subsample"]
max_depth=xgb_cv.best_params_["max_depth"]
learning_rate=xgb_cv.best_params_["learning_rate"]
xgb_tuned=XGBClassifier(n_estimator=n_estimator,
                        subsample=subsample,
                        max_depth=max_depth,learning_rate=learning_rate)
xgb_tuned.fit(x_train,y_train)
predict=xgb_tuned.predict(x_test)
acscore=accuracy_score(y_test,predict)
print(acscore)


















