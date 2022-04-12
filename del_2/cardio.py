import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import joblib

cardio_train = "Data/cardio_train.csv"

data = pd.read_csv(cardio_train, delimiter=';')


def initial_analyse(df):
    """
    :param df: DataFrame
    :return: print
    """
    print("info():\n", df.info(), "\n")
    print("describe():\n", df.describe(), "\n")
    print("value_counts():\n", df.value_counts(), "\n")
    print("head():\n", df.head(), "\n")
    print("tail():\n", df.tail(), "\n")
    print("columns:\n", df.columns, "\n")
    print("index:\n", df.index, "\n")


initial_analyse(data)


def check_min_max(df, df_col):
    """
    :param df: DataFrame
    :param df_col: str
    :return:
    """
    print(f"min: {df[df_col].min()} max: {df[df_col].max()}")


check_min_max(data, 'age')
check_min_max(data, 'height')
check_min_max(data, 'weight')
check_min_max(data, 'ap_hi')
check_min_max(data, 'ap_lo')
check_min_max(data, 'cholesterol')
check_min_max(data, 'gluc')
check_min_max(data, 'smoke')
check_min_max(data, 'alco')
check_min_max(data, 'active')
check_min_max(data, 'cardio')
check_min_max(data, 'id')

# store length on DataFrame for calculations
num_id = len(data['id'])


# a) Hur många är positiva för hjärt-kärlsjukdom och hur många är negativa?
def calculate_rate_heart_disease(df):
    """
    :param df: DataFrame
    :return: None
    """
    pos_heart_disease = len(df.query("cardio == 1"))
    neg_heart_disease = len(df.query("cardio == 0"))
    print(f"hjärt-kärlsjukdom: Antal positiva:{pos_heart_disease} Antal negativa:{neg_heart_disease}")
    print(f"hjärt-kärlsjukdom: Andel positiva:{pos_heart_disease / num_id} Andel negativa:{neg_heart_disease / num_id}")


calculate_rate_heart_disease(data)

# b) Hur stor andel har normala, över normala och långt över normala kolesterolvärden? Rita ett tårtdiagram.
cardio_low = len(data.query("cholesterol == 1")) / num_id
cardio_normal = len(data.query("cholesterol == 2")) / num_id
cardio_high = len(data.query("cholesterol == 3")) / num_id


def plot_pie_cholesterol(lst, title):
    """
    Plot of cholesterol values
    :param lst: floats
    :param title: str
    :return:
    """
    plt.pie(lst, labels=["low", "normal", "high"])
    plt.title(title)
    plt.show()


sizes = [cardio_low, cardio_normal, cardio_high]
plot_pie_cholesterol(sizes, "Cardio rate for % on sum population samples")


# c) Hur ser åldersfördelningen ut? Rita ett histogram.
def plot_hist(df, x_value, bin_num, title):
    """
     approx 29.5 - 65 years , split age in days in interval were min= 10798 and max= 23713, delta=12915
     Setting interval to 10000 - 24000, delta = 14000, delta divided by 10 ger 1400 (3.8 years) in each interval
     gives bin=10
    :param df: DataFrame
    :param x_value: str
    :param bin_num: int
    :param title: str
    :return: None
    """
    plt.hist(data=df, x=x_value, bins=bin_num)
    plt.title(title)
    plt.xlabel(x_value)
    plt.ylabel("Amount")
    plt.show()


plot_hist(data, 'age', 10, "Age distribution in days")
plot_hist(data, 'height', 40, "height distribution in cm")


# d) Hur stor andel röker?
def rate_smoker(df):
    """
    :param df: DataFrame
    :return: None
    """
    smoker = len(df.query("smoke == 1"))
    non_smoker = len(df.query("smoke == 0"))

    part_non_smoker = 100 * (non_smoker / num_id)
    part_smoker = 100 * (smoker / num_id)
    print(f"smoker: {part_non_smoker:.2f} % \nnon smoker: {part_smoker:.2f} % ")


rate_smoker(data)

# e) Hur ser viktfördelningen ut? Rita lämpligt diagram.
# min: 10.0 max: 200.0 bin 30 gives 6,3 kg steps
plot_hist(data, 'weight', 30, "Weight distribution in kg")


# f) Hur ser längdfördelningen ut? Rita lämpligt diagram.
def plot_box(df, x_value, title, y_value=None):
    """
    :param df: DataFrame
    :param x_value: str
    :param title: str
    :param y_value: None or
    :return: None or str
    """
    sns.boxplot(x=x_value, y=y_value, data=df)
    plt.title(title)
    plt.show()


plot_box(data, 'height', "height distribution in cm")

# g) Hur stor andel av kvinnor respektive män har hjärt-kärlsjukdom? Rita lämpligt diagram
num_women = len(data.query("gender == 1"))
num_men = len(data.query("gender == 2"))
women_heart_disease = len(data.query("gender == 1 & cardio == 1")) / num_women
men_heart_disease = len(data.query("gender == 2 & cardio == 1")) / num_men
women_heart_wo_disease = len(data.query("gender == 1 & cardio == 0")) / num_women
men_heart_wo_disease = len(data.query("gender == 2 & cardio == 0")) / num_men

print(f"women w heart disease {women_heart_disease:.2f} women wo heart disease {women_heart_wo_disease:.2f}")
print(f"men w heart disease {men_heart_disease:.2f} men wo heart disease {men_heart_wo_disease:.2f}")


def plot_pie(lst, title):
    """
    :param lst:
    :param title:
    :return:
    """
    plt.pie(lst,
            labels=["women wo heart disease", "women w heart disease", "men w heart disease", "men wo heart disease"],
            autopct='%1.1f%%')
    plt.title(title)
    plt.show()


size = [women_heart_wo_disease, women_heart_disease, men_heart_disease, men_heart_wo_disease]
plot_pie(size, "Women and men")


def plot_histogram(df, x_value, hue_str):
    """
    :param df: DataFrame
    :param x_value: str
    :param hue_str: str
    :return: None
    """
    sns.histplot(data=df, x=x_value, hue=hue_str, multiple="dodge", discrete=True)
    plt.legend(title='Cardio', loc='upper right', labels=['Men', 'Women'])
    plt.title("Women and men rate of heart disease")
    plt.xlabel("")
    plt.show()


plot_histogram(data, "cardio", "gender")

# 2.1.0 - Feature engineering BMI
# adding a column with age in year for plot
data['age_year'] = data['age'] / 365


def calculate_BMI(df, height_str, weight_str):
    """
    :param df: DataFrame
    :param height_str: str
    :param weight_str: str
    :return: DataFrame
    """
    data["bmi"] = (df[weight_str] / (df[height_str] / 100) ** 2).round(1)
    return data["bmi"]


data["bmi"] = calculate_BMI(data, "height", "weight")

"""plot_histogram(data, "bmi", "active")
plot_histogram(data, "bmi", "gender")
plot_histogram(data, "bmi", "height")"""

# filtering
data_filtered = data[(data['bmi'] > 10) & (data['bmi'] < 138) &
                     (data['height'] > 90) & (data['height'] < 208) &
                     (data['ap_hi'] < 400) & (data['ap_hi'] > 30) &
                     (data['ap_lo'] < 250) & (data['ap_lo'] > 30)]

data_filtered = data_filtered[data_filtered["ap_lo"] < data_filtered["ap_hi"]]

print(f"height min: {data_filtered['height'].min()}, height max: {data_filtered['height'].max()}")
print(f"BMI min: {data_filtered['bmi'].min()}, BMI max: {data_filtered['bmi'].max()}")
print(f"ap_hi min: {data_filtered['ap_hi'].min()}, ap_hi max: {data_filtered['ap_hi'].max()}")
print(f"ap_lo min: {data_filtered['ap_lo'].min()}, ap_lo max: {data_filtered['ap_lo'].max()}")
print(f"weight min: {data_filtered['weight'].min()}, weight max: {data_filtered['weight'].max()}")


def box_plot_before_after(df1, df2, x_value, sup_title):
    """
    Box plot before and after filtering outliers
    :param df1: DataFrame , not filtered
    :param df2: DataFrame , filtered
    :param x_value: str
    :param sup_title: str
    :return: None
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    sns.boxplot(x=x_value, data=df1, ax=ax[0])
    sns.boxplot(x=x_value, data=df2, ax=ax[1])
    fig.suptitle(sup_title)
    ax[0].set_title("before filter", loc='right')
    ax[1].set_title("after filter", loc='right')
    plt.show()


box_plot_before_after(data, data_filtered, "ap_lo", "ap_lo distribution")
box_plot_before_after(data, data_filtered, "ap_hi", "ap_hi distribution")
box_plot_before_after(data, data_filtered, "age_year", "age_year distribution")
box_plot_before_after(data, data_filtered, "weight", "weight distribution")

# drop colum age_year after plot and categories
data_filtered = data_filtered.drop(columns=["age_year"], axis=1)


# 2.1.0 b) kapa en kategorisk BMI-feature med kategorierna:
def add_category(df, col, low_threshold, new_column, category_str, limit_type=None, high_threshold=None):
    """
    adding new column with categories to DataFrame
    :param df: DataFrame
    :param col: column to get values from
    :param low_threshold: the low limit for filtering
    :param new_column: str for the new column name
    :param category_str: str for category eg. "overweight"
    :param limit_type: To control which limit types are wanted only < or  >= and <  or only >=
    :param high_threshold: the high limit for filtering
    :return: DataFrame wih new column added with categories
    länk - jag tappade sidan då datorn hängde sig men det jag lärde mig var att man i loc kunde
    skriva in nytt kolumn namn och appenda en str i denna kolumn
    alltså , new_column] = category_str
    """
    if limit_type == 1:
        df.loc[(df[col] < low_threshold), new_column] = category_str
    elif limit_type == 2:
        df.loc[(df[col] >= low_threshold) & (df[col] < high_threshold), new_column] = category_str
    elif limit_type == 3:
        df.loc[(df[col] >= low_threshold), new_column] = category_str


add_category(data_filtered, 'bmi', 18.5, "bmi_status", "Underweight", 1)
add_category(data_filtered, 'bmi', 18.5, "bmi_status", "normal range", 2, 25)
add_category(data_filtered, 'bmi', 25, "bmi_status", "overweight", 2, 30)
add_category(data_filtered, 'bmi', 30, "bmi_status", "obese (class I)", 2, 35)
add_category(data_filtered, 'bmi', 35, "bmi_status", "obese (class II)", 2, 40)
add_category(data_filtered, 'bmi', 40, "bmi_status", "obese (class III)", 3)

# 2.1.1 - Feature engineering blodtryck
data_filtered.loc[(data_filtered['ap_hi'] < 120) & (data_filtered['ap_lo'] < 80), "blood_pressure"] = "healthy"

data_filtered.loc[(data_filtered['ap_hi'] >= 120) & (data_filtered['ap_hi'] < 130) &
                  (data_filtered['ap_lo'] < 80), "blood_pressure"] = "elevated"

data_filtered.loc[(data_filtered['ap_hi'] >= 130) & (data_filtered['ap_hi'] < 140) |
                  (data_filtered['ap_lo'] >= 80) & (data_filtered['ap_lo'] < 90),
                  "blood_pressure"] = "stage 1 hypertension"

data_filtered.loc[(data_filtered['ap_hi'] >= 140) & (data_filtered['ap_hi'] <= 180)
                  | (data_filtered['ap_lo'] >= 90) & (data_filtered['ap_lo'] <= 120),
                  "blood_pressure"] = "stage 2 hypertension"

data_filtered.loc[(data_filtered['ap_hi'] > 180) | (data_filtered['ap_lo'] > 120),
                  "blood_pressure"] = "hypertension crisis"

"""
2.2.0 - Visualiseringar andel sjukdomar
"""
dict_bmi = {'Underweight': "bmi_status", 'normal range': "bmi_status", 'overweight': "bmi_status",
            'obese (class I)': "bmi_status", 'obese (class II)': "bmi_status", 'obese (class III)': "bmi_status"}
dict_blood_pressure = {'healthy': "blood_pressure", 'elevated': "blood_pressure",
                       'stage 1 hypertension': "blood_pressure", 'stage 2 hypertension': "blood_pressure",
                       'hypertension crisis': "blood_pressure"}
dict_smoking = {1: "smoke", 0: "smoke"}
dict_alco = {1: "alco", 0: "alco"}
dict_training = {1: "active", 0: "active"}
dict_gluc = {1: "gluc", 2: "gluc", 3: "gluc"}
dict_cholesterol = {1: "cholesterol", 2: "cholesterol", 3: "cholesterol"}
dict_gender = {1: "gender", 2: "gender"}


def calculate_proportion(df, d_category_col):
    """
    :param df: DataFrame
    :param d_category_col: dict
    :return: DataFrame
    """
    proportion_df2 = pd.DataFrame()
    n = 0
    for key, value in d_category_col.items():
        filtered_df = df[(df[value] == key)]
        count_cardio = len(filtered_df[filtered_df["cardio"] == 1])
        precent = count_cardio / len(filtered_df)
        prop2 = pd.DataFrame({key: precent}, index=[n + 1])
        proportion_df2 = pd.concat([proportion_df2, prop2], axis=1)
        n = 0
    return proportion_df2


proportion_bmi_lst = calculate_proportion(data_filtered, dict_bmi)
proportion_smoking_lst = calculate_proportion(data_filtered, dict_smoking)
proportion_bp_lst = calculate_proportion(data_filtered, dict_blood_pressure)
proportion_alco_lst = calculate_proportion(data_filtered, dict_alco)
proportion_train_lst = calculate_proportion(data_filtered, dict_training)
proportion_gluc_lst = calculate_proportion(data_filtered, dict_gluc)
proportion_chol_lst = calculate_proportion(data_filtered, dict_cholesterol)
proportion_sex_lst = calculate_proportion(data_filtered, dict_gender)


def create_lst_for_plot():
    """
    renaming columns and create list for plot
    :return: list
    """
    proportion_smoking_lst.columns = ["smoker", "non smoker"]
    proportion_alco_lst.columns = ["alcohol", "no alcohol"]
    proportion_train_lst.columns = ["training", "no training"]
    proportion_gluc_lst.columns = ["normal", "above normal", "well above normal"]
    proportion_chol_lst.columns = ["normal", "above normal", "well above normal"]
    proportion_sex_lst.columns = ["women", "men"]

    data_lst = [proportion_bmi_lst, proportion_smoking_lst, proportion_bp_lst, proportion_alco_lst,
                proportion_train_lst, proportion_gluc_lst, proportion_chol_lst, proportion_sex_lst]
    return data_lst


df_lst = create_lst_for_plot()


def plot_cardio_vs_feature(data_lst):
    titles = ["BMI Status", "Smoking", "Blood pressure", "Alchohol", "Training", "Glucose", "Cholesterol", "Gender"]
    num_row = 4
    num_col = 2

    fig, axes = plt.subplots(num_row, num_col, figsize=(15, 8))
    count = 0
    for r in range(num_row):
        for c in range(num_col):
            sns.barplot(data=data_lst[count], ax=axes[r, c], orient="horizontal")
            plt.tight_layout()
            axes[r, c].set_title(titles[count])
            axes[r, c].tick_params(axis="x", rotation=45)
            count += 1
            plt.suptitle("Cardio vs features")
    plt.show()


plot_cardio_vs_feature(df_lst)


# 2.2.1 Visualisering korrelation
def heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')

    # https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
    plt.figure(figsize=(8, 12))
    heatmaps = sns.heatmap(df.corr()[['cardio']].sort_values(by='cardio', ascending=False), vmin=-1,
                           vmax=1, annot=True, cmap='BrBG')
    heatmaps.set_title('Features Correlating with cardio', fontdict={'fontsize': 18}, pad=16)
    plt.show()


heatmap(data_filtered)

"""
2.3 - Skapa två dataset
created one dataframe since I used OneHotEncoding and then splited in two dataframes
"""
df_reduced = data_filtered.drop(columns=["ap_hi", "ap_lo", "height", "weight", "bmi"], axis=1)


def one_hot_encoder(df_to_convert, sub_lst):
    """
    :param df_to_convert: DataFrame
    :param sub_lst: list
    :return: DataFrame
    """
    # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features
    X = df_to_convert[sub_lst]  # drop='first' but if more choices than 2 'first' needed to be removed
    drop_enc = OneHotEncoder(handle_unknown='ignore').fit(X)
    encoded = drop_enc.transform(X).toarray()
    print(drop_enc.transform(X).toarray())
    print(drop_enc.get_feature_names_out())
    return encoded


encode = one_hot_encoder(df_reduced, ["gender", "bmi_status", "blood_pressure", "cholesterol", "gluc"])
data_filtered = data_filtered.reset_index().drop(columns=['index'])
data_encode = pd.DataFrame(encode, columns=['gender_1', 'gender_2', 'bmi_status_Underweight', 'bmi_status_normal range',
                                            'bmi_status_obese (class I)', 'bmi_status_obese (class II)',
                                            'bmi_status_obese (class III)',
                                            'bmi_status_overweight', 'blood_pressure_elevated',
                                            'blood_pressure_healthy',
                                            'blood_pressure_hypertension crisis', 'blood_pressure_stage 1 hypertension',
                                            'blood_pressure_stage 2 hypertension', 'cholesterol_1', 'cholesterol_2',
                                            'cholesterol_3',
                                            'gluc_1', 'gluc_2', 'gluc_3'])

data_encoded = data_filtered.join(data_encode, how='outer')


def split_data(df):
    """
    split to two dataframe to see which one is the best
    :param df: DataFrame
    :return: DataFrame
    """
    data_encode1 = df.drop(columns=["id", "gender", "ap_hi", "ap_lo", "height", "weight", "bmi",
                                    "bmi_status", "blood_pressure", "gluc", "cholesterol"], axis=0)
    data_encode2 = df.drop(columns=["id", "gender", "height", "weight", "bmi_status", "blood_pressure", "gluc",
                                    "cholesterol"], axis=0)
    return data_encode1


chosen_data = split_data(data_encoded)


# 2.4 - Välja modell
def split_to_y_x(df, col):
    """
    :param df: DataFrame
    :param col: str
    :return: DataFrame, DataFrame
    """

    x, y = df.drop([col], axis=1), df[col]
    return x, y


x, y = split_to_y_x(chosen_data, "cardio")


def split_train_test(x_value, y_value, t_size):
    """
    :param x_value: DataFrame
    :param y_value: DataFrame
    :param t_size: float
    :return: DataFrame, DataFrame, DataFrame, DataFrame
    """
    x_train, x_test, y_train, y_test = train_test_split(x_value, y_value, test_size=t_size, random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test


# train, val, test - 80, 10, 10
X_train, X_test, y_train, y_test = split_train_test(np.array(x), np.array(y), 0.20)
X_val, X_test, y_val, y_test = split_train_test(X_test, y_test, 0.50)


def evaluate_model(model, test_x, test_y, title):
    """
    :param model: model
    :param test_x: DataFrame
    :param test_y: DataFrame
    :return: None
    """
    y_predict = model.predict(test_x)
    print(title)
    print(classification_report(test_y, y_predict))
    cm = confusion_matrix(test_y, y_predict)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()


"""
The purpose of the pipeline is to assemble several steps that can be cross-validated together while 
setting different parameters. Pipelines with KNeighborsClassifier, LogisticRegression, RandomForestClassifier
"""
pipe_KNN = Pipeline([("scaler", None), ("knn", KNeighborsClassifier())])
pipe_log_reg = Pipeline([("scaler", None), ("logistic", LogisticRegression())])
pipe_RFC = Pipeline([("scaler", None), ("random_forest", RandomForestClassifier())])


def cross_validation(train_x, train_y):
    """
    :param train_x: DataFrame
    :param train_y: DataFrame
    :return: None
    """
    param_grid_RFC = {"random_forest__n_estimators": list(range(20, 60)),
                      "random_forest__criterion": ["gini", "entropy"],
                      "random_forest__max_features": ["auto", "sqrt", "log2"],
                      "scaler": [StandardScaler(), MinMaxScaler()]}

    param_grid_Log_reg = {"logistic__penalty": ['l2', 'elasticnet', 'none', 'l1'],
                          "logistic__solver": ["newton-cg", "lbfgs", "liblinear", 'saga'],
                          "logistic__C": [100, 10, 1.0, 0.1, 0.01],
                          "logistic__multi_class": ["auto", "ovr", "multinomial"],
                          "logistic__max_iter": [100, 1000, 10000],
                          "scaler": [StandardScaler(), MinMaxScaler()]}

    param_grid_knn = {"knn__n_neighbors": list(range(5, 30)), "scaler": [StandardScaler(), MinMaxScaler()]}

    # cross validation by GridSearchCV
    classifier_RFC = GridSearchCV(estimator=pipe_RFC, param_grid=param_grid_RFC, cv=10, verbose=1, scoring="recall")
    classifier_Log_reg = GridSearchCV(estimator=pipe_log_reg, param_grid=param_grid_Log_reg, cv=5, verbose=1,
                                      scoring="recall")
    classifier_knn = GridSearchCV(estimator=pipe_KNN, param_grid=param_grid_knn, cv=5, verbose=1, scoring="recall")

    grid_search_RFC = classifier_RFC.fit(train_x, train_y)
    grid_search_Log_reg = classifier_Log_reg.fit(train_x, train_y)
    grid_search_knn = classifier_knn.fit(train_x, train_y)

    # Print best score and best hyperparameter
    print(f"Best Score RFC: {grid_search_RFC.best_score_:.4f} using {grid_search_RFC.best_params_}")
    print(f"Best Score Log_reg: {grid_search_Log_reg.best_score_:.4f} using {grid_search_Log_reg.best_params_}")
    print(f"Best Score KNN: {grid_search_knn.best_score_:.4f} using {grid_search_knn.best_params_}")

    classifier_VC = VotingClassifier([('logistic', classifier_knn),
                                      ('rfc', classifier_RFC),
                                      ('knn', classifier_Log_reg)], voting="hard")
    classifier_VC.fit(X_val, y_val)

    # evaluate model on decided hyperparameter
    evaluate_model(classifier_Log_reg, X_val, y_val, "logistic")
    evaluate_model(classifier_RFC, X_val, y_val, "random_forest")
    evaluate_model(classifier_knn, X_val, y_val, "knn")
    evaluate_model(classifier_VC, X_val, y_val, "VotingClassifier")


cross_validation(X_train, y_train)


def model(x_training, y_training):
    """
    :param x_training: DataFrame
    :param y_training: DataFrame
    :return: model
    """
    model = Pipeline([("scaler", MinMaxScaler()), ("knn", KNeighborsClassifier(n_neighbors=13))])
    model.fit(x_training, y_training)
    return model


model(X_test, y_test)


# 2.7 "Deploy"
def reduced_data(df):
    """
    :param df: DataFrame
    :return: DataFrame
    """
    df_100_row = df.sample(n=100, random_state=42).drop(columns=["cardio"], axis=0)
    df_100_row.to_csv("Data/test_samples.csv", sep=";", index=False)
    return df_100_row


df_100_rows = reduced_data(chosen_data)
data_reduced = chosen_data[~chosen_data.isin(df_100_rows)].dropna()


def save_model(df):
    """
    :return: None
    """
    x, y = split_to_y_x(df, "cardio")
    X_train, X_test, y_train, y_test = split_train_test(x, y, 0.30)
    chosen_model = model(X_train, y_train)
    # save the model to disk
    filename = 'Model/model.pkl'
    joblib.dump(chosen_model, open(filename, 'wb'), compress=True)


save_model(data_reduced)
