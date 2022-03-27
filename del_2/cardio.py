import pandas as pd
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import joblib


cardio_train = "Data/cardio_train.csv"

df_cardio = pd.read_csv(cardio_train, delimiter=';')


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


initial_analyse(df_cardio)


def check_min_max(df, df_col):
    """
    :param df: DataFrame
    :param df_col: str
    :return:
    """
    print(f"min: {df[df_col].min()} max: {df[df_col].max()}")


check_min_max(df_cardio, 'age')
check_min_max(df_cardio, 'height')
check_min_max(df_cardio, 'weight')
check_min_max(df_cardio, 'ap_hi')
check_min_max(df_cardio, 'ap_lo')
check_min_max(df_cardio, 'cholesterol')
check_min_max(df_cardio, 'gluc')
check_min_max(df_cardio, 'smoke')
check_min_max(df_cardio, 'alco')
check_min_max(df_cardio, 'active')
check_min_max(df_cardio, 'cardio')
check_min_max(df_cardio, 'id')

# store length on DataFrame for calculations
num_id = len(df_cardio['id'])


# a) Hur många är positiva för hjärt-kärlsjukdom och hur många är negativa?
def calculate_rate_heart_disease(df):
    """
    :param df: DataFrame
    :return: None
    """
    pos_heart_disease = len(df.query("cardio == 1"))
    neg_heart_disease = len(df.query("cardio == 0"))
    print(f"hjärt-kärlsjukdom: Antal positiva:{pos_heart_disease} Antal negativa:{neg_heart_disease}")
    print(f"hjärt-kärlsjukdom: Andel positiva:{pos_heart_disease/num_id} Andel negativa:{neg_heart_disease/num_id}")


calculate_rate_heart_disease(df_cardio)

# b) Hur stor andel har normala, över normala och långt över normala kolesterolvärden? Rita ett tårtdiagram.
cardio_low = len(df_cardio.query("cholesterol == 1"))/num_id
cardio_normal = len(df_cardio.query("cholesterol == 2"))/num_id
cardio_high = len(df_cardio.query("cholesterol == 3"))/num_id


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


plot_hist(df_cardio, 'age', 10, "Age distribution in days")
plot_hist(df_cardio, 'height', 40, "height distribution in cm")


# d) Hur stor andel röker?
def rate_smoker(df):
    """
    :param df: DataFrame
    :return: None
    """
    smoker = len(df.query("smoke == 1"))
    non_smoker = len(df.query("smoke == 0"))

    part_non_smoker = 100 * (non_smoker/num_id)
    part_smoker = 100 * (smoker/num_id)
    print(f"smoker: {part_non_smoker:.2f} % \nnon smoker: {part_smoker:.2f} % ")


rate_smoker(df_cardio)

# e) Hur ser viktfördelningen ut? Rita lämpligt diagram.
# min: 10.0 max: 200.0 bin 30 gives 6,3 kg steps
plot_hist(df_cardio, 'weight', 30, "Weight distribution in kg")


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


plot_box(df_cardio, 'height', "height distribution in cm")


# g) Hur stor andel av kvinnor respektive män har hjärt-kärlsjukdom? Rita lämpligt diagram
num_women = len(df_cardio.query("gender == 1"))
num_men = len(df_cardio.query("gender == 2"))
women_heart_disease = len(df_cardio.query("gender == 1 & cardio == 1"))/num_women
men_heart_disease = len(df_cardio.query("gender == 2 & cardio == 1"))/num_men
women_heart_wo_disease = len(df_cardio.query("gender == 1 & cardio == 0"))/num_women
men_heart_wo_disease = len(df_cardio.query("gender == 2 & cardio == 0"))/num_men

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


plot_histogram(df_cardio, "cardio", "gender")


# 2.1.0 - Feature engineering BMI
# adding a column with age in year for plot
df_cardio['age_year'] = df_cardio['age']/365


def calculate_BMI(df, height_str, weight_str):
    """
    :param df: DataFrame
    :param height_str: str
    :param weight_str: str
    :return: DataFrame
    """
    df_cardio["bmi"] = (df[weight_str] / (df[height_str]/100)**2).round(1)
    return df_cardio["bmi"]


df_cardio["bmi"] = calculate_BMI(df_cardio, "height", "weight")

plot_histogram(df_cardio, "bmi", "active")
plot_histogram(df_cardio, "bmi", "gender")
plot_histogram(df_cardio, "bmi", "height")

# filtering
df_cardio_filt = df_cardio[(df_cardio['bmi'] > 10) & (df_cardio['bmi'] < 138) &
                           (df_cardio['height'] > 90) & (df_cardio['height'] < 208) &
                           (df_cardio['ap_hi'] < 400) & (df_cardio['ap_hi'] > 30) &
                           (df_cardio['ap_lo'] < 250) & (df_cardio['ap_lo'] > 30)]

df_cardio_filt = df_cardio_filt[df_cardio_filt["ap_lo"] < df_cardio_filt["ap_hi"]]

print(f"height min: {df_cardio_filt['height'].min()}, height max: {df_cardio_filt['height'].max()}")
print(f"BMI min: {df_cardio_filt['bmi'].min()}, BMI max: {df_cardio_filt['bmi'].max()}")
print(f"ap_hi min: {df_cardio_filt['ap_hi'].min()}, ap_hi max: {df_cardio_filt['ap_hi'].max()}")
print(f"ap_lo min: {df_cardio_filt['ap_lo'].min()}, ap_lo max: {df_cardio_filt['ap_lo'].max()}")
print(f"weight min: {df_cardio_filt['weight'].min()}, weight max: {df_cardio_filt['weight'].max()}")


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


box_plot_before_after(df_cardio, df_cardio_filt, "ap_lo", "ap_lo distribution")
box_plot_before_after(df_cardio, df_cardio_filt, "ap_hi", "ap_hi distribution")
box_plot_before_after(df_cardio, df_cardio_filt, "age_year", "age_year distribution")
box_plot_before_after(df_cardio, df_cardio_filt, "weight", "weight distribution")

# drop colum age_year after plot
df_cardio_filt = df_cardio_filt.drop(columns=["age_year"], axis=1)


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


add_category(df_cardio_filt, 'bmi',  18.5, "bmi_status", "Underweight", 1)
add_category(df_cardio_filt, 'bmi',  18.5, "bmi_status", "normal range", 2, 25)
add_category(df_cardio_filt, 'bmi',  25, "bmi_status", "overweight", 2, 30)
add_category(df_cardio_filt, 'bmi',  30, "bmi_status", "obese (class I)", 2, 35)
add_category(df_cardio_filt, 'bmi',  35, "bmi_status", "obese (class II)", 2, 40)
add_category(df_cardio_filt, 'bmi',  40, "bmi_status", "obese (class III)", 3)


# 2.1.1 - Feature engineering blodtryck
df_cardio_filt.loc[(df_cardio_filt['ap_hi'] < 120) & (df_cardio_filt['ap_lo'] < 80), "blood_pressure"] = "healthy"

df_cardio_filt.loc[(df_cardio_filt['ap_hi'] >= 120) & (df_cardio_filt['ap_hi'] < 130) &
                   (df_cardio_filt['ap_lo'] < 80), "blood_pressure"] = "elevated"

df_cardio_filt.loc[(df_cardio_filt['ap_hi'] >= 130) & (df_cardio_filt['ap_hi'] < 140) |
                   (df_cardio_filt['ap_lo'] >= 80) & (df_cardio_filt['ap_lo'] < 90),
                   "blood_pressure"] = "stage 1 hypertension"

df_cardio_filt.loc[(df_cardio_filt['ap_hi'] >= 140) & (df_cardio_filt['ap_hi'] <= 180)
                   | (df_cardio_filt['ap_lo'] >= 90) & (df_cardio_filt['ap_lo'] <= 120),
                   "blood_pressure"] = "stage 2 hypertension"

df_cardio_filt.loc[(df_cardio_filt['ap_hi'] > 180) | (df_cardio_filt['ap_lo'] > 120),
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
        prop2 = pd.DataFrame({key: precent}, index=[n+1])
        proportion_df2 = pd.concat([proportion_df2, prop2], axis=1)
        n = 0
    return proportion_df2


proportion_bmi_lst = calculate_proportion(df_cardio_filt, dict_bmi)
proportion_smoking_lst = calculate_proportion(df_cardio_filt, dict_smoking)
proportion_bp_lst = calculate_proportion(df_cardio_filt, dict_blood_pressure)
proportion_alco_lst = calculate_proportion(df_cardio_filt, dict_alco)
proportion_train_lst = calculate_proportion(df_cardio_filt, dict_training)
proportion_gluc_lst = calculate_proportion(df_cardio_filt, dict_gluc)
proportion_chol_lst = calculate_proportion(df_cardio_filt, dict_cholesterol)
proportion_sex_lst = calculate_proportion(df_cardio_filt, dict_gender)


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


heatmap(df_cardio_filt)

"""
2.3 - Skapa två dataset
created one dataframe since I used OneHotEncoding and then splited in two dataframes
"""
df_bmi_sex_bloodpressure = df_cardio_filt.drop(columns=["ap_hi", "ap_lo", "height", "weight", "bmi"], axis=1)


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


encode = one_hot_encoder(df_bmi_sex_bloodpressure, ["gender", "bmi_status", "blood_pressure", "cholesterol", "gluc"])
df_cardio_filt = df_cardio_filt.reset_index().drop(columns=['index'])
df_encode = pd.DataFrame(encode, columns=['gender_1', 'gender_2', 'bmi_status_Underweight', 'bmi_status_normal range',
                                          'bmi_status_obese (class I)', 'bmi_status_obese (class II)',
                                          'bmi_status_obese (class III)', 'bmi_status_overweight',
                                          'blood_pressure_elevated', 'blood_pressure_healthy',
                                          'blood_pressure_hypertension crisis', 'blood_pressure_stage 1 hypertension',
                                          'blood_pressure_stage 2 hypertension', 'cholesterol_1', 'cholesterol_2',
                                          'cholesterol_3', 'gluc_1', 'gluc_2', 'gluc_3'])


df_cardio_encoded = df_cardio_filt.join(df_encode, how='outer')


# split to two DataFrame
df_cardio_encode1 = df_cardio_encoded.drop(columns=["id", "gender", "ap_hi", "ap_lo", "height", "weight", "bmi",
                                                          "bmi_status", "blood_pressure", "gluc",
                                                          "cholesterol"], axis=0)
df_cardio_encode2 = df_cardio_encoded.drop(columns=["id", "gender", "height", "weight", "bmi_status",
                                                          "blood_pressure", "gluc", "cholesterol"], axis=0)


# 2.4 - Välja modell
def split_to_y_x(df, col):
    """
    :param df: DataFrame
    :param col: str
    :return: DataFrame, DataFrame
    """

    x, y = df.drop([col], axis=1), df[col]
    return x, y


x, y = split_to_y_x(df_cardio_encode1, "cardio")
# x, y = split_to_y_x(df_cardio_encode2, "cardio") # decide not to use after choosing model


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


# since small data approx 70/30 train/val
X_train, X_test, y_train, y_test = split_train_test(x, y, 0.33)
X_train_val, X_test_val, y_train_val, y_test_val = split_train_test(X_train, y_train, 0.30)


def evaluate_model(model, test_x, test_y):
    """
    :param model: model
    :param test_x: DataFrame
    :param test_y: DataFrame
    :return: None
    """
    y_predict = model.predict(test_x)
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


def tune_hyperparameter(train_x_val, train_y_val):
    """
    :param train_x_val: DataFrame
    :param train_y_val: DataFrame
    :return: None
    """
    param_grid_RFC = {"random_forest__n_estimators": [100, 150, 200, 300],
                      "random_forest__criterion": ["gini", "entropy"],
                      "random_forest__max_features": ["auto", "sqrt", "log2"],
                      "scaler": [StandardScaler(), MinMaxScaler()]}

    param_grid_Log_reg = {"logistic__penalty": ['l2', 'elasticnet', 'none', 'l1'],
                          "logistic__solver": ["newton-cg", "lbfgs", "liblinear", 'saga'],
                          "logistic__C": [100, 10, 1.0, 0.1, 0.01],
                          "logistic__multi_class": ["auto", "ovr", "multinomial"],
                          "logistic__max_iter": [100, 1000, 10000],
                          "scaler": [StandardScaler(), MinMaxScaler()]}

    param_grid_knn = {"knn__n_neighbors": list(range(1, 50)), "scaler": [StandardScaler(), MinMaxScaler()]}

    # cross validation by GridSearchCV
    classifier_RFC = GridSearchCV(estimator=pipe_RFC, param_grid=param_grid_RFC, cv=5, verbose=1, scoring="recall")
    classifier_Log_reg = GridSearchCV(estimator=pipe_log_reg, param_grid=param_grid_Log_reg, cv=5, verbose=1,
                                      scoring="recall")
    classifier_knn = GridSearchCV(estimator=pipe_KNN, param_grid=param_grid_knn, cv=5, verbose=1, scoring="recall")

    grid_search_RFC = classifier_RFC.fit(train_x_val, train_y_val)
    grid_search_Log_reg = classifier_Log_reg.fit(train_x_val, train_y_val)
    grid_search_knn = classifier_knn.fit(train_x_val, train_y_val)

    # Print best score and best hyperparameter
    print(f"Best Score RFC: {grid_search_RFC.best_score_:.4f} using {grid_search_RFC.best_params_}")
    print(f"Best Score Log_reg: {grid_search_Log_reg.best_score_:.4f} using {grid_search_Log_reg.best_params_}")
    print(f"Best Score KNN: {grid_search_knn.best_score_:.4f} using {grid_search_knn.best_params_}")

    # evaluate model
    evaluate_model(classifier_Log_reg, X_test_val, y_test_val)
    evaluate_model(classifier_RFC, X_test_val, y_test_val)
    evaluate_model(classifier_knn, X_test_val, y_test_val)


#tune_hyperparameter(X_train_val, y_train_val)


def evaluate_on_decided_hyperparameter(test_x, test_y):
    """
    Evaluate on decided hyperparameter for all models
    :param test_x: DataFrame
    :param test_y: DataFramed
    :return: pipeline
    """
    pipe_logistic = Pipeline([("scaler", MinMaxScaler()), ("logistic", LogisticRegression(penalty="none",
                             multi_class='ovr', solver='saga', max_iter=10000, C=1.0))])
    pipe_logistic.fit(test_x, test_y)

    pipe_knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=15))])
    pipe_knn.fit(test_x, test_y)

    pipe_rfc = Pipeline([("scaler", StandardScaler()), ("rfc", RandomForestClassifier(n_estimators=150,
                        criterion="gini", max_features="log2"))])
    pipe_rfc.fit(test_x, test_y)

    classifier_VC = VotingClassifier([('logistic', pipe_knn),
                                      ('rfc', pipe_rfc),
                                      ('knn', pipe_logistic)], voting="hard")
    classifier_VC.fit(test_x, test_y)

    for clf, label in zip([pipe_logistic, pipe_knn, pipe_rfc, classifier_VC],
                          ['Logistic Regression', 'KNN', 'Random Forest', "Ensemble"]):
        scores = cross_val_score(clf, test_x, test_y, scoring='recall', cv=5)
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    return pipe_logistic, pipe_knn, pipe_rfc


pipe_logistic, pipe_knn, pipe_rfc = evaluate_on_decided_hyperparameter(X_train_val, y_train_val)
evaluate_model(pipe_logistic, X_test_val, y_test_val)
evaluate_model(pipe_knn, X_test_val, y_test_val)
evaluate_model(pipe_rfc, X_test_val, y_test_val)


# Välj bästa modellen, träna modellen på X_train, y_train, träna på X_test, y_test
def model(x_training, y_training):
    """
    :param x_training: DataFrame
    :param y_training: DataFrame
    :return: model
    """
    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(x_training, y_training)

    return model


model_train = model(X_train, y_train)
evaluate_model(model_train, X_test, y_test)
model_test = model(X_test, y_test)
evaluate_model(model_test, X_test_val, y_test_val)


# 2.7 "Deploy"
def reduced_data(df):
    """
    :param df: DataFrame
    :return: DataFrame
    """
    df_100_row = df.sample(n=100, random_state=42).drop(columns="cardio")
    df_100_row.to_csv("Data/test_samples.csv", sep=";", index=False)

    return df_100_row


df_100_rows = reduced_data(df_cardio_encode2)
df_cardio2_reduced = df_cardio_encode2[~df_cardio_encode2.isin(df_100_rows)].dropna()


def save_model(df):
    """
    :param df: DataFrame
    :return: None
    """
    x, y = split_to_y_x(df, "cardio")
    X_train, X_test, y_train, y_test = split_train_test(x, y, 0.33)
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=15))])
    pipe.fit(X_train, y_train)

    # save the model to disk
    filename = 'Model/model.pkl'
    joblib.dump(pipe, open(filename, 'wb'))


save_model(df_cardio2_reduced)
