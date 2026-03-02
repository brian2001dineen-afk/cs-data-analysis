import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    """Copy of DM-Lab05.ipynb

    # Lab 05 - Data Mining
    The goal of this lab is to preprocess a dataset to prepare it for a machine learning task.
    The only libraty you can use is **pandas**.

    We will be using the Titanic dataset. The goal is to predict if the person will survive or not.

    ### Load the dataset
    """

    import pandas as pd

    df = pd.read_csv('https://raw.githubusercontent.com/andvise/DataAnalyticsDatasets/refs/heads/main/titanic.csv')
    # remove irrelevant variables
    df = df.drop(axis=1, labels = ['Name'])


    """### Encode the categorical variable into numerical
    Focus on only those that you'll be using in your machine learning model
    """

    df = df.replace(df['Sex'].unique(), [0, 1])

    """### Split the dataset in 80% training and 20% test."""

    df_train = df.sample(frac=0.8, random_state=1)
    df

    df_test = df.drop(df_train.index)

    # df_test = df.iloc[df_train. != df]
    # print(df_test)
    # print(df_train)
    # print(df_train.index)


    """### Divide the dataset between input variables (X) and output (y).
    Select which columns should be kept for this particular task.

    """

    y_train = df_train['Survived']
    y_test = df_test['Survived']
    X_train = df_train.drop(labels = 'Survived', axis=1)
    X_test = df_test.drop(labels = 'Survived', axis=1)



    """### Scale the data
    Create 2 versions of the datasets, one scaled with a min-max scaler and onw with a standard scaler (using only pandas).

    """

    mins = X_train.min()
    maxs = X_train.max()

    for col in X_train.columns:
        X_train[col] = (X_train[col] - mins[col])/ (maxs[col] - mins[col])
        X_test[col] = (X_test[col] - mins[col])/ (maxs[col] - mins[col])


    # X_train.describe()

    """### What is the baseline for the accuracy?
    What is the minimum accuracy a classifier should reach?
    
    By baseline accuracy, we mean always predicting the class that is most common
    """

    print(f"Baseline accuracy: {1-y_test.mean()}")
    # same:
    lala = 1-sum(y_test == 1)/len(y_test)
    print(f"Baseline accuracy: {lala}")


    """### Create a tree classifier and check the train and test accuracy

    **You can use sklearn**


    """
    print(df)
    return


if __name__ == "__main__":
    app.run()
