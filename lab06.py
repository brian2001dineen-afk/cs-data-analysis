import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bag of Words and feature reduction

    In this notebook, we will see how to develop a machine learning model on textual inputs. The goal of the project is to classify the sentiment of movie reviews.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's import useful Python packages.
    """)
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dataset

    Stanford University researchers have taken 50,000 movie reviews from [IMDB](https://www.imdb.com/) labelled them as either positive or negative and [made them available](http://ai.stanford.edu/~amaas/data/sentiment/). We created a dataset with 2,500 positive reviews and the 2,500 negative reviews

    Let's read the dataset available on: https://github.com/andvise/DataAnalyticsDatasets/blob/8e8f6475f49d2a587e4f5c76cdf0b011b22c6ac1/dataset_5000_reviews.csv
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("https://github.com/andvise/DataAnalyticsDatasets/blob/8e8f6475f49d2a587e4f5c76cdf0b011b22c6ac1/dataset_5000_reviews.csv?raw=true")
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.iloc[0,0]
    return


@app.cell
def _(df):
    df.tail()
    return


@app.cell
def _(df):
    df['Sentiment'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's encode the labels as 0 and 1 using the *LabelEncoder*.

    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
    """)
    return

@app.cell()
def _(df, ):
    from sklearn.preprocessing import LabelEncoder
    # encode the labels as 0 and 1 using the LabelEncoder
    le = LabelEncoder()
    # le.fit(df["Sentiment"])
    # le.classes_
    y= le.fit_transform(df["Sentiment"])
    X = df["Review"]
    return y, X


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Split the dataset into training set (80%) and test set (20%).



    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """)
    return

@app.cell
def _(df, y, X):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classification Task
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *   Create a machine learning approach using Count Vectorizer and KNN Classifier.
    *   Fit the the best model on the full training set.
    *   Evaluate its performance on the test set.


    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html


    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """)
    return


@app.cell
def _(X, y, X_train, X_test, y_train, y_test):

    from sklearn.feature_extraction.text import CountVectorizer
    # TODO: why stopwords?
    # many attributes as well
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,1))
    from sklearn.decomposition import PCA

    _X_train= vectorizer.fit_transform(X_train)

    # pca
    pca = PCA(n_components = 100)
    _X_train = pca.fit_transform(_X_train)
    

    # do same for testset
    _X_test= vectorizer.transform(X_test)
    _X_test = pca.transform(_X_test)

    from sklearn.neighbors import KNeighborsClassifier
    n = 5
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(_X_train, y_train)
    knn.score(_X_test, y_test)

    # TODO: write conclusion

    return

@app.cell
def _():
    # svd
    return


if __name__ == "__main__":
    app.run()
