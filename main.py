import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)


def process_message(text):
    text = str(text).lower()
    return text

def process_label(text):
    if text == "ham":
        return 0
    return 1

def prepare_data(df):
    df["processed"] = [process_message(msg) for msg in df["message"]]
    df["mapped"] = [process_label(label) for label in df["category"]]

    inputs = df["processed"]
    outputs = df["mapped"]

    input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42, stratify=outputs)
    return (input_train, input_test, output_train, output_test)

def prepare_model(df):
    input_train, input_test, output_train, output_test = prepare_data(df)
    pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),('classifier', MultinomialNB())])
    
    pipeline.fit(input_train, output_train)
    
    return (pipeline, input_test, output_test)
    

def test_model(df):
    pipeline, input_test, output_test = prepare_model(df)
    output_predictions = pipeline.predict(input_test)
    
    print(output_predictions)


def main():
    
    df = pd.read_csv('/Users/ryanfriess/Desktop/projects/spam-classification/data/messages.csv')
    # print(df["Category"])

    df.columns = [col.lower() for col in df.columns]

    # print(df.columns)
    # label_list = []
    # for label in df["Category"]:
    #     label_list.append(label)
    
    test_model(df)
    
main()