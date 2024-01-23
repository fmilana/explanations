import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from augment import oversample
from classifier import MultiLabelProbClassifier


# load entire data
df = pd.read_csv("data/train.csv")
# process embedding strings
df["sentence_embedding"] = df["sentence_embedding"].apply(
    lambda x: np.fromstring(
        x.replace("\n","")
        .replace("[","")
        .replace(",", " ")
        .replace("]",""), sep=" "
    )
)
# create list of review ids for training
review_ids = df["review_id"].unique().tolist()
# set test size
test_size = 5 # arbitrary
# randomly select 5 test ids
test_ids = random.sample(review_ids, test_size)
# create test df based on test ids
test_df = df[df["review_id"].isin(test_ids)]
# remove test ids from review ids
review_ids = [id for id in review_ids if id not in test_ids]
# filter df to exclude test ids
df = df[~df["review_id"].isin(test_ids)]
# initialise group kfold
gkf = GroupKFold(n_splits=len(review_ids))
# initialise classifier
clf = MultiLabelProbClassifier()

thresholds = np.arange(0.1, 1, 0.1)

scores = []
best_thresholds = []
# iterate over splits
for train_indices, validation_indices in gkf.split(df, groups=df['review_id']):
    # split data into training and validation sets
    train_df = df.iloc[train_indices]
    validation_df = df.iloc[validation_indices]
    # prepare training data
    X_train = np.array(train_df["sentence_embedding"].tolist())
    Y_train = np.array(train_df.iloc[:, 7:])
    # oversample minority classes if present
    X_train, Y_train = oversample(X_train, Y_train)
    # prepare validation data
    X_validation = np.array(validation_df["sentence_embedding"].tolist())
    Y_validation = np.array(validation_df.iloc[:, 7:])
    # fit classifier
    clf.fit(X_train, Y_train)
    # predict probabilities on validation set
    Y_prob = clf.predict_proba(X_validation)

    best_score = 0
    best_threshold = 0
    # iterate over thresholds
    for threshold in thresholds:
        # convert probabilities to binary predictions
        Y_pred = (Y_prob > threshold).astype(int)
        # calculate f1 score
        score = f1_score(Y_validation, Y_pred, average="weighted")
        # update best score and threshold if score is better
        if score > best_score:
            best_score = score
            best_threshold = threshold
    # store best threshold and score
    best_thresholds.append(best_threshold)
    scores.append(best_score)


print(f"Best Thresholds: {best_thresholds}")
print(f"Average Best Threshold: {np.mean(best_thresholds)}")
print(f"Best F1 Scores: {scores}")
print(f"Average Best F1 score: {np.mean(scores)}")


# # Train the classifier on the full training set
# X_train_full = np.array(df["sentence_embedding"].tolist())
# Y_train_full = np.array(df.iloc[:, 7:])
# clf.fit(X_train_full, Y_train_full)

# # Prepare test data
# X_test = np.array(test_df["sentence_embedding"].tolist())
# Y_test = np.array(test_df.iloc[:, 7:])

# # Predict probabilities on the test set
# Y_test_prob = clf.predict_proba(X_test)

# # Apply the average best threshold from cross-validation
# average_best_threshold = np.mean(best_thresholds)
# Y_test_pred = (Y_test_prob >= average_best_threshold).astype(int)

# # Evaluate performance
# test_score = f1_score(Y_test, Y_test_pred, average="weighted")

# print(f"Test F1 Score using average best threshold ({average_best_threshold}): {test_score}")



# low threshold issue:
# https://chat.openai.com/share/1f5a8731-e515-4360-9eec-addf66cb7a9f