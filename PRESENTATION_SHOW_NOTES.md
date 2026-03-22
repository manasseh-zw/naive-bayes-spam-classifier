# Group 3 Presentation Script: Naive Bayes SMS Classification

This script is written as what we will actually say in class. It is organized by step so we do not get lost while presenting.

## Introduction

Good day everyone. We are Group 3, and today we are presenting Naive Bayes classification. To demonstrate it clearly, we chose an SMS spam classification problem. Our model reads a message and predicts whether it is spam or ham. Ham means a genuine normal message, not spam.

This is a binary classification task because there are only two possible outputs. That makes it a very good fit for Naive Bayes.

## Step 1: What Naive Bayes Means in This Project

Before we run the pipeline, we explain the core idea in simple language. Bayes theorem helps us answer this question: given a message, what is the probability that it belongs to spam, and what is the probability that it belongs to ham?

Naive Bayes keeps this idea and adds one simplifying assumption: each word contributes independently to the final class decision. The assumption is not perfectly true in real language, but it works very well in many text classification tasks.

In practical terms, the model combines base class probability with evidence from each word, then predicts the class with the higher score.

## Step 2: Why We Use scikit-learn

We use scikit-learn because it is reliable, well-tested, and widely used in machine learning. It gives us strong tools for splitting data, building pipelines, tuning models, and evaluating performance. This helps us focus on the actual machine learning logic instead of low-level implementation details.

It is also transparent. We can inspect model settings, probabilities, and feature behavior, so we can explain our results clearly instead of presenting a black box.

## Step 3: Setup and Imports

At this stage, we prepare the environment and import all required libraries. We import data tools, visualization tools, dataset loading tools, and modeling tools. This step is simply preparation so the rest of the notebook runs smoothly.

## Step 4: Data Collection

Now we load one fixed dataset directly from Hugging Face. We intentionally use one source so the experiment remains consistent and easy to reproduce.

Then we standardize labels to binary format: spam is 1 and ham is 0. We also show that the dataset has more ham than spam, which is common in real spam detection problems.

## Step 5: Exploratory Data Analysis

In this step, we inspect the data before modeling. We look at class distribution and text-length patterns such as character length and word count.

What we tell the class is simple: this gives us confidence that our dataset is usable and helps us understand visible differences between spam and ham messages.

## Step 6: Text Cleaning and Preprocessing

Now we clean the messages by lowercasing text, removing links, removing punctuation and extra characters, and normalizing spaces.

Why are we doing this? Because noisy text can reduce model quality. Cleaning helps the model focus on meaningful language patterns instead of irrelevant symbols and formatting.

## Step 7: Train, Validation, and Test Split

Here we split the data into three parts.

Train set is used to learn patterns.
Validation set is used to choose the best model settings.
Test set is used only at the end for final unbiased evaluation.

We use stratified splitting so class proportions remain consistent across all splits.

## Step 8: TF-IDF Vectorization

Before Naive Bayes can learn, text must be converted into numbers. TF-IDF is the method we use for that conversion.

TF means Term Frequency, which is how often a word appears in one message.

IDF means Inverse Document Frequency, which reduces the weight of very common words that appear everywhere and gives more importance to words that are more distinctive.

So TF-IDF helps the model focus on informative words instead of common filler words. In simple terms, it turns each message into a weighted numeric vector, and those weights become the input features for Naive Bayes.

## Step 9: Model Selection and Training

This is the main modeling step. We build a pipeline that combines TF-IDF vectorization with Naive Bayes so both stages run together in one clean workflow.

We compare two Naive Bayes variants and run grid search with cross-validation. This means we are not manually guessing settings. Instead, we systematically test combinations and choose the best one based on performance.

## Step 10: Understanding the Evaluation Metrics

Before we present results, we explain each metric in plain English.

Accuracy is the overall percentage of correct predictions.

Precision answers: out of all messages predicted as spam, how many were truly spam.

Recall answers: out of all real spam messages, how many did we successfully catch.

F1-score is a balance score between precision and recall. We use it when both false alarms and missed spam matter.

ROC-AUC tells us how well the model separates spam and ham across many decision thresholds. A value closer to 1 means better separation.

## Step 11: Validation Results

We first evaluate on validation data. The model performs very strongly, with high accuracy and strong precision and recall. At this point we conclude that the selected model is promising and ready for final testing.

## Step 12: Test Results and Visuals

Now we evaluate on unseen test data. Performance remains strong, which means the model generalizes well.

In the confusion matrix, we explain four outcomes: correct ham, false spam alarm, missed spam, and correct spam detection. This helps the audience see not only how accurate the model is, but also the type of mistakes it makes.

In the ROC curve, we show that the model has strong class separation, which supports the numerical performance metrics.

## Step 13: Interpretability

After evaluation, we explain why the model predicts what it predicts.

We show top spam-indicative and ham-indicative tokens, then we break down one sample message into token contributions. This directly connects the model behavior back to the Naive Bayes idea presented at the beginning.

This is where we show the project is explainable, not just accurate.

## Step 14: Live Demo

Finally, we run the live message classifier. We type a message and the notebook predicts spam or ham in real time with probability.

This demonstrates practical usability: the model is not only good in charts and tables, but also works interactively on new user input.

## Conclusion

To conclude, we presented a complete and explainable machine learning workflow: theory, data preparation, model selection, evaluation, interpretation, and live inference.

Our key message is this: Naive Bayes is fast, effective, and easy to explain for text classification, and our SMS spam classifier shows that clearly.
