# Set working directory and load data
setwd("/Users/romyl/OneDrive/Desktop/Bachelor Thesis")
df <- read.csv("amazon.csv")

# Load necessary libraries
library(stringr)
library(dplyr)
library(tidytext)
library(sentimentr)
library(ggplot2)
update.packages(ask = FALSE)

# Data Cleaning

# Check for unusual strings in the `rating` column
special_chars_pattern <- "[!@#$%^&*(),.?\":{}|<>]"
df$rating[str_detect(df$rating, special_chars_pattern) & df$product_id == "B08L12N5H1"] <- 4.2

# Replace empty strings with NA and remove rows with missing `rating_count`
df[df == ""] <- NA
df <- df[!is.na(df$rating_count), ]

# Convert `discounted_price` and `actual_price` columns to numeric after removing currency symbols
df$discounted_price <- as.numeric(gsub("[₹,]", "", df$discounted_price))
df$actual_price <- as.numeric(gsub("[₹,]", "", df$actual_price))

# Convert `rating` and `rating_count` columns to numeric
df$rating <- as.numeric(gsub("[^0-9.]", "", df$rating))
df$rating_count <- as.numeric(gsub("[^0-9]", "", df$rating_count))

# Extract Main and Subcategories from the Category Column
category_split <- strsplit(df$category, "\\|")
category_df <- do.call(rbind, lapply(category_split, function(x) c(x[1], x[2])))
colnames(category_df) <- c("main_category", "sub_category")
df$main_category <- category_df[, 1]
df$sub_category <- category_df[, 2]
df$category <- NULL  # Remove original category column

# Add an identifier column to `df` for joining purposes
df <- df %>% mutate(id = row_number())

# Feature Engineering

# Create a rating score category
df$rating_score <- with(df, ifelse(rating < 2.0, "Poor",
                                   ifelse(rating >= 2.0 & rating <= 2.9, "Below Average",
                                          ifelse(rating >= 3.0 & rating <= 3.9, "Average",
                                                 ifelse(rating >= 4.0 & rating <= 4.9, "Good", "Excellent")))))

# Combine rating and rating_count into a weighted rating score
min_rating_count <- mean(df$rating_count, na.rm = TRUE)
df$weighted_rating <- with(df, ((rating * rating_count) + (min_rating_count * mean(rating, na.rm = TRUE))) / (rating_count + min_rating_count))

# Target Encoding for `product_name` and `product_id`
product_name_encoding <- df %>%
  group_by(product_name) %>%
  summarize(product_name_avg_rating = mean(rating, na.rm = TRUE))

product_id_encoding <- df %>%
  group_by(product_id) %>%
  summarize(product_id_avg_rating = mean(rating, na.rm = TRUE))

df <- left_join(df, product_name_encoding, by = "product_name")
df <- left_join(df, product_id_encoding, by = "product_id")

# Convert `discount_percentage` to numeric by removing the "%" sign and dividing by 100 to get the actual discount proportion
df$discount_percentage <- as.numeric(gsub("%", "", df$discount_percentage)) / 100

# Verify the transformation
summary(df$discount_percentage)

# Load necessary libraries
library(dplyr)
library(sentimentr)
library(tidytext)

# Process review content and title to sentences, calculate sentiment scores
sentiment_scores_content <- sentiment(get_sentences(df$review_content)) %>%
  group_by(element_id) %>%
  summarize(content_sentiment_score = mean(sentiment, na.rm = TRUE)) %>%
  mutate(id = row_number())

sentiment_scores_title <- sentiment(get_sentences(df$review_title)) %>%
  group_by(element_id) %>%
  summarize(title_sentiment_score = mean(sentiment, na.rm = TRUE)) %>%
  mutate(id = row_number())

# Merge sentiment scores back into the original data frame
df <- df %>%
  left_join(sentiment_scores_content %>% select(id, content_sentiment_score), by = "id") %>%
  left_join(sentiment_scores_title %>% select(id, title_sentiment_score), by = "id") %>%
  mutate(
    # Calculate combined sentiment score
    combined_sentiment_score = rowMeans(cbind(content_sentiment_score, title_sentiment_score), na.rm = TRUE),
    
    # Normalize content, title, and combined sentiment scores to a 0-1 range
    normalized_content_sentiment_score = (content_sentiment_score - min(content_sentiment_score, na.rm = TRUE)) / 
      (max(content_sentiment_score, na.rm = TRUE) - min(content_sentiment_score, na.rm = TRUE)),
    normalized_title_sentiment_score = (title_sentiment_score - min(title_sentiment_score, na.rm = TRUE)) / 
      (max(title_sentiment_score, na.rm = TRUE) - min(title_sentiment_score, na.rm = TRUE)),
    normalized_combined_sentiment_score = (combined_sentiment_score - min(combined_sentiment_score, na.rm = TRUE)) / 
      (max(combined_sentiment_score, na.rm = TRUE) - min(combined_sentiment_score, na.rm = TRUE)),
    
    # Scale normalized scores to a 1-5 range
    scaled_content_sentiment_score = normalized_content_sentiment_score * 4 + 1,
    scaled_title_sentiment_score = normalized_title_sentiment_score * 4 + 1,
    scaled_combined_sentiment_score = normalized_combined_sentiment_score * 4 + 1
  ) %>%
  # Categorize sentiment scores into 5 sentiment categories
  mutate(
    content_sentiment_category = case_when(
      scaled_content_sentiment_score < 2 ~ "Strong Negative",
      scaled_content_sentiment_score < 3 ~ "Moderately Negative",
      scaled_content_sentiment_score < 4 ~ "Neutral",
      scaled_content_sentiment_score < 5 ~ "Moderately Positive",
      scaled_content_sentiment_score == 5 ~ "Strong Positive"
    ),
    title_sentiment_category = case_when(
      scaled_title_sentiment_score < 2 ~ "Strong Negative",
      scaled_title_sentiment_score < 3 ~ "Moderately Negative",
      scaled_title_sentiment_score < 4 ~ "Neutral",
      scaled_title_sentiment_score < 5 ~ "Moderately Positive",
      scaled_title_sentiment_score == 5 ~ "Strong Positive"
    ),
    combined_sentiment_category = case_when(
      scaled_combined_sentiment_score < 2 ~ "Strong Negative",
      scaled_combined_sentiment_score < 3 ~ "Moderately Negative",
      scaled_combined_sentiment_score < 4 ~ "Neutral",
      scaled_combined_sentiment_score < 5 ~ "Moderately Positive",
      scaled_combined_sentiment_score == 5 ~ "Strong Positive"
    )
  )

# Check the first few rows to verify the sentiment scores and categories
head(df %>% select(id, content_sentiment_score, content_sentiment_category, 
                   title_sentiment_score, title_sentiment_category, 
                   combined_sentiment_score, combined_sentiment_category))

# EDA
library(dplyr)
library(ggplot2)
library(tidyverse)
library(corrplot)
update.packages(ask = FALSE)

# Display the structure of the data
str(df)

# Summary statistics
summary(df)

# Check for missing values
colSums(is.na(df))

# Calculate correlations while keeping 1 correlations and excluding id column
numeric_data <- df %>% select(where(is.numeric)) %>% select(-id, -normalized_combined_sentiment_score , -normalized_title_sentiment_score ,-normalized_content_sentiment_score, -scaled_combined_sentiment_score, -scaled_title_sentiment_score, -scaled_content_sentiment_score, -product_name_avg_rating, -product_id_avg_rating )  # Remove only the id column
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# Plot the correlation matrix with the original values of 1 retained
library(corrplot)
corrplot::corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45, na.label = " ")

# Distribution of Rating Count
ggplot(df, aes(x = rating_count)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Rating Count", x = "Rating Count", y = "Count") +
  theme_minimal()

# Distribution of Discounted Price
ggplot(df, aes(x = discounted_price)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  labs(title = "Distribution of Discounted Price", x = "Discounted Price", y = "Count") +
  theme_minimal()

# Distribution of Combined Sentiment Score
ggplot(df, aes(x = combined_sentiment_score)) +
  geom_histogram(bins = 30, fill = "darkblue", color = "black") +
  labs(title = "Distribution of Combined Sentiment Score", x = "Combined Sentiment Score", y = "Count") +
  theme_minimal()


# Average Combined Sentiment Score by Main Category (Barplot)
category_sentiment <- df %>%
  group_by(main_category) %>%
  summarize(avg_combined_sentiment = mean(combined_sentiment_score, na.rm = TRUE))

ggplot(category_sentiment, aes(x = reorder(main_category, -avg_combined_sentiment), y = avg_combined_sentiment)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  coord_flip() +
  labs(title = "Distribution of Average Combined Sentiment Score by Main Category", x = "Main Category", y = "Average Combined Sentiment Score") +
  theme_minimal()

# Average Combined Sentiment Score by Sub-Category (Barplot)
sub_category_sentiment <- df %>%
  group_by(sub_category) %>%
  summarize(avg_combined_sentiment = mean(combined_sentiment_score, na.rm = TRUE))

ggplot(sub_category_sentiment, aes(x = reorder(sub_category, -avg_combined_sentiment), y = avg_combined_sentiment)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  coord_flip() +
  labs(title = "Distribution of Combined Sentiment Scores by Sub-Category", x = "Sub-Category", y = "Average Combined Sentiment Score") +
  theme_minimal()

# Average Content Sentiment Score by Main Category (Barplot)
main_category_content_sentiment <- df %>%
  group_by(main_category) %>%
  summarize(avg_content_sentiment = mean(content_sentiment_score, na.rm = TRUE))

ggplot(main_category_content_sentiment, aes(x = reorder(main_category, -avg_content_sentiment), y = avg_content_sentiment)) +
  geom_bar(stat = "identity", fill = "orange") +
  coord_flip() +
  labs(title = "Distribution of Content Sentiment Scores by Main Category", x = "Main Category", y = "Average Content Sentiment Score") +
  theme_minimal()

# Average Content Sentiment Score by Sub-Category (Barplot)
sub_category_content_sentiment <- df %>%
  group_by(sub_category) %>%
  summarize(avg_content_sentiment = mean(content_sentiment_score, na.rm = TRUE))

ggplot(sub_category_content_sentiment, aes(x = reorder(sub_category, -avg_content_sentiment), y = avg_content_sentiment)) +
  geom_bar(stat = "identity", fill = "orange") +
  coord_flip() +
  labs(title = "Distribution of Content Sentiment Scores by Sub-Category", x = "Sub-Category", y = "Average Content Sentiment Score") +
  theme_minimal()

# Average Title Sentiment Score by Main Category (Barplot)
main_category_title_sentiment <- df %>%
  group_by(main_category) %>%
  summarize(avg_title_sentiment = mean(title_sentiment_score, na.rm = TRUE))

ggplot(main_category_title_sentiment, aes(x = reorder(main_category, -avg_title_sentiment), y = avg_title_sentiment)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Distribution of Title Sentiment Scores by Main Category", x = "Main Category", y = "Average Title Sentiment Score") +
  theme_minimal()

# Average Title Sentiment Score by Sub-Category (Barplot)
sub_category_title_sentiment <- df %>%
  group_by(sub_category) %>%
  summarize(avg_title_sentiment = mean(title_sentiment_score, na.rm = TRUE))

ggplot(sub_category_title_sentiment, aes(x = reorder(sub_category, -avg_title_sentiment), y = avg_title_sentiment)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Distribution of Title Sentiment Scores by Sub-Category", x = "Sub-Category", y = "Average Title Sentiment Score") +
  theme_minimal()


# Load required libraries
library(dplyr)
library(ggplot2)

# Exploring Relationships

# Summary Statistics for Key Variables
summary_stats <- df %>%
  summarize(
    avg_rating_count = mean(rating_count, na.rm = TRUE),
    avg_discounted_price = mean(discounted_price, na.rm = TRUE),
    avg_combined_sentiment = mean(combined_sentiment_score, na.rm = TRUE),
    avg_content_sentiment = mean(content_sentiment_score, na.rm = TRUE),
    avg_title_sentiment = mean(title_sentiment_score, na.rm = TRUE)
  )
print(summary_stats)

# Set factor levels for sentiment categories for consistent ordering in plots
sentiment_levels <- c("Strong Negative", "Moderately Negative", "Neutral", "Moderately Positive", "Strong Positive")

# Binning Sentiment Scores into Categories
df <- df %>%
  mutate(
    combined_sentiment_category = factor(case_when(
      combined_sentiment_score < -0.6 ~ "Strong Negative",
      combined_sentiment_score >= -0.6 & combined_sentiment_score < -0.1 ~ "Moderately Negative",
      combined_sentiment_score >= -0.1 & combined_sentiment_score <= 0.1 ~ "Neutral",
      combined_sentiment_score > 0.1 & combined_sentiment_score <= 0.6 ~ "Moderately Positive",
      combined_sentiment_score > 0.6 ~ "Strong Positive"
    ), levels = sentiment_levels),
    
    content_sentiment_category = factor(case_when(
      content_sentiment_score < -0.6 ~ "Strong Negative",
      content_sentiment_score >= -0.6 & content_sentiment_score < -0.1 ~ "Moderately Negative",
      content_sentiment_score >= -0.1 & content_sentiment_score <= 0.1 ~ "Neutral",
      content_sentiment_score > 0.1 & content_sentiment_score <= 0.6 ~ "Moderately Positive",
      content_sentiment_score > 0.6 ~ "Strong Positive"
    ), levels = sentiment_levels),
    
    title_sentiment_category = factor(case_when(
      title_sentiment_score < -0.6 ~ "Strong Negative",
      title_sentiment_score >= -0.6 & title_sentiment_score < -0.1 ~ "Moderately Negative",
      title_sentiment_score >= -0.1 & title_sentiment_score <= 0.1 ~ "Neutral",
      title_sentiment_score > 0.1 & title_sentiment_score <= 0.6 ~ "Moderately Positive",
      title_sentiment_score > 0.6 ~ "Strong Positive"
    ), levels = sentiment_levels)
  )

# Distribution of Sentiment Categories with Correct Ordering
ggplot(df, aes(x = combined_sentiment_category)) +
  geom_bar(fill = "lightblue", color = "black") +
  labs(title = "Distribution of Combined Sentiment Categories", x = "Sentiment Category", y = "Count") +
  theme_minimal()

ggplot(df, aes(x = content_sentiment_category)) +
  geom_bar(fill = "blue", color = "black") +
  labs(title = "Distribution of Content Sentiment Categories", x = "Sentiment Category", y = "Count") +
  theme_minimal()

ggplot(df, aes(x = title_sentiment_category)) +
  geom_bar(fill = "darkblue", color = "black") +
  labs(title = "Distribution of Title Sentiment Categories", x = "Sentiment Category", y = "Count") +
  theme_minimal()

# Sentiment Distribution by Rating
ggplot(df, aes(x = factor(rating), fill = combined_sentiment_category)) +
  geom_bar(position = "fill") +
  labs(title = "Sentiment Distribution by Product Rating", x = "Product Rating", y = "Proportion") +
  theme_minimal() +
  scale_fill_manual(values = c("red", "orange", "gray", "lightblue", "green")) +
  scale_x_discrete(drop = FALSE)  # Ensures all ratings are shown

# Price vs Sentiment Analysis
# Relationship between Combined Sentiment Score and Discounted Price
ggplot(df, aes(x = discounted_price, y = combined_sentiment_score)) +
  geom_point(alpha = 0.5, color = "purple") +
  labs(title = "Relationship between Discounted Price and Combined Sentiment Score", x = "Discounted Price", y = "Combined Sentiment Score") +
  theme_minimal()

# Relationship between Combined Sentiment Score and Actual Price
ggplot(df, aes(x = actual_price, y = combined_sentiment_score)) +
  geom_point(alpha = 0.5, color = "blue") +
  labs(title = "Relationship between Actual Price and Combined Sentiment Score", x = "Actual Price", y = "Combined Sentiment Score") +
  theme_minimal()

# Average Rating Count by Main Category
df %>%
  group_by(main_category) %>%
  summarize(avg_rating_count = mean(rating_count, na.rm = TRUE)) %>%
  ggplot(aes(x = reorder(main_category, -avg_rating_count), y = avg_rating_count)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  coord_flip() +
  labs(title = "Average Rating Count by Main Category", x = "Main Category", y = "Average Rating Count") +
  theme_minimal()

# Average Rating Count by Sub Category
df %>%
  group_by(sub_category) %>%
  summarize(avg_rating_count = mean(rating_count, na.rm = TRUE)) %>%
  ggplot(aes(x = reorder(sub_category, -avg_rating_count), y = avg_rating_count)) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  coord_flip() +
  labs(title = "Average Rating Count by Sub Category", x = "Sub Category", y = "Average Rating Count") +
  theme_minimal()

# Logistic Regression to find out what effect features have on rating_count
# Median rating_count
threshold <- median(df$rating_count, na.rm = TRUE)
df <- df %>% mutate(rating_category = ifelse(rating_count > threshold, "High", "Low"))
df$rating_category <- factor(df$rating_category, levels = c("Low", "High"))

# LR
logit_model <- glm(rating_category ~ combined_sentiment_score + content_sentiment_score + 
                     title_sentiment_score + rating, 
                   data = df, family = binomial)

summary(logit_model)


lm_model <- lm(rating ~ content_sentiment_score + title_sentiment_score + rating_count + discount_percentage + actual_price + discounted_price, data = df)

summary(lm_model)

if (!require("stargazer")) install.packages("stargazer")
library(stargazer)

stargazer(lm_model, type = "text", title = "LM-Result")


# Ridge and Lasso
# Install and load required packages if not already installed
if (!require("glmnet")) install.packages("glmnet", dependencies = TRUE)
library(dplyr)
library(caret)
library(glmnet)

# Step 1: Create binary target variable `rating_category` based on `rating_count`
if ("rating_count" %in% colnames(df)) {
  threshold <- median(df$rating_count, na.rm = TRUE)
  df <- df %>%
    mutate(rating_category = ifelse(rating_count > threshold, "High", "Low"))
} else {
  stop("`rating_count` column is missing in the dataset.")
}

# Convert `rating_category` to a factor for glmnet compatibility
df$rating_category <- factor(df$rating_category, levels = c("Low", "High"))

# Step 2: Select only the relevant features and the target variable
df <- df %>% 
  select(discounted_price, actual_price, rating, weighted_rating, 
         content_sentiment_score, title_sentiment_score,combined_sentiment_score, rating_category)

# Step 3: Split the data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(df$rating_category, p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]

# Step 4: Prepare data for glmnet by creating model matrices
y_train <- as.numeric(trainData$rating_category) - 1  # Convert High/Low to binary 1/0
x_train <- model.matrix(~ 0 + ., data = trainData %>% select(-rating_category))
x_test <- model.matrix(~ 0 + ., data = testData %>% select(-rating_category))

# Step 5: Fit Lasso and Ridge models

# Lasso model (alpha = 1 for Lasso)
lasso_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)

# Predict and evaluate Lasso model
predicted_probs_lasso <- predict(lasso_model, newx = x_test, s = "lambda.min", type = "response")
predicted_classes_lasso <- ifelse(predicted_probs_lasso > 0.5, "High", "Low")
confusion_matrix_lasso <- confusionMatrix(factor(predicted_classes_lasso, levels = c("Low", "High")), testData$rating_category)
cat("Lasso Model Confusion Matrix:\n")
print(confusion_matrix_lasso)

# Ridge model (alpha = 0 for Ridge)
ridge_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0)

# Predict and evaluate Ridge model
predicted_probs_ridge <- predict(ridge_model, newx = x_test, s = "lambda.min", type = "response")
predicted_classes_ridge <- ifelse(predicted_probs_ridge > 0.5, "High", "Low")
confusion_matrix_ridge <- confusionMatrix(factor(predicted_classes_ridge, levels = c("Low", "High")), testData$rating_category)
cat("Ridge Model Confusion Matrix:\n")
print(confusion_matrix_ridge)

# Hyperparameter Tuning
lambda_grid <- 10^seq(-4, 2, length = 100)
lasso_cv <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = lambda_grid)
best_lambda_lasso <- lasso_cv$lambda.min

cat("Best lambda for Lasso: ", best_lambda_lasso, "\n")
predicted_probs_lasso <- predict(lasso_cv, newx = x_test, s = "lambda.min", type = "response")

predicted_classes_lasso <- ifelse(predicted_probs_lasso > 0.5, "High", "Low")
confusion_matrix_lasso <- confusionMatrix(factor(predicted_classes_lasso, levels = c("Low", "High")), testData$rating_category)

cat("Lasso Model Confusion Matrix:\n")
print(confusion_matrix_lasso)

ridge_cv <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0, lambda = lambda_grid)
best_lambda_ridge <- ridge_cv$lambda.min
cat("Best lambda for Ridge: ", best_lambda_ridge, "\n")

predicted_probs_ridge <- predict(ridge_cv, newx = x_test, s = "lambda.min", type = "response")
predicted_classes_ridge <- ifelse(predicted_probs_ridge > 0.5, "High", "Low")

confusion_matrix_ridge <- confusionMatrix(factor(predicted_classes_ridge, levels = c("Low", "High")), testData$rating_category)
cat("Ridge Model Confusion Matrix:\n")

print(confusion_matrix_ridge)

# Random Forests
# Load necessary libraries once at the beginning
library(caret)
library(randomForest)

# Define the data selection and preprocessing steps
model_data <- df %>%
  select(rating_count, rating, discounted_price, combined_sentiment_score, content_sentiment_score, title_sentiment_score, main_category, sub_category) %>%
  mutate(main_category = as.factor(main_category),
         sub_category = as.factor(sub_category))

# Train-Test Split
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(model_data$rating_count, p = 0.75, list = FALSE)
trainData <- model_data[trainIndex, ]
testData <- model_data[-trainIndex, ]

# Adjust factor levels in test data
trainData$main_category <- fct_na_value_to_level(trainData$main_category, level = "Other")
trainData$sub_category <- fct_na_value_to_level(trainData$sub_category, level = "Other")
testData$main_category <- factor(testData$main_category, levels = levels(trainData$main_category))
testData$sub_category <- factor(testData$sub_category, levels = levels(trainData$sub_category))

# Define the formula for the model once and reuse it
formula <- rating_count ~ rating + discounted_price + combined_sentiment_score + content_sentiment_score + title_sentiment_score + main_category + sub_category

# Initial Random Forest model for baseline performance
set.seed(123)
rf_model <- randomForest(formula, data = trainData, ntree = 100, mtry = 3, importance = TRUE)

# Model Evaluation on Test Data
rf_predictions <- predict(rf_model, newdata = testData)
rmse_value <- sqrt(mean((rf_predictions - testData$rating_count)^2))
mae_value <- mean(abs(rf_predictions - testData$rating_count))
cat("Initial Random Forest Performance:\nRMSE:", rmse_value, "\nMAE:", mae_value, "\n")

# Hyperparameter Tuning Setup
control <- trainControl(method = "cv", number = 5, search = "grid")  # Define control once

# Broad Hyperparameter Grid for Initial Tuning
tuneGrid_broad <- expand.grid(
  mtry = seq(2, ncol(trainData) / 2, by = 2),
  splitrule = "variance",
  min.node.size = c(1, 5, 10, 15, 20)
)

# First Tuning Experiment (Broad Search)
set.seed(123)
rf_tuned_broad <- train(
  formula, data = trainData, method = "ranger",
  tuneGrid = tuneGrid_broad, trControl = control
)

# Refine Tuning Grid Around Best Parameters
best_mtry <- rf_tuned_broad$bestTune$mtry
best_node_size <- rf_tuned_broad$bestTune$min.node.size
tuneGrid_refined <- expand.grid(
  mtry = seq(max(1, best_mtry - 2), best_mtry + 2, by = 1),
  splitrule = "variance",
  min.node.size = seq(max(1, best_node_size - 2), best_node_size + 2, by = 1)
)

# Refined Tuning
set.seed(123)
rf_tuned_refined <- train(
  formula, data = trainData, method = "ranger",
  tuneGrid = tuneGrid_refined, trControl = control
)

# Output Best Model Parameters
print(rf_tuned_refined)
best_params <- rf_tuned_refined$bestTune
cat("Best Parameters: mtry =", best_params$mtry, "and min.node.size =", best_params$min.node.size, "\n")

# Evaluation on Test Data with Best Tuned Model
test_predictions_tuned <- predict(rf_tuned_refined, newdata = testData)
rmse_tuned <- sqrt(mean((test_predictions_tuned - testData$rating_count)^2))
mae_tuned <- mean(abs(test_predictions_tuned - testData$rating_count))
cat("Tuned Model Performance:\nRMSE:", rmse_tuned, "\nMAE:", mae_tuned, "\n")

# Feature Importance Plot
varImpPlot(rf_model)  # Initial feature importance plot
importance(rf_model) 


# KNN
library(class)      
library(caret)      
library(dplyr)
library(lattice)
update.packages(ask = FALSE)

# Assume 'df' is the dataset
model_data <- df %>%
  select(rating_count, rating, discounted_price, combined_sentiment_score, content_sentiment_score, title_sentiment_score, main_category, sub_category) %>%
  mutate(
    main_category = as.numeric(as.factor(main_category)),  # Convert categorical to numeric
    sub_category = as.numeric(as.factor(sub_category))
  )

# Normalize the data (important for KNN)
preProc <- preProcess(model_data, method = c("center", "scale"))
model_data <- predict(preProc, model_data)

set.seed(123)  # Ensure reproducibility
trainIndex <- createDataPartition(model_data$rating_count, p = 0.75, list = FALSE)

trainData <- model_data[trainIndex, ]
testData <- model_data[-trainIndex, ]

# Define features and target
x_train <- trainData %>% select(-rating_count)
y_train <- trainData$rating_count
x_test <- testData %>% select(-rating_count)
y_test <- testData$rating_count


# Choose the number of neighbors (k)
k <- 5  # Example value; tune this based on performance

# Fit the model
knn_model <- knn(
  train = x_train,
  test = x_test,
  cl = y_train,
  k = k
)


# Convert predictions to numeric if needed
knn_predictions <- as.numeric(as.character(knn_model))

# Calculate RMSE and MAE
rmse <- sqrt(mean((knn_predictions - y_test)^2))
mae <- mean(abs(knn_predictions - y_test))

cat("KNN RMSE:", rmse, "\n")
cat("KNN MAE:", mae, "\n")

#Hyperparameter Tuning
set.seed(123)

# Define a grid of k values to test
k_values <- seq(1, 20, by = 2)

# Create a control function for CV
train_control <- trainControl(method = "cv", number = 5)

# Train the model with tuning
knn_tuned <- train(
  x = x_train,
  y = y_train,
  method = "knn",
  tuneGrid = expand.grid(k = k_values),
  trControl = train_control
)

# Display the best k value
print(knn_tuned)
cat("Best k:", knn_tuned$bestTune$k, "\n")

#Evaluation
# Use the tuned k value
best_k <- knn_tuned$bestTune$k

# Refit KNN with best k
knn_final <- knn(
  train = x_train,
  test = x_test,
  cl = y_train,
  k = best_k
)

# Evaluate
knn_predictions_tuned <- as.numeric(as.character(knn_final))
rmse_tuned <- sqrt(mean((knn_predictions_tuned - y_test)^2))
mae_tuned <- mean(abs(knn_predictions_tuned - y_test))

cat("Tuned KNN RMSE:", rmse_tuned, "\n")
cat("Tuned KNN MAE:", mae_tuned, "\n")

# Support Vector Machines
install.packages("e1071")  # For SVM
install.packages("caret")  # For preprocessing and evaluation
library(e1071)
library(caret)
library(dplyr)





