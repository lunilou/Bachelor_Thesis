df <- read.csv("processed_amazon_data.csv")

library(dplyr)
library(car)
library(broom)
library(ggplot2)
library(knitr)
library(kableExtra)
library(glmnet)
library(caret)
library(pROC)

set.seed(123)

# ------------------ Data Preparation ------------------
model_data <- df %>%
  mutate(
    rating_binary = factor(ifelse(rating_count > median(rating_count, na.rm = TRUE), "High", "Low"), levels = c("Low", "High"))
  ) %>%
  select(
    rating_binary, content_sentiment_score, title_sentiment_score, weighted_rating, discounted_price, discount_percentage, actual_price
  )

model_data <- na.omit(model_data)

trainIndex <- createDataPartition(model_data$rating_binary, p = 0.8, list = FALSE)
trainData <- model_data[trainIndex, ]
testData <- model_data[-trainIndex, ]

trainData$rating_binary <- factor(trainData$rating_binary, levels = c("Low", "High"))
testData$rating_binary <- factor(testData$rating_binary, levels = c("Low", "High"))

x_train <- trainData %>% select(-rating_binary)
x_test <- testData %>% select(-rating_binary)
y_train <- trainData$rating_binary
y_test <- testData$rating_binary

# ------------------- Normalisierung der Daten -------------------
preprocess_params <- preProcess(x_train, method = c("center", "scale"))
x_train_normalized <- predict(preprocess_params, x_train)
x_test_normalized <- predict(preprocess_params, x_test)
trainData_normalized <- cbind(x_train_normalized, rating_binary = trainData$rating_binary)
testData_normalized <- cbind(x_test_normalized, rating_binary = testData$rating_binary)

# -------------------Grid Search for Logistic Regression: weighted_rating------------------------------------------

tune_grid <- expand.grid(
  .alpha = c(0, 0.5, 1),  # 0 = Ridge, 1 = Lasso, 0.5 = ElasticNet
  .lambda = seq(0.0001, 1, length = 100)  # Bereich für Lambda (mehr Werte)
)

train_control <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE,  
  summaryFunction = twoClassSummary
)

logit_model_weighted <- train(
  rating_binary ~ content_sentiment_score + 
    title_sentiment_score + weighted_rating + discounted_price + discount_percentage + actual_price,
  data = trainData_normalized,  
  method = "glmnet",  
  trControl = train_control,
  tuneGrid = tune_grid,  
  na.action = na.omit
)


print(logit_model_weighted)

prediction_probabilities_weighted <- predict(logit_model_weighted, newdata = testData_normalized, type = "prob")[, 2]
predicted_classes_weighted <- ifelse(prediction_probabilities_weighted > 0.5, 1, 0)

roc_curve_weighted <- roc(y_test, prediction_probabilities_weighted)
plot(roc_curve_weighted, col = "blue")
abline(a = 0, b = 1, col = "red", lty = 2)
cat("AUC (Weighted Rating - CV):", auc(roc_curve_weighted), "\n")

calculate_metrics <- function(predictions, y_true, probabilities) {
  
  tp <- sum(predictions == 1 & y_true == 1)
  tn <- sum(predictions == 0 & y_true == 0)
  fp <- sum(predictions == 1 & y_true == 0)
  fn <- sum(predictions == 0 & y_true == 1)
  
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  accuracy <- (tp + tn) / (tp + tn + fp + fn)

  mae <- mean(abs(y_true - probabilities))
  mse <- mean((y_true - probabilities)^2)
  
  metrics <- data.frame(
    Metric = c("Precision", "Recall", "F1-Score", "Accuracy", "MAE", "MSE"),
    Value = round(c(precision, recall, f1_score, accuracy, mae, mse), 3)
  )
  
  return(metrics)
}
y_test_numeric <- ifelse(y_test == "High", 1, 0)

logit_metrics_weighted <- calculate_metrics(predicted_classes_weighted, y_test_numeric, prediction_probabilities_weighted)

print(logit_metrics_weighted)
# -------------------Logistic Regression: rating------------------------------------------
set.seed(123)
model_data <- df %>%
  mutate(
    rating_binary = factor(ifelse(rating_count > median(rating_count, na.rm = TRUE), "High", "Low"), levels = c("Low", "High"))
  ) %>%
  select(
    rating_binary, content_sentiment_score, title_sentiment_score, rating, discounted_price, discount_percentage, actual_price
  )


model_data <- na.omit(model_data)

trainIndex <- createDataPartition(model_data$rating_binary, p = 0.8, list = FALSE)
trainData <- model_data[trainIndex, ]
testData <- model_data[-trainIndex, ]

trainData$rating_binary <- factor(trainData$rating_binary, levels = c("Low", "High"))
testData$rating_binary <- factor(testData$rating_binary, levels = c("Low", "High"))

x_train <- trainData %>% select(-rating_binary)
x_test <- testData %>% select(-rating_binary)
y_train <- trainData$rating_binary
y_test <- testData$rating_binary

# ------------------- Normalisierung der Daten -------------------
preprocess_params <- preProcess(x_train, method = c("center", "scale"))
x_train_normalized <- predict(preprocess_params, x_train)
x_test_normalized <- predict(preprocess_params, x_test)
trainData_normalized <- cbind(x_train_normalized, rating_binary = trainData$rating_binary)
testData_normalized <- cbind(x_test_normalized, rating_binary = testData$rating_binary)

#-----------------------LR rating---------------------------
tune_grid <- expand.grid(
  .alpha = c(0, 0.5, 1),  # 0 = Ridge, 1 = Lasso, 0.5 = ElasticNet
  .lambda = seq(0.0001, 1, length = 100)  # Bereich für Lambda (mehr Werte)
)

train_control <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE,  
  summaryFunction = twoClassSummary
)

logit_model_rating <- train(
  rating_binary ~ content_sentiment_score + 
    title_sentiment_score + rating + discounted_price + discount_percentage + actual_price,
  data = trainData_normalized,  
  method = "glmnet",  
  trControl = train_control,
  tuneGrid = tune_grid,  
  na.action = na.omit
)

print(logit_model_rating)

prediction_probabilities_rating <- predict(logit_model_rating, newdata = testData_normalized, type = "prob")[,2]
predicted_classes_rating <- ifelse(prediction_probabilities_rating > 0.5, 1, 0)



roc_curve <- roc(y_test_numeric, prediction_probabilities_rating)
plot(roc_curve, col = "blue")
abline(a = 0, b = 1, col = "red", lty = 2)
cat("AUC:", auc(roc_curve), "\n")

y_test_numeric <- ifelse(y_test == "High", 1, 0)

calculate_metrics <- function(predictions, y_true, probabilities) {
  
  tp <- sum(predictions == 1 & y_true == 1)
  tn <- sum(predictions == 0 & y_true == 0)
  fp <- sum(predictions == 1 & y_true == 0)
  fn <- sum(predictions == 0 & y_true == 1)
  
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  
  mae <- mean(abs(y_true - probabilities))
  mse <- mean((y_true - probabilities)^2)
  
  metrics <- data.frame(
    Metric = c("Precision", "Recall", "F1-Score", "Accuracy", "MAE", "MSE"),
    Value = round(c(precision, recall, f1_score, accuracy, mae, mse), 3)
  )
  
  return(metrics)
}


logit_metrics_rating<- calculate_metrics(predicted_classes_rating, y_test_numeric, prediction_probabilities_rating)

print(logit_metrics_rating)

# ------------------- Visualization for Comparison --------------------------
plot_metrics <- function(logit_metrics_weighted, logit_metrics_rating) {
  logit_metrics_weighted$Model <- "Weighted Rating"
  logit_metrics_rating$Model <- "Rating"
  combined_metrics <- rbind(logit_metrics_weighted, logit_metrics_rating)
  
  ggplot(combined_metrics, aes(x = Metric, y = Value, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge", show.legend = TRUE) +
    geom_text(aes(label = Value), vjust = -0.5, position = position_dodge(0.9)) +
    theme_minimal() +
    labs(
      y = "Value",
      x = "Metric"
    ) +
    scale_fill_brewer(palette = "Set3") + 
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_metrics(logit_metrics_weighted, logit_metrics_rating)

# ------------------- Visualization for Logistic Regression: Weighted Rating --------------------------

plot_metrics_weighted <- function(logit_metrics_weighted) {
  logit_metrics_weighted$Metric <- factor(logit_metrics_weighted$Metric, 
                                          levels = c("Accuracy", "F1-Score", "Precision", "Recall", "MAE", "MSE"))
  
  ggplot(logit_metrics_weighted, aes(x = Metric, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    geom_text(aes(label = Value), vjust = -0.5) +
    theme_minimal() +
    labs(
      y = "Value",
      x = "Metric"
    ) +
    scale_fill_brewer(palette = "Set3") 
}

plot_metrics_weighted(logit_metrics_weighted)


# ------------------- Visualization for Logistic Regression: Rating --------------------------

plot_metrics_rating <- function(logit_metrics_rating) {
  logit_metrics_rating$Metric <- factor(logit_metrics_rating$Metric, 
                                        levels = c("Accuracy", "F1-Score", "Precision", "Recall", "MAE", "MSE"))
  
  ggplot(logit_metrics_rating, aes(x = Metric, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    geom_text(aes(label = Value), vjust = -0.5) +
    theme_minimal() +
    labs(
      y = "Value",
      x = "Metric"
    ) +
    scale_fill_brewer(palette = "Set3")  
}

plot_metrics_rating(logit_metrics_rating)

#----------------Logistic Regression weighted_rating Outputs-----------------------------
threshold <- median(df$rating_count, na.rm = TRUE)
df <- df %>% mutate(rating_category = ifelse(rating_count > threshold, "High", "Low"))
df$rating_category <- factor(df$rating_category, levels = c("Low", "High"))


logit_model_weighted <- glm(
  formula = rating_category ~  content_sentiment_score + 
    title_sentiment_score + weighted_rating + discounted_price + discount_percentage + actual_price,
  family = binomial,
  data = df
)

summary(logit_model_weighted)

logit_table_weighted <- tidy(logit_model_weighted) %>%
  mutate(
    Significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  dplyr::rename(
    Term = term,
    Estimate = estimate,
    `Std. Error` = std.error,
    Statistic = statistic,
    `P-value` = p.value
  )

logit_weighted_latex <- kable(
  logit_table_weighted,
  format = "latex",
  booktabs = TRUE,
  caption = "Coefficients and Significance for the Logistic Regression Model"
) %>%
  kable_styling(latex_options = c("hold_position"))
cat(logit_weighted_latex)

#----------------Logistic Regression rating Outputs-----------------------------
logit_model_rating <- glm(
  formula = rating_category ~  content_sentiment_score + 
    title_sentiment_score + rating + discounted_price + discount_percentage + actual_price,
  family = binomial,
  data = df
)

summary(logit_model_rating)

logit_table_rating <- tidy(logit_model_rating) %>%
  mutate(
    Significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  dplyr::rename(
    Term = term,
    Estimate = estimate,
    `Std. Error` = std.error,
    Statistic = statistic,
    `P-value` = p.value
  )

logit_rating_latex <- kable(
  logit_table_rating,
  format = "latex",
  booktabs = TRUE,
  caption = "Coefficients and Significance for the Logistic Regression Model"
) %>%
  kable_styling(latex_options = c("hold_position"))

cat(logit_rating_latex)

#-----------------------Linear Model weighted_rating-------------------------------------------------
lm_model_weighted <- lm(weighted_rating ~ content_sentiment_score + title_sentiment_score + rating_count + discounted_price + discount_percentage + actual_price, data = df)

summary(lm_model_weighted)

lm_table_weighted <- tidy(lm_model_weighted) %>%
  mutate(
    Significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  dplyr::rename(
    Term = term,
    Estimate = estimate,
    `Std. Error` = std.error,
    Statistic = statistic,
    `P-value` = p.value
  )

lm_weighted_latex <- kable(
  lm_table_weighted,
  format = "latex",
  booktabs = TRUE,
  caption = "Coefficients and Significance for the Linear Regression Model"
) %>%
  kable_styling(latex_options = c("hold_position"))


#----------------------Linear Model rating------------------------------------------------------------
lm_model_rating <- lm(rating ~ content_sentiment_score + title_sentiment_score + rating_count + discounted_price + discount_percentage + actual_price, data = df)

summary(lm_model_rating)

lm_table_rating <- tidy(lm_model_rating) %>%
  mutate(
    Significance = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  dplyr::rename(
    Term = term,
    Estimate = estimate,
    `Std. Error` = std.error,
    Statistic = statistic,
    `P-value` = p.value
  )

lm_rating_latex <- kable(
  lm_table_rating,
  format = "latex",
  booktabs = TRUE,
  caption = "Coefficients and Significance for the Linear Regression Model"
) %>%
  kable_styling(latex_options = c("hold_position"))

cat(lm_rating_latex)






