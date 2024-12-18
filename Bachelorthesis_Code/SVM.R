df <- read.csv("processed_amazon_data.csv")

library(caret)
library(ggplot2)
library(lattice)
library(pROC)
library(dplyr)
library(e1071) 
library(kernlab)

set.seed(123)

# ------------------ Data Preparation ------------------
model_data <- df %>%
  mutate(
    rating_binary = factor(ifelse(rating_count > median(rating_count, na.rm = TRUE), "High", "Low"), levels = c("Low", "High"))
  ) %>%
  select(
    rating_binary, discounted_price, content_sentiment_score, title_sentiment_score,
    combined_sentiment_score, actual_price, discount_percentage, weighted_rating
  )

trainIndex <- createDataPartition(model_data$rating_binary, p = 0.8, list = FALSE)
trainData <- model_data[trainIndex, ]
testData <- model_data[-trainIndex, ]

preProc <- preProcess(trainData %>% select(-rating_binary), method = c("center", "scale"))
x_train <- predict(preProc, trainData %>% select(-rating_binary))
x_test <- predict(preProc, testData %>% select(-rating_binary))
y_train <- as.factor(trainData$rating_binary)  
y_test <- as.factor(testData$rating_binary)

# ------------------ SVM Model Training ------------------
svm_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

svm_grid <- expand.grid(
  sigma = seq(0.001, 0.1, by = 0.01),
  C = seq(1, 100, by = 10)
)
#--------------Hyperparameter Tuning---------------------
svm_grid_refined <- expand.grid(
  sigma = seq(0.02, 0.1, by = 0.005),
  C = seq(1, 100, by = 10)
)
#--------------------------------------------------------

svm_model <- train(
  x = x_train,
  y = y_train,
  method = "svmRadial",
  metric = "ROC",
  trControl = svm_control,
  tuneGrid = svm_grid
)

plot(svm_model)

print(svm_model)
# ------------------ Model Evaluation ------------------
svm_predictions <- predict(svm_model, newdata = x_test)
svm_probabilities <- predict(svm_model, newdata = x_test, type = "prob")[, "High"]

svm_roc <- roc(y_test, svm_probabilities, levels = c("Low", "High"))
auc_value <- auc(svm_roc)

confusion <- confusionMatrix(svm_predictions, y_test)
cat("SVM Results:\n")
print(confusion)
cat("AUC:", round(auc_value, 2), "\n")

plot(svm_roc, col = "blue")
abline(a = 0, b = 1, col = "red", lty = 2)
# ------------------ SHAP for Models -----------------------------------
library(iml)

compute_shap <- function(model, model_name, x_train, y_train, x_test) {
  cat(paste("\nComputing SHAP values for", model_name, "...\n"))
  
  predictor <- Predictor$new(
    model = model,
    data = as.data.frame(x_train),
    y = as.numeric(as.factor(y_train)) - 1  
  )
  
  sample_instance <- as.data.frame(x_test[1, , drop = FALSE])
  
  shap <- Shapley$new(predictor, x.interest = sample_instance)
  
  cat(paste("SHAP values for a single prediction (", model_name, "):\n"))
  print(shap$results)
  plot(shap)

  feature_importance <- FeatureImp$new(predictor, loss = "mae")
  cat(paste("\nGlobal SHAP Feature Importance (", model_name, "):\n"))
  print(feature_importance$results)
  plot(feature_importance)
}

svm_predict <- function(newdata) {
  probabilities <- predict(svm_model, newdata, type = "prob")[, "High"]
  return(probabilities)
}

svm_predictor <- Predictor$new(
  predict.fun = svm_predict,
  data = as.data.frame(x_train),
  y = as.numeric(as.factor(y_train)) - 1
)
sample_instance <- as.data.frame(x_test[1, , drop = FALSE])
shap_svm <- Shapley$new(svm_predictor, x.interest = sample_instance)
cat("\nSHAP values for a single prediction (SVM):\n")
print(shap_svm$results)
plot(shap_svm)


feature_importance <- FeatureImp$new(svm_predictor, loss = "mae")


print(feature_importance$results)

ggplot(feature_importance$results, aes(x = importance, y = reorder(feature, importance))) +
  geom_bar(stat = "identity", fill = "darkblue") +
  theme_minimal() +
  labs(
    title = "Feature Importance - SVM",
    x = "Importance (Loss: function(actual, predicted))",
    y = "Features"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.y = element_text(size = 10)
  )

plot(feature_importance)


# ------------------ Generate and Save Classification Report ------------------
library(ggplot2)

calculate_metrics <- function(predictions, y_true, probabilities) {

  predictions <- factor(predictions, levels = c("Low", "High"))
  y_true <- factor(y_true, levels = c("Low", "High"))
  
  tp <- sum(predictions == "High" & y_true == "High")
  tn <- sum(predictions == "Low" & y_true == "Low")
  fp <- sum(predictions == "High" & y_true == "Low")
  fn <- sum(predictions == "Low" & y_true == "High")

  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  accuracy <- (tp + tn) / (tp + tn + fp + fn)

  y_numeric <- ifelse(y_true == "High", 1, 0)
  pred_numeric <- ifelse(predictions == "High", 1, 0)

  metrics <- data.frame(
    Metric = c("Precision", "Recall", "F1-Score", "Accuracy"),
    Value = round(c(precision, recall, f1_score, accuracy), 3)
  )
  
  return(metrics)
}
library(knitr)
library(kableExtra)

svm_predictions <- predict(svm_model, newdata = x_test)
svm_probabilities <- predict(svm_model, newdata = x_test, type = "prob")[, "High"]

svm_metrics_table <- calculate_metrics(svm_predictions, y_test, svm_probabilities)
print(svm_metrics_table)

kable(svm_metrics_table, caption = "SVM Metrics") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)

write.csv(svm_metrics_table, "SVM_Metrics.csv", row.names = FALSE)

plot_metrics <- function(metrics_table) {
  ggplot(metrics_table, aes(x = Metric, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    geom_text(aes(label = Value), vjust = -0.5) +
    theme_minimal() +
    labs(
      y = "Value",
      x = "Metric"
    ) +
    scale_fill_brewer(palette = "Set3")
}

plot_metrics(svm_metrics_table)

