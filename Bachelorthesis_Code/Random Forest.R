df <- read.csv("processed_amazon_data.csv")

library(caret)
library(ggplot2)
library(lattice)
library(dplyr)
library(pROC)
library(randomForest)
library(MLmetrics)


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

x_train <- as.matrix(trainData %>% select(-rating_binary))
y_train <- trainData$rating_binary
x_test <- as.matrix(testData %>% select(-rating_binary))
y_test <- testData$rating_binary

# ------------------ Utility Functions ------------------
classification_report <- function(predictions, y_true) {
  confusion <- confusionMatrix(predictions, y_true)
  cat("Classification Report:\n")
  print(confusion)
  return(confusion)
}

# ------------------ Random Forest ------------------
rf_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
rf_grid <- expand.grid(mtry = seq(2, ncol(x_train), by = 1))

rf_model <- train(
  x = x_train, y = y_train, method = "rf", metric = "ROC", trControl = rf_control,
  tuneGrid = rf_grid, ntree = 300, importance = TRUE
)

rf_predictions <- predict(rf_model, newdata = x_test)
rf_probabilities <- predict(rf_model, newdata = x_test, type = "prob")[, "High"]
rf_roc <- roc(y_test, rf_probabilities, levels = c("Low", "High"))
rf_auc <- auc(rf_roc)

cat("Random Forest Results:\n")
classification_report(rf_predictions, y_test)
cat("Random Forest AUC:", round(rf_auc, 2), "\n")
plot(rf_roc, col = "blue")
abline(a = 0, b = 1, col = "red", lty = 2)
#-------------------Hyperparameter Tuning with Grid Search-------------------
rf_control <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

rf_grid <- expand.grid(mtry = c(2, 4, 6, sqrt(ncol(x_train)), ncol(x_train)))

tree_sizes <- c(100, 200, 300)

all_results <- data.frame()

for (ntree in tree_sizes) {
  rf_model <- train(
    x = x_train, 
    y = y_train, 
    method = "rf", 
    metric = "ROC", 
    trControl = rf_control, 
    tuneGrid = rf_grid, 
    ntree = ntree, 
    importance = TRUE
  )
  
  model_results <- rf_model$results
  model_results$ntree <- ntree
  all_results <- rbind(all_results, model_results)
}

ggplot(all_results, aes(x = mtry, y = ROC, color = as.factor(ntree))) +
  geom_line() +
  geom_point() +
  labs(
    x = "Number of Randomly Selected Features (mtry)",
    y = "ROC AUC",
    color = "Number of Trees"
  ) +
  theme_minimal()

best_result <- all_results[which.max(all_results$ROC), ]
cat("Best Combination:\n")
print(best_result) #mtry= 4, ntree=300
#-----------------------------------------------------------------
final_rf_model <- train(
  x = x_train,
  y = y_train,
  method = "rf",
  metric = "ROC",
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
  tuneGrid = expand.grid(mtry = 4), 
  ntree = 300, 
  importance = TRUE  
)

final_rf_predictions <- predict(final_rf_model, newdata = x_test)
final_rf_probabilities <- predict(final_rf_model, newdata = x_test, type = "prob")[, "High"]

final_rf_roc <- roc(y_test, final_rf_probabilities, levels = c("Low", "High"))
final_rf_auc <- auc(final_rf_roc)

cat("Final Random Forest Results:\n")
print(confusionMatrix(final_rf_predictions, y_test))
cat("Final Random Forest AUC:", round(final_rf_auc, 3), "\n")

plot(final_rf_roc, col = "blue", main = "Final Random Forest ROC Curve")
abline(a = 0, b = 1, col = "red", lty = 2)

# ------------------ SHAP ------------------
library(iml)

compute_shap_rf <- function(rf_model, x_train, y_train, x_test) 
  cat("\nComputing SHAP values for Random Forest...\n")
  
 
  y_train_numeric <- as.numeric(as.factor(y_train)) - 1 
  
  
  predictor_rf <- Predictor$new(
    model = rf_model$finalModel,  
    data = as.data.frame(x_train),
    y = y_train_numeric,  
    predict.fun = function(model, newdata) {
      prob <- predict(model, newdata, type = "prob")[, 2]  
      return(prob)
    }
  )
  
  
  sample_instance <- as.data.frame(x_test[1, , drop = FALSE])  
  shap_rf <- Shapley$new(predictor_rf, x.interest = sample_instance)
  
  
  cat("SHAP values for a single prediction (Random Forest):\n")
  print(shap_rf$results)
  
  
  plot(shap_rf)
  
  
  cat("\nComputing Global Feature Importance...\n")
  feature_importance <- FeatureImp$new(
    predictor_rf, 
    loss = function(actual, predicted) {
      mean(abs(actual - predicted)) 
    }
  )
  print(feature_importance$results)
  
  ggplot(feature_importance$results, aes(x = importance, y = reorder(feature, importance))) +
    geom_bar(stat = "identity", fill = "darkblue") +
    theme_minimal() +
    labs(
      title = "Feature Importance - Random Forest",
      x = "Importance (Loss: function(actual, predicted))",
      y = "Features"
    ) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      axis.text.y = element_text(size = 10)
    ) 
  
  print(feature_importance$results)
  plot(feature_importance)



compute_shap_rf(rf_model, x_train, y_train, x_test)

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


rf_predictions <- predict(rf_model, newdata = x_test)
rf_probabilities <- predict(rf_model, newdata = x_test, type = "prob")[, "High"]


rf_metrics_table <- calculate_metrics(rf_predictions, y_test, rf_probabilities)
print(rf_metrics_table)


library(knitr)
library(kableExtra)


kable(rf_metrics_table, caption = "Random Forest Metrics") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)


write.csv(rf_metrics_table, "Random_Forest_Metrics.csv", row.names = FALSE)

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


plot_metrics(rf_metrics_table)

