df <- read.csv("processed_amazon_data.csv")

library(caret)
library(ggplot2)
library(lattice)
library(dplyr)
library(pROC)
library(rpart)
library(rpart.plot)
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

x_train <-  trainData %>% select(-rating_binary)
x_test <-  testData %>% select(-rating_binary)
y_train <- trainData$rating_binary
y_test <- testData$rating_binary

# ------------------ Utility Functions ------------------
classification_report <- function(predictions, y_true) {
  confusion <- confusionMatrix(predictions, y_true)
  cat("Classification Report:\n")
  print(confusion)
  return(confusion)
}
# ------------------ Decision Tree ------------------
dt_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
dt_grid <- expand.grid(cp = seq(0.006, 0.009, by = 0.0002))
#----------------Hyperparamter Tuning------------------------------------
# Testing different complexity parameters

dt_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.002)) #Accuracy: 0,863

dt_grid <- expand.grid(cp = seq(0.005, 0.01, by = 0.0005)) #Accuracy: 0,8699

dt_grid <- expand.grid(cp = seq(0.0001, 0.02, by = 0.0005)) #Accuracy: 0,8767

dt_grid <- expand.grid(cp = seq(0.006, 0.009, by = 0.0002)) #Accuracy: 0,8767 # best choice

dt_grid <- expand.grid(cp = seq(0.003, 0.02, by = 0.001)) # For Visualization

results <- dt_model$results  
ggplot(results, aes(x = cp, y = ROC)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(x = "Complexity Parameter (cp)", y = "ROC AUC")
#--------------------------------------------------------------------------- 
set.seed(123)
dt_model <- train(
  x = x_train,
  y = y_train,
  method = "rpart",  
  metric = "ROC",    
  trControl = dt_control, 
  tuneGrid = dt_grid
)

dt_predictions <- predict(dt_model, newdata = x_test)
dt_probabilities <- predict(dt_model, newdata = x_test, type = "prob")[, "High"]
dt_roc <- roc(y_test, dt_probabilities, levels = c("Low", "High"))
dt_auc <- auc(dt_roc)

cat("Decision Tree Results:\n")
classification_report(dt_predictions, y_test)
cat("Decision Tree AUC:", round(dt_auc, 2), "\n")

plot(dt_roc, col = "blue")
abline(a = 0, b = 1, col = "red", lty = 2)

# ------------------ Visualize Decision Tree ---------------------------
cat("Visualizing Pruned Decision Tree:\n")

dt_model_pruned <- prune(dt_model$finalModel, cp = 0.0075)  
rpart.plot(dt_model_pruned, cex = 0.6)
# -------------------------- SHAP --------------------------------------
library(iml)

compute_shap_dt <- function(dt_model, x_train, y_train, x_test) {
  cat("\nComputing SHAP values for Decision Tree...\n")
  
  y_train_numeric <- as.numeric(as.factor(y_train)) - 1
  
  predictor_dt <- Predictor$new(
    model = dt_model$finalModel,  
    data = as.data.frame(x_train),
    y = y_train_numeric,
    predict.fun = function(model, newdata) {
      prob <- predict(model, newdata, type = "prob")[, "High"]
      return(prob)
    }
  )
  
  sample_instance <- as.data.frame(x_test[1, , drop = FALSE])
  shap_dt <- Shapley$new(predictor_dt, x.interest = sample_instance)
  
  cat("SHAP values for a single prediction (Decision Tree):\n")
  print(shap_dt$results)
  plot(shap_dt)
  
  cat("\nComputing Global Feature Importance...\n")
  feature_importance <- FeatureImp$new(
  predictor_dt, 
  loss = function(actual, predicted) {
    mean(abs(actual - predicted)) 
  }
)

print(feature_importance$results)

ggplot(feature_importance$results, aes(x = importance, y = reorder(feature, importance))) +
  geom_bar(stat = "identity", fill = "darkblue") +
  theme_minimal() +
  labs(
    title = "Feature Importance - Decision Tree",
    x = "Importance (Loss: function(actual, predicted))",
    y = "Features"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.y = element_text(size = 10)
  )
}
compute_shap_dt(dt_model, x_train, y_train, x_test)


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


dt_predictions <- predict(dt_model, newdata = x_test)
dt_probabilities <- predict(dt_model, newdata = x_test, type = "prob")[, "High"]


dt_metrics_table <- calculate_metrics(dt_predictions, y_test, dt_probabilities)
print(dt_metrics_table)

library(knitr)
library(kableExtra)


kable(dt_metrics_table, caption = "Decision Tree Metrics") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)


write.csv(dt_metrics_table, "Decision_Tree_Metrics.csv", row.names = FALSE)

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


plot_metrics(dt_metrics_table)



