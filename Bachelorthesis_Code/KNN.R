df <- read.csv("processed_amazon_data.csv")


library(caret)
library(lattice)
library(dplyr)
library(pROC)
library(MLmetrics)
library(e1071)


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
y_train <- trainData$rating_binary
y_test <- testData$rating_binary

# ------------------ Utility Functions ------------------
classification_report <- function(predictions, y_true) {
  confusion <- confusionMatrix(predictions, y_true)
  cat("Classification Report:\n")
  print(confusion)
  return(confusion)
}

#---------------------------KNN------------------------------------------------
knn_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
knn_grid <- expand.grid(k = 9) #best k

#-------------------------Hyperparameter Tuning--------------------------------------------------
knn_grid <- expand.grid(k = seq(3, 21, by = 2))

knn_model <- train(
  x = x_train, 
  y = y_train, 
  method = "knn", 
  metric = "ROC", 
  trControl = knn_control, 
  tuneGrid = knn_grid
)


cat("Best KNN Model:\n")
print(knn_model$bestTune) 


ggplot(knn_model$results, aes(x = k, y = ROC)) +
  geom_line(color = "blue") +
  geom_point(color = "red", size = 2) +
  labs(
    x = "Number of Neighbors (k)",
    y = "ROC AUC"
  ) +
  theme_minimal()

#--------------------Model Evaluation------------------------
knn_predictions <- predict(knn_model, newdata = x_test)  # Hard class predictions
knn_probabilities <- predict(knn_model, newdata = x_test, type = "prob")[, "High"]  # Probabilities for the "High" class

knn_roc <- roc(y_test, knn_probabilities, levels = c("Low", "High"))  # Create ROC curve
auc_value <- auc(knn_roc)  # Calculate AUC

confusion <- confusionMatrix(knn_predictions, y_test)
cat("KNN Results:\n")
print(confusion)
cat("AUC:", round(auc_value, 2), "\n")

plot(knn_roc, col = "blue")
abline(a = 0, b = 1, col = "red", lty = 2)

# ------------------ SHAP for Models ------------------
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

knn_predictions <- predict(knn_model, newdata = x_test, type = "prob")[, "High"]
summary(knn_predictions)
x_train <- as.data.frame(x_train)  
x_test <- as.data.frame(x_test)    


y_train_numeric <- as.numeric(as.factor(y_train)) - 1

knn_predict <- function(newdata) {
  probabilities <- predict(knn_model, newdata, type = "prob")[, "High"]  
  return(probabilities)
}
knn_predictor <- Predictor$new(
  predict.fun = knn_predict,         
  data = x_train,                    
  y = y_train_numeric               
)

sample_instance <- x_test[1, , drop = FALSE]  

shap_knn <- Shapley$new(knn_predictor, x.interest = sample_instance)

plot(shap_knn)

feature_importance <- FeatureImp$new(knn_predictor, loss = "mae")

ggplot(feature_importance$results, aes(x = importance, y = reorder(feature, importance))) +
  geom_bar(stat = "identity", fill = "darkblue") +
  theme_minimal() +
  labs(
    title = "Feature Importance - KNN",
    x = "Importance (Loss: function(actual, predicted))",
    y = "Features"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.y = element_text(size = 10)
  ) 

print(feature_importance$results)
plot(feature_importance)


cat("\nGlobal SHAP Feature Importance (KNN):\n")
print(feature_importance$results)

plot(feature_importance)

# ------------------ Generate and Save Classification Report for KNN ------------------
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
  mse <- mean((pred_numeric - y_numeric)^2)
  mae <- mean(abs(pred_numeric - y_numeric))
  
  metrics <- data.frame(
    Metric = c("Precision", "Recall", "F1-Score", "Accuracy"),
    Value = round(c(precision, recall, f1_score, accuracy), 3)
  )
  
  return(metrics)
}

knn_predictions <- predict(knn_model, newdata = x_test)
knn_probabilities <- predict(knn_model, newdata = x_test, type = "prob")[, "High"]

knn_metrics_table <- calculate_metrics(knn_predictions, y_test, knn_probabilities)
print(knn_metrics_table)


library(knitr)
library(kableExtra)

kable(knn_metrics_table, caption = "KNN Metrics") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)

write.csv(knn_metrics_table, "KNN_Metrics.csv", row.names = FALSE)

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

plot_metrics(knn_metrics_table)




