setwd("/Users/romyl/OneDrive/Desktop/Bachelor Thesis")
df <- read.csv("processed_amazon_data.csv")

library(caret)
library(lattice)
library(dplyr)
library(pROC)
library(xgboost)
library(MLmetrics)
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

calculate_mse_rmse <- function(predictions, y_true) {
  y_numeric <- ifelse(y_true == "High", 1, 0)
  pred_numeric <- ifelse(predictions == "High", 1, 0)
  mse <- MSE(pred_numeric, y_numeric)
  rmse <- RMSE(pred_numeric, y_numeric)
  cat("MSE:", mse, "\n")
  cat("RMSE:", rmse, "\n")
  return(list(MSE = mse, RMSE = rmse))
}

# ------------------ XGBoost ------------------
y_train_xgb <- as.numeric(y_train) - 1  
y_test_xgb <- as.numeric(y_test) - 1

dtrain <- xgb.DMatrix(data = x_train, label = y_train_xgb)
dtest <- xgb.DMatrix(data = x_test, label = y_test_xgb)

xgb_params <- list(
  objective = "binary:logistic",  
  eval_metric = "auc",           
  eta = 0.2,                     
  max_depth = 8,                 
  subsample = 0.8,               
  colsample_bytree = 0.6      
)

xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 100,                        
  watchlist = list(train = dtrain, test = dtest), 
  early_stopping_rounds = 10,           
  print_every_n = 10                   
)
#---------------------Hyperparameter Tuning with AUc---------------------------
hyperparameter_grid <- expand.grid(
  eta = c(0.01, 0.1, 0.2),                
  max_depth = c(4, 6, 8),                 
  subsample = c(0.6, 0.8, 1),             
  colsample_bytree = c(0.6, 0.8, 1),      
  nrounds = c(50, 100, 150)               
)


grid_results <- data.frame(
  eta = numeric(),
  max_depth = numeric(),
  subsample = numeric(),
  colsample_bytree = numeric(),
  nrounds = numeric(),
  AUC = numeric()
)


for (i in 1:nrow(hyperparameter_grid)) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = hyperparameter_grid$eta[i],
    max_depth = hyperparameter_grid$max_depth[i],
    subsample = hyperparameter_grid$subsample[i],
    colsample_bytree = hyperparameter_grid$colsample_bytree[i]
  )
  

  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = hyperparameter_grid$nrounds[i],
    watchlist = list(train = dtrain, test = dtest),
    early_stopping_rounds = 10,
    verbose = 0
  )

  predictions <- predict(xgb_model, dtest)
  roc_curve <- roc(y_test_xgb, predictions)
  auc_value <- auc(roc_curve)
  

  grid_results <- rbind(
    grid_results,
    data.frame(
      eta = params$eta,
      max_depth = params$max_depth,
      subsample = params$subsample,
      colsample_bytree = params$colsample_bytree,
      nrounds = hyperparameter_grid$nrounds[i],
      AUC = auc_value
    )
  )
}


best_params <- grid_results[which.max(grid_results$AUC), ]
print(best_params) # eta:0.1, max_depth: 8, subsample: 0.6, colsample_bytree:0.8, nrounds: 50, AUC: 0.963


#-------------------------------------------------------------------------------
xgb_predictions <- predict(xgb_model, dtest)
xgb_class <- ifelse(xgb_predictions > 0.5, "High", "Low")

xgb_roc <- roc(y_test_xgb, xgb_predictions)  
xgb_auc <- auc(xgb_roc)

cat("XGBoost Results:\n")
classification_report(as.factor(xgb_class), as.factor(ifelse(y_test_xgb == 1, "High", "Low")))
cat("XGBoost AUC:", round(xgb_auc, 2), "\n")
calculate_mse_rmse(as.factor(xgb_class), as.factor(ifelse(y_test_xgb == 1, "High", "Low")))

plot(xgb_roc, col = "blue", main = "XGBoost ROC Curve")
abline(a = 0, b = 1, col = "red", lty = 2)


# ------------------ Learning Curve across different Training Set Sizes------------------
plot_learning_curve <- function(method, trainData, metric = "ROC", tuneGrid = NULL) {
  library(pROC)  
  
  train_sizes <- seq(0.1, 1.0, by = 0.2) 
  results <- data.frame(Train_Size = numeric(), Metric = numeric()) 
  
  for (size in train_sizes) {
    idx <- sample(1:nrow(trainData), size = floor(size * nrow(trainData)))
    train_subset <- trainData[idx, ]

    x_train_subset <- as.matrix(train_subset %>% select(-rating_binary))
    y_train_subset <- train_subset$rating_binary

    model <- train(
      x = x_train_subset, 
      y = y_train_subset, 
      method = method, 
      metric = metric,
      trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
      tuneGrid = tuneGrid
    )

    probabilities <- predict(model, newdata = x_train_subset, type = "prob")[, "High"]

    roc_curve <- roc(y_train_subset, probabilities, levels = c("Low", "High"))
    auc_value <- auc(roc_curve)

    results <- rbind(results, data.frame(Train_Size = size, Metric = auc_value))
  }

  plot(results$Train_Size, results$Metric, type = "b", col = "blue", pch = 19,
       xlab = "Training Set Size", ylab = metric)
  title(main = paste("Learning Curve -", method))
}
small_tune_grid <- expand.grid(
  nrounds = c(50, 100),
  eta = 0.1,
  max_depth = 6,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.7
)

plot_learning_curve(
  method = "xgbTree",
  trainData = trainData,
  metric = "ROC",
  tuneGrid = small_tune_grid
)
#--------------Learning Curve across different Test Set Sizes-------------------
tuneGrid <- expand.grid(
  nrounds = c(50, 100),
  eta = c(0.1, 0.2),
  max_depth = c(4, 6),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

plot_learning_curve_test <- function(method, trainData, testData, metric = "ROC", tuneGrid = NULL) {
  library(ggplot2)
  library(pROC)
  
  test_sizes <- seq(0.1, 1.0, by = 0.2)  # Test set sizes to vary
  results <- data.frame(Test_Size = numeric(), Metric = numeric())
  
  for (size in test_sizes) {
   
    idx <- sample(1:nrow(testData), size = floor(size * nrow(testData)))
    test_subset <- testData[idx, ]
    
   
    x_train <- as.matrix(trainData %>% select(-rating_binary))
    y_train <- trainData$rating_binary
    
    x_test_subset <- as.matrix(test_subset %>% select(-rating_binary))
    y_test_subset <- test_subset$rating_binary
    
  
    model <- train(
      x = x_train, 
      y = y_train, 
      method = method, 
      metric = metric,
      trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
      tuneGrid = tuneGrid
    )
    
   
    probabilities <- predict(model, newdata = x_test_subset, type = "prob")[, "High"]
    roc_curve <- roc(y_test_subset, probabilities, levels = c("Low", "High"))
    auc_value <- auc(roc_curve)
    
    
    results <- rbind(results, data.frame(Test_Size = size, Metric = auc_value))
  }
  
  
  ggplot(results, aes(x = Test_Size, y = Metric)) +
    geom_line(color = "blue", size = 1) +
    geom_point(color = "red", size = 3) +
    labs(
      title = paste("Learning Curve - Test Set Size for", method),
      x = "Test Set Size (Proportion)",
      y = metric
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10)
    )
}

plot_learning_curve_test(
  method = "xgbTree", 
  trainData = trainData, 
  testData = testData, 
  metric = "ROC", 
  tuneGrid = tuneGrid
)
# ------------------ SHAP for Models -----------------------------------------------
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

x_train <- as.data.frame(x_train)  
x_test <- as.data.frame(x_test)    

y_train_numeric <- as.numeric(as.factor(y_train)) - 1
predict_xgb <- function(model, newdata) {
  dmatrix <- xgb.DMatrix(data = as.matrix(newdata)) 
  pred <- predict(model, dmatrix)                  
  return(pred)
}

predictor <- Predictor$new(
  model = xgb_model,
  data = x_train,
  y = y_train_numeric,  
  predict.fun = predict_xgb  
)

sample_instance <- x_test[1, , drop = FALSE]  
shap <- Shapley$new(predictor, x.interest = sample_instance)

print(shap$results)

plot(shap)

feature_importance <- FeatureImp$new(predictor, loss = "mae")

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
  mse <- mean((pred_numeric - y_numeric)^2)
  mae <- mean(abs(pred_numeric - y_numeric))

  metrics <- data.frame(
    Metric = c("Precision", "Recall", "F1-Score", "Accuracy", "MSE", "MAE"),
    Value = round(c(precision, recall, f1_score, accuracy, mse, mae), 3)
  )
  
  return(metrics)
}
library(knitr)
library(kableExtra)

xgb_probabilities <- predict(xgb_model, dtest)  
xgb_classifications <- ifelse(xgb_probabilities > 0.5, "High", "Low")  
xgb_metrics_table <- calculate_metrics(
  predictions = xgb_classifications, 
  y_true = y_test,                    
  probabilities = xgb_probabilities   
)

print(xgb_metrics_table)

write.csv(xgb_metrics_table, "XGBoost_Metrics.csv", row.names = FALSE)

plot_metrics <- function(metrics_table) {
  ggplot(metrics_table, aes(x = Metric, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    geom_text(aes(label = Value), vjust = -0.5) +
    theme_minimal() +
    labs(
      title = "Classification Metrics",
      y = "Value",
      x = "Metric"
    ) +
    scale_fill_brewer(palette = "Set3")
}

plot_metrics(xgb_metrics_table)



