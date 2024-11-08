# Required packages
library(glmnet)    # For regularization methods
library(pROC)      # For ROC curve analysis
library(caret)     # For data splitting
library(dplyr)     # For data manipulation
library(ggplot2)   # For visualization
library(gridExtra) # For plot arrangement
library(MASS)      # For stepwise regression

# Function to generate synthetic wave dataset 
generate_wave_data <- function(n_samples = 33334, n_active = 21, n_noise = 100) {
  # Generate active variables based on wave patterns
  X_active <- matrix(rnorm(n_samples * n_active), nrow = n_samples)
  
  # Generate noise variables
  X_noise <- matrix(rnorm(n_samples * n_noise), nrow = n_samples)
  
  # Combine active and noise variables
  X <- cbind(X_active, X_noise)
  colnames(X) <- c(paste0("Active_", 1:n_active), paste0("Noise_", 1:n_noise))
  
  # Generate binary response (simplified version)
  prob <- plogis(rowSums(X_active[, 1:15] * rnorm(15)))
  y <- rbinom(n_samples, 1, prob)
  
  # Return data frame
  data.frame(X, y = y)  # Note: not converting to factor for linear regression
}

# Function to perform linear regression with different variable selection methods
train_linear_classifier <- function(X_train, y_train, X_test, y_test, method = "stepwise") {
  # Prepare data
  train_data <- data.frame(X_train, y = y_train)
  
  # Train model based on selected method
  if(method == "stepwise") {
    # Stepwise regression using AIC
    full_model <- lm(y ~ ., data = train_data)
    null_model <- lm(y ~ 1, data = train_data)
    step_model <- stepAIC(null_model, 
                          scope = list(lower = null_model, upper = full_model),
                          direction = "both",
                          trace = 0)
    final_model <- step_model
    
    # Get selected variables
    selected_vars <- names(coef(final_model))[-1]  # Remove intercept
    
  } else if(method == "ridge") {
    # Ridge regression
    x_matrix <- as.matrix(X_train)
    cv_fit <- cv.glmnet(x_matrix, y_train, alpha = 0)
    final_model <- glmnet(x_matrix, y_train, alpha = 0, lambda = cv_fit$lambda.min)
    selected_vars <- rownames(coef(final_model))[which(abs(coef(final_model)) > 1e-5)][-1]
    
  } else if(method == "elastic_net") {
    # Elastic Net (alpha = 0.5)
    x_matrix <- as.matrix(X_train)
    cv_fit <- cv.glmnet(x_matrix, y_train, alpha = 0.5)
    final_model <- glmnet(x_matrix, y_train, alpha = 0.5, lambda = cv_fit$lambda.min)
    selected_vars <- rownames(coef(final_model))[which(abs(coef(final_model)) > 1e-5)][-1]
  }
  
  # Make predictions
  if(method == "stepwise") {
    pred_train <- predict(final_model)
    pred_test <- predict(final_model, newdata = data.frame(X_test))
  } else {
    pred_train <- predict(final_model, newx = as.matrix(X_train))
    pred_test <- predict(final_model, newx = as.matrix(X_test))
  }
  
  # Convert predictions to probabilities using sigmoid function
  pred_prob_train <- 1/(1 + exp(-pred_train))
  pred_prob_test <- 1/(1 + exp(-pred_test))
  
  # Calculate performance metrics
  pred_class_test <- ifelse(pred_prob_test > 0.5, 1, 0)
  roc_obj <- roc(y_test, as.vector(pred_prob_test))
  auc_value <- auc(roc_obj)
  accuracy <- mean(pred_class_test == y_test)
  
  # Return results
  list(
    model = final_model,
    method = method,
    accuracy = accuracy,
    auc = auc_value,
    selected_variables = selected_vars,
    predictions_train = pred_prob_train,
    predictions_test = pred_prob_test,
    roc_obj = roc_obj
  )
}

# Function to create visualization plots
create_analysis_plots <- function(results, X_train, y_train) {
  # ROC Curve Plot
  roc_plot <- ggplot(data = data.frame(
    FPR = rev(results$roc_obj$specificities),
    TPR = rev(results$roc_obj$sensitivities)
  )) +
    geom_line(aes(x = 1 - FPR, y = TPR), color = "steelblue", size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    labs(
      title = paste("ROC Curve -", toupper(results$method)),
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    annotate("text", x = 0.75, y = 0.25,
             label = paste("AUC =", round(results$auc, 3))) +
    theme_minimal()
  
  # Classification Distribution Plot
  pred_dist <- data.frame(
    Prediction = as.vector(results$predictions_train),
    Actual = as.factor(y_train)
  )
  
  dist_plot <- ggplot(pred_dist, aes(x = Prediction, fill = Actual)) +
    geom_density(alpha = 0.5) +
    labs(
      title = "Distribution of Predictions by Class",
      x = "Predicted Probability",
      y = "Density"
    ) +
    scale_fill_manual(values = c("red", "blue")) +
    theme_minimal()
  
  # Variable Selection Plot
  if(results$method == "stepwise") {
    var_importance <- data.frame(
      Variable = names(coef(results$model))[-1],
      Coefficient = abs(coef(results$model)[-1])
    )
  } else {
    var_importance <- data.frame(
      Variable = rownames(coef(results$model))[-1],
      Coefficient = abs(as.vector(coef(results$model))[-1])
    )
  }
  
  var_imp_plot <- var_importance %>%
    arrange(desc(Coefficient)) %>%
    head(20) %>%
    ggplot(aes(x = reorder(Variable, Coefficient), y = Coefficient)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(
      title = "Top 20 Variable Importance",
      x = "Variables",
      y = "Absolute Coefficient Value"
    ) +
    theme_minimal()
  
  # Arrange plots
  grid.arrange(roc_plot, var_imp_plot, dist_plot, 
               layout_matrix = rbind(c(1,2), c(3,3)),
               heights = c(1, 0.8))
}

# Main analysis function
analyze_wave_dataset <- function(n_train = 10000, method = "stepwise") {
  # Generate data
  set.seed(123)
  full_data <- generate_wave_data()
  
  # Split into training and testing
  train_indices <- sample(1:nrow(full_data), n_train)
  train_data <- full_data[train_indices, ]
  test_data <- full_data[-train_indices, ]
  
  # Separate features and response
  X_train <- train_data[, !names(train_data) %in% "y"]
  y_train <- train_data$y
  X_test <- test_data[, !names(test_data) %in% "y"]
  y_test <- test_data$y
  
  # Train model and get results
  results <- train_linear_classifier(X_train, y_train, X_test, y_test, method)
  
  # Print summary
  cat("\nAnalysis Results (", toupper(method), "):\n", sep="")
  cat("----------------\n")
  cat("Number of selected variables:", length(results$selected_variables), "\n")
  cat("Classification accuracy:", round(results$accuracy, 4), "\n")
  cat("Area under ROC curve:", round(results$auc, 4), "\n")
  cat("\nSelected variables:\n")
  print(results$selected_variables)
  
  # Create and display visualizations
  create_analysis_plots(results, X_train, y_train)
  
  return(results)
}

# Compare different methods
methods <- c("stepwise", "ridge", "elastic_net")
results_list <- lapply(methods, function(m) {
  cat("\nRunning analysis with", m, "regression...\n")
  analyze_wave_dataset(method = m)
})

# Name the results
names(results_list) <- methods