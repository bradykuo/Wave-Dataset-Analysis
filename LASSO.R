# Install required packages if needed
# install.packages(c("glmnet", "pROC", "caret", "dplyr", "ggplot2", "gridExtra"))

# Required packages
library(glmnet)  # For LASSO regression
library(pROC)    # For ROC curve analysis
library(caret)   # For data splitting and preprocessing
library(dplyr)   # For data manipulation
library(ggplot2)
library(gridExtra)

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
  data.frame(X, y = factor(y))
}

# Function to perform LASSO-based classification
train_lasso_classifier <- function(X_train, y_train, X_test, y_test) {
  # Prepare matrix format for glmnet
  x_train_matrix <- as.matrix(X_train)
  x_test_matrix <- as.matrix(X_test)
  
  # Fit LASSO model with cross-validation
  cv_fit <- cv.glmnet(x_train_matrix, y_train, 
                      family = "binomial",
                      alpha = 1,  # LASSO penalty
                      nfolds = 10)
  
  # Get best lambda
  best_lambda <- cv_fit$lambda.min
  
  # Fit final model with best lambda
  final_model <- glmnet(x_train_matrix, y_train,
                        family = "binomial",
                        alpha = 1,
                        lambda = best_lambda)
  
  # Make predictions for both train and test
  pred_prob_train <- predict(final_model, x_train_matrix, type = "response")
  pred_prob_test <- predict(final_model, x_test_matrix, type = "response")
  pred_class_test <- ifelse(pred_prob_test > 0.5, 1, 0)
  
  # Calculate performance metrics
  roc_obj <- roc(y_test, as.vector(pred_prob_test))
  auc_value <- auc(roc_obj)
  accuracy <- mean(pred_class_test == y_test)
  
  # Get non-zero coefficients (selected variables)
  coef_matrix <- as.matrix(coef(final_model))
  selected_vars <- rownames(coef_matrix)[which(coef_matrix != 0)]
  
  # Return results
  list(
    model = final_model,
    cv_fit = cv_fit,  # Added for cross-validation plot
    lambda = best_lambda,
    accuracy = accuracy,
    auc = auc_value,
    selected_variables = selected_vars,
    predictions_train = pred_prob_train,
    predictions_test = pred_prob_test,
    roc_obj = roc_obj,
    coef_matrix = coef_matrix
  )
}

# Function to create visualization plots
create_analysis_plots <- function(results, X_train, y_train) {
  # 1. ROC Curve Plot
  roc_plot <- ggplot(data = data.frame(
    FPR = rev(results$roc_obj$specificities),
    TPR = rev(results$roc_obj$sensitivities)
  )) +
    geom_line(aes(x = 1 - FPR, y = TPR), color = "steelblue", size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    labs(
      title = "ROC Curve",
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    annotate("text", x = 0.75, y = 0.25,
             label = paste("AUC =", round(results$auc, 3))) +
    theme_minimal()
  
  # 2. Variable Importance Plot
  var_importance <- data.frame(
    Variable = rownames(results$coef_matrix),
    Coefficient = abs(as.vector(results$coef_matrix))
  ) %>%
    filter(Variable != "(Intercept)") %>%
    arrange(desc(Coefficient)) %>%
    filter(Coefficient > 0) %>%
    head(20)  # Top 20 variables
  
  var_imp_plot <- ggplot(var_importance, 
                         aes(x = reorder(Variable, Coefficient), 
                             y = Coefficient)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(
      title = "Top 20 Variable Importance",
      x = "Variables",
      y = "Absolute Coefficient Value"
    ) +
    theme_minimal()
  
  # 3. Cross-validation Plot
  cv_data <- data.frame(
    Lambda = results$cv_fit$lambda,
    MSE = results$cv_fit$cvm,
    MSE_hi = results$cv_fit$cvm + results$cv_fit$cvsd,
    MSE_lo = results$cv_fit$cvm - results$cv_fit$cvsd
  )
  
  cv_plot <- ggplot(cv_data, aes(x = log(Lambda))) +
    geom_line(aes(y = MSE), color = "steelblue") +
    geom_ribbon(aes(ymin = MSE_lo, ymax = MSE_hi), alpha = 0.2) +
    geom_vline(xintercept = log(results$cv_fit$lambda.min), 
               linetype = "dashed", color = "red") +
    geom_vline(xintercept = log(results$cv_fit$lambda.1se), 
               linetype = "dashed", color = "blue") +
    labs(
      title = "Cross-validation Results",
      x = "Log(Lambda)",
      y = "Binomial Deviance"
    ) +
    theme_minimal()
  
  # 4. Classification Distribution Plot
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
  
  # Arrange all plots in a grid
  grid.arrange(roc_plot, var_imp_plot, cv_plot, dist_plot, 
               ncol = 2, nrow = 2)
}

# Main analysis function
analyze_wave_dataset <- function(n_train = 10000) {
  # Generate data
  set.seed(123)  # For reproducibility
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
  results <- train_lasso_classifier(X_train, y_train, X_test, y_test)
  
  # Print summary
  cat("\nAnalysis Results:\n")
  cat("----------------\n")
  cat("Number of selected variables:", length(results$selected_variables) - 1, "\n")
  cat("Classification accuracy:", round(results$accuracy, 4), "\n")
  cat("Area under ROC curve:", round(results$auc, 4), "\n")
  cat("\nSelected variables:\n")
  print(results$selected_variables[-1])  # Exclude intercept
  
  # Create and display visualizations
  create_analysis_plots(results, X_train, y_train)
  
  return(results)
}

# Run the analysis
results <- analyze_wave_dataset()