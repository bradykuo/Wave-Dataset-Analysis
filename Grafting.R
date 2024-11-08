# Reference: https://www.sciencedirect.com/science/article/pii/S0167947318301968
# Required packages
library(glmnet)    # For regularization
library(pROC)      # For ROC curve analysis
library(caret)     # For data splitting
library(dplyr)     # For data manipulation
library(ggplot2)   # For visualization
library(gridExtra) # For plot arrangement

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

# Function to calculate gradient for grafting
calculate_gradient <- function(X, y, p) {
  # Returns absolute gradient for each variable
  colSums(sweep(X, 1, (y - p), "*"))
}

# Function to perform grafting-based variable selection
train_grafting_classifier <- function(X_train, y_train, X_test, y_test, 
                                      epsilon = 1e-5, max_iter = 100) {
  n <- nrow(X_train)
  p <- ncol(X_train)
  
  # Convert factor to numeric
  y_train <- as.numeric(as.character(y_train))
  y_test <- as.numeric(as.character(y_test))
  
  # Initialize
  active_set <- c()
  beta <- rep(0, p)
  
  # Initial predictions
  p_hat <- rep(0.5, n)
  
  for(iter in 1:max_iter) {
    # Calculate gradients for inactive variables
    inactive_set <- setdiff(1:p, active_set)
    if(length(inactive_set) == 0) break
    
    gradients <- abs(calculate_gradient(X_train[, inactive_set, drop = FALSE], 
                                        y_train, p_hat))
    
    # Find variable with largest gradient
    max_grad_idx <- which.max(gradients)
    if(gradients[max_grad_idx] < epsilon) break
    
    # Add variable to active set
    new_var <- inactive_set[max_grad_idx]
    active_set <- c(active_set, new_var)
    
    # Fit logistic regression on active set
    if(length(active_set) > 0) {
      X_active <- X_train[, active_set, drop = FALSE]
      df_train <- data.frame(X_active)
      names(df_train) <- paste0("V", 1:ncol(X_active))
      fit <- glm(y_train ~ ., data = df_train, family = binomial())
      beta[active_set] <- coef(fit)[-1]  # Exclude intercept
      p_hat <- predict(fit, type = "response")
    }
  }
  
  # Final model predictions
  if(length(active_set) > 0) {
    X_active_test <- X_test[, active_set, drop = FALSE]
    df_test <- data.frame(X_active_test)
    names(df_test) <- paste0("V", 1:ncol(X_active_test))
    
    # Prepare training data with same column names
    df_train <- data.frame(X_train[, active_set, drop = FALSE])
    names(df_train) <- paste0("V", 1:ncol(df_train))
    
    # Fit final model
    final_fit <- glm(y_train ~ ., data = df_train, family = binomial())
    pred_prob_test <- predict(final_fit, newdata = df_test, type = "response")
    pred_class_test <- ifelse(pred_prob_test > 0.5, 1, 0)
  } else {
    # Handle case where no variables are selected
    pred_prob_test <- rep(0.5, length(y_test))
    pred_class_test <- rep(0, length(y_test))
    final_fit <- NULL
  }
  
  # Calculate performance metrics
  roc_obj <- roc(y_test, pred_prob_test)
  auc_value <- auc(roc_obj)
  accuracy <- mean(pred_class_test == y_test)
  
  # Return results
  list(
    model = final_fit,
    active_variables = active_set,
    accuracy = accuracy,
    auc = auc_value,
    predictions_test = pred_prob_test,
    roc_obj = roc_obj,
    beta = beta
  )
}

# Main analysis function with grafting
analyze_wave_dataset_grafting <- function(n_train = 10000) {
  # Generate data
  set.seed(123)
  full_data <- generate_wave_data()
  
  # Split into training and testing
  train_indices <- sample(1:nrow(full_data), n_train)
  train_data <- full_data[train_indices, ]
  test_data <- full_data[-train_indices, ]
  
  # Separate features and response
  X_train <- as.matrix(train_data[, !names(train_data) %in% "y"])
  y_train <- train_data$y
  X_test <- as.matrix(test_data[, !names(test_data) %in% "y"])
  y_test <- test_data$y
  
  # Train model using grafting
  results <- train_grafting_classifier(X_train, y_train, X_test, y_test)
  
  # Print summary
  cat("\nGrafting Analysis Results:\n")
  cat("----------------\n")
  cat("Number of selected variables:", length(results$active_variables), "\n")
  cat("Classification accuracy:", round(results$accuracy, 4), "\n")
  cat("Area under ROC curve:", round(results$auc, 4), "\n")
  cat("\nSelected variables:\n")
  print(paste("Active_", results$active_variables, sep=""))
  
  # Create visualization
  create_grafting_plots(results, X_train, y_train)
  
  return(results)
}

# Visualization function
create_grafting_plots <- function(results, X_train, y_train) {
  # ROC Curve Plot
  roc_plot <- ggplot(data = data.frame(
    FPR = rev(results$roc_obj$specificities),
    TPR = rev(results$roc_obj$sensitivities)
  )) +
    geom_line(aes(x = 1 - FPR, y = TPR), color = "steelblue", size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    labs(
      title = "ROC Curve (Grafting)",
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    annotate("text", x = 0.75, y = 0.25,
             label = paste("AUC =", round(results$auc, 3))) +
    theme_minimal()
  
  # Variable Importance Plot
  var_importance <- data.frame(
    Variable = paste0("Active_", 1:length(results$beta)),
    Coefficient = abs(results$beta)
  ) %>%
    filter(Coefficient > 0) %>%
    arrange(desc(Coefficient)) %>%
    head(20)
  
  var_imp_plot <- ggplot(var_importance, 
                         aes(x = reorder(Variable, Coefficient), 
                             y = Coefficient)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(
      title = "Top 20 Variable Importance (Grafting)",
      x = "Variables",
      y = "Absolute Coefficient Value"
    ) +
    theme_minimal()
  
  # Arrange plots
  grid.arrange(roc_plot, var_imp_plot, ncol = 2)
}

# Run the analysis
grafting_results <- analyze_wave_dataset_grafting()