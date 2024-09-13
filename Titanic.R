# Import Data
training_data <- read.csv("minichallenge_titanic-1661852450_modified-1726068537.csv", sep = ";", row.names = "PassengerId" )

# Remove column "X", is clone ID
df_remove_X <- training_data[ , !names(training_data) %in% c("X")]

# Check missing values
missing_values <- sapply(df_remove_X, function(x) sum(is.na(x)))

# Count average_age for male and female
average_age <- aggregate(Age ~ Sex, data = df_remove_X, FUN = mean, na.rm = TRUE)

library(dplyr)

# Fill missting values
df_remove_X <- df_remove_X %>%
  mutate(Age = ifelse(is.na(Age) & Sex == "male", 31, 
                      ifelse(is.na(Age) & Sex == "female", 28, Age)))

# Clean data
df_remove_Cabin <- df_remove_X %>% select(-Cabin)
df_cleaned <- df_remove_Cabin %>% select(-Name, -Ticket)

# Convert Sex to int dtype
df_cleaned$Sex <- ifelse(df_cleaned$Sex == "male", 1, 0)

# Convert Embarked to int dtype
df_cleaned$Embarked <- ifelse(df_cleaned$Embarked == "C", 1, 
                               ifelse(df_cleaned$Embarked == "Q", 2, 
                                      ifelse(df_cleaned$Embarked == "S", 3, NA)))

# Create cor_matrix
cor_matrix <- cor(df_cleaned, use = "complete.obs")

library(ggcorrplot)

# display cor_matrix
ggcorrplot(cor_matrix, lab = TRUE)

library(ggplot2)

# Check missed Embarked value
missing_value_Embarked <- sapply(df_cleaned, function(x) sum(is.na(x)))

# Create pre_traned_data
pre_traned_data <- na.omit(df_cleaned)

library(caret)

# Divide the data into training rooms and the test (80% for training)
trainIndex <- createDataPartition(pre_traned_data$Survived, p = 0.8, list = FALSE)
train_data <- pre_traned_data[trainIndex, ]
test_data <- pre_traned_data[-trainIndex, ]

library(rpart)
library(rpart.plot)

# Set seed for recovery
set.seed(123)

# Set parametrs Crossvalidation
train_control <- trainControl(method = "cv", number = 10)

# learnt desision tree
desision_cv_model <- train(Survived ~ ., 
                  data = train_data, 
                  method = "rpart",
                  parms = list(split = "information"),
                  trControl = train_control)

# Print result
print(desision_cv_model)

# predict with test_data
desision_test_predictions <- predict(desision_cv_model, test_data)

desision_test_predictions_round <- round(desision_test_predictions, 0)

# Create confusion_matrix (confusion matrix)
desision_confusion_matrix <- table(test_data$Survived, desision_test_predictions_round)

# Count accuracy
desision_accuracy <- sum(diag(desision_confusion_matrix)) / sum(desision_confusion_matrix)
print(paste("desision_tree_accuracy:", round(desision_accuracy * 100, 2), "%"))

library(xgboost)

# Prepare XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, -1]), label = train_data$Survived)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, -1]), label = test_data$Survived)

# We adjust the parameters for the model XGBoost
params <- list(
  objective = "binary:logistic",  # for binary classification
  eval_metric = "error",          # we use the error for estimation
  max_depth = 6,                  # tree depth
  eta = 0.1,                      # learning speed
  subsample = 0.8,                # part of the data for training at each iteration
  colsample_bytree = 0.8,         # feature proportion for tree training
  seed = 123
)

# Number of training rounds
nrounds <- 100

# We train the model XGBoost
xgb_model <- xgb.train(params = params, 
                       data = train_matrix, 
                       nrounds = nrounds, 
                       watchlist = list(train = train_matrix), 
                       verbose = 0)

# We predict for test data
xgb_test_predictions <- predict(xgb_model, test_matrix)

# We round the results to the nearest whole number (0 or 1)
xgb_test_predictions_round <- ifelse(xgb_test_predictions > 0.5, 1, 0)

# We create a confusion matrix
xgb_confusion_matrix <- table(test_data$Survived, xgb_test_predictions_round)

# Count xgb_accuracy
xgb_accuracy <- sum(diag(xgb_confusion_matrix)) / sum(xgb_confusion_matrix)

# Print xgboost_accuracy
print(paste("xgboost_accuracy:", round(xgb_accuracy * 100, 2), "%"))
