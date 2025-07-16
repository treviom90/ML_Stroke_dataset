# --- Carga librerías ---
library(tidyverse)
library(ggplot2)
library(summarytools)
library(mice)
library(ggpubr)
library(GGally)
library(mgcv)
library(caret)
library(pROC)
library(MASS)
library(ROSE)
library(devtools)
library(DMwR2)

# --- Carga y preprocesamiento datos ---
data <- read.csv("C:/Users/cecil/OneDrive/Documentos/GitHub_Proyectos/ML_Proyecto_Final/healthcare-dataset-stroke-data.csv",
                 sep = ";", na.strings = c("N/A"))

# Imputación simple para BMI
data$bmi[is.na(data$bmi)] <- mean(data$bmi, na.rm = TRUE)

# Conversión a factores (excepto id que no es categórica)
categorical_vars <- c("stroke","gender","hypertension","heart_disease","ever_married",
                      "work_type","Residence_type","smoking_status")
data[categorical_vars] <- lapply(data[categorical_vars], factor)

# División entrenamiento/prueba
set.seed(123)
trainIndex <- createDataPartition(data$stroke, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# --- Convertir variable respuesta a factores "No" y "Yes" para caret y pROC ---
trainData$stroke <- factor(ifelse(trainData$stroke == 1, "Yes", "No"), levels = c("No", "Yes"))
testData$stroke <- factor(ifelse(testData$stroke == 1, "Yes", "No"), levels = c("No", "Yes"))

# --- Modelos ---

# Modelo GLM completo + stepwise
glm_fit <- glm(stroke ~ ., data = trainData, family = binomial)
glm_step <- stepAIC(glm_fit, direction = "both")

# Modelo GLM con balanceo ROSE
data_rose <- ROSE(stroke ~ ., data = trainData, seed = 123)$data
glm_rose <- glm(stroke ~ ., data = data_rose, family = binomial)

# Modelo GAM
modelo_gam <- gam(stroke ~ s(age) + s(avg_glucose_level) + s(bmi) + gender + hypertension + heart_disease +
                    ever_married + work_type + Residence_type + smoking_status,
                  data = trainData, family = binomial)

# --- Función para sincronizar niveles factores en testData ---
factor_vars <- c("gender", "hypertension", "heart_disease", "ever_married",
                 "work_type", "Residence_type", "smoking_status")

sync_factor_levels <- function(train_df, test_df, factor_vars) {
  for (fv in factor_vars) {
    test_df[[fv]] <- factor(test_df[[fv]], levels = levels(train_df[[fv]]))
  }
  return(test_df)
}

# Sincronizar niveles de factores en testData
testData <- sync_factor_levels(trainData, testData, factor_vars)

# Sincronizar niveles variable respuesta en testData
testData$stroke <- factor(testData$stroke, levels = levels(trainData$stroke))

# --- Predicciones ---
glm_probs <- predict(glm_step, newdata = testData, type = "response")
gam_probs <- predict(modelo_gam, newdata = testData, type = "response")
glm_rose_probs <- predict(glm_rose, newdata = testData, type = "response")

# --- Curvas ROC y AUC ---
roc_glm <- roc(testData$stroke, glm_probs, levels = c("No", "Yes"))
roc_gam <- roc(testData$stroke, gam_probs, levels = c("No", "Yes"))
roc_rose <- roc(testData$stroke, glm_rose_probs, levels = c("No", "Yes"))

auc_glm <- auc(roc_glm)
auc_gam <- auc(roc_gam)
auc_rose <- auc(roc_rose)

print(auc_glm)
print(auc_gam)
print(auc_rose)

# --- Matrices de confusión usando umbral 0.5 ---
glm_pred_class <- factor(ifelse(glm_probs > 0.5, "Yes", "No"), levels = c("No", "Yes"))
gam_pred_class <- factor(ifelse(gam_probs > 0.5, "Yes", "No"), levels = c("No", "Yes"))
rose_pred_class <- factor(ifelse(glm_rose_probs > 0.5, "Yes", "No"), levels = c("No", "Yes"))

conf_glm <- confusionMatrix(glm_pred_class, testData$stroke, positive = "Yes")
conf_gam <- confusionMatrix(gam_pred_class, testData$stroke, positive = "Yes")
conf_rose <- confusionMatrix(rose_pred_class, testData$stroke, positive = "Yes")

print(conf_glm)
print(conf_gam)
print(conf_rose)

# --- Curvas ROC comparativas ---
plot(roc_glm, col = "blue", print.auc = TRUE, main = "Curvas ROC - Modelos Stroke")
lines(roc_gam, col = "green", print.auc = TRUE)
lines(roc_rose, col = "red", print.auc = TRUE)
legend("bottomright", legend = c("GLM Stepwise", "GAM", "GLM ROSE"),
       col = c("blue", "green", "red"), lwd = 2)


# --- Tabla resumen de métricas ---
extraer_metricas <- function(conf, modelo_nombre) {
  data.frame(
    Modelo = modelo_nombre,
    Accuracy = round(conf$overall["Accuracy"], 4),
    Sensibilidad = round(conf$byClass["Sensitivity"], 4),
    Especificidad = round(conf$byClass["Specificity"], 4),
    Precision = round(conf$byClass["Precision"], 4),
    Valor_Negativo = round(conf$byClass["Neg Pred Value"], 4),
    Balanced_Accuracy = round(conf$byClass["Balanced Accuracy"], 4)
  )
}

tabla_glm <- extraer_metricas(conf_glm, "GLM Stepwise")
tabla_gam <- extraer_metricas(conf_gam, "GAM")
tabla_rose <- extraer_metricas(conf_rose, "GLM ROSE")

tabla_comparativa <- bind_rows(tabla_glm, tabla_gam, tabla_rose)
print(tabla_comparativa)

