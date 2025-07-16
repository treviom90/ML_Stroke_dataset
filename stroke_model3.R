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
library(rpart)
# --- Carga y preprocesamiento datos ---
data <- read.csv("C:/Users/cecil/OneDrive/Documentos/GitHub_Proyectos/ML_Proyecto_Final/healthcare-dataset-stroke-data.csv",
                 sep = ";", na.strings = c("N/A"))
print(data)
#------------------------------------------------------------------------------------
# Análisis exploratorio inicial de los datos
dfSummary(data)  # Reporte detallado de estadísticos y gráficos por variable
summary(data)    # Estadísticas descriptivas básicas (mín, máx, medias, etc.)
str(data)        # Estructura del dataset: tipos y muestra de valores

# Comentarios:
# - summary() permite ver valores faltantes y distribuciones básicas.
# - str() confirma tipos de variables (numéricas, factores).
# - dfSummary() ayuda a detectar outliers, missing data y patrones.
#------------------------------------------------------------------------------------
# Revisión de valores faltantes por columna
colSums(is.na(data))
#---------------------------------------------------------------------------------------
# Paso 1: Separar los datos con y sin valores faltantes en bmi
missing_bmi <- data[is.na(data$bmi), ]
complete_bmi <- data[!is.na(data$bmi), ]

# Paso 2: Convertir gender a factor o numérico como en Python
# (Aquí un ejemplo simple, puede ser factor)
complete_bmi$gender <- factor(complete_bmi$gender, levels = c("Male", "Female", "Other"))
missing_bmi$gender <- factor(missing_bmi$gender, levels = levels(complete_bmi$gender))

# Paso 3: Entrenar árbol de decisión para predecir bmi con age y gender
tree_model <- rpart(bmi ~ age + gender, data = complete_bmi, method = "anova", control = rpart.control(cp = 0.01))

# Paso 4: Predecir bmi faltante
predicted_bmi <- predict(tree_model, newdata = missing_bmi)

# Paso 5: Asignar valores predichos al dataframe original
data$bmi[is.na(data$bmi)] <- predicted_bmi

colSums(is.na(data))
## Imputación simple: reemplazar NA en BMI con la media de BMI (para evitar perder datos)
#data$bmi[is.na(data$bmi)] <- mean(data$bmi, na.rm = TRUE)
#------------------------------------------------------------------------------------
# Conversión a factores (excepto id que no es categórica)
categorical_vars <- c("stroke","gender","hypertension","heart_disease","ever_married",
                      "work_type","Residence_type","smoking_status")

data[categorical_vars] <- lapply(data[categorical_vars], factor)

#------------------------------------------------------------------------------------
# Análisis descriptivo gráfico

# Densidades para variables numéricas según clase stroke (para ver diferencias en distribución)
ggplot(data, aes(x=age, fill=stroke)) + geom_density(alpha=0.5)
ggplot(data, aes(x=avg_glucose_level, fill=stroke)) + geom_density(alpha=0.5)
ggplot(data, aes(x=bmi, fill=stroke)) + geom_density(alpha=0.5)

# Boxplots para comparar valores numéricos por grupo stroke
ggplot(data, aes(x=stroke, y=age)) + geom_boxplot()
ggplot(data, aes(x=stroke, y=avg_glucose_level)) + geom_boxplot()

# Barras proporcionales para variables categóricas según stroke
ggplot(data, aes(x=gender, fill=stroke)) + geom_bar(position = "fill")
ggplot(data, aes(x=ever_married, fill=stroke)) + geom_bar(position = "fill")

# Matriz de correlación visual entre variables numéricas relevantes
data_numeric <- data %>%
  dplyr::select(age, avg_glucose_level, bmi) %>%
  na.omit()
ggpairs(data_numeric)

#------------------------------------------------------------------------------------
# División entrenamiento/prueba
set.seed(123)
trainIndex <- createDataPartition(data$stroke, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
#------------------------------------------------------------------------------------
# --- Convertir variable respuesta a factores "No" y "Yes" para caret y pROC ---
trainData$stroke <- factor(ifelse(trainData$stroke == 1, "Yes", "No"), levels = c("No", "Yes"))
testData$stroke <- factor(ifelse(testData$stroke == 1, "Yes", "No"), levels = c("No", "Yes"))
#------------------------------------------------------------------------------------
# --- Modelos ---

# 1) GLM completo + selección stepwise
glm_fit <- glm(stroke ~ ., data = trainData, family = binomial)
glm_step <- stepAIC(glm_fit, direction = "both")
summary(glm_step)

# 2) GLM con ROSE + stepwise
data_rose <- ROSE(stroke ~ ., data = trainData, seed = 123)$data
glm_rose_fit <- glm(stroke ~ ., data = data_rose, family = binomial)
glm_rose_step <- stepAIC(glm_rose_fit, direction = "both")
summary(glm_rose_step)

# 3) GAM con selección automática de suavizados
modelo_gam <- gam(stroke ~ s(age) + s(avg_glucose_level) + s(bmi) + gender + 
                    hypertension + heart_disease + ever_married + work_type + 
                    Residence_type + smoking_status,
                  data = trainData, family = binomial, select = TRUE)
summary(modelo_gam)
#------------------------------------------------------------------------------------
# --- Función para sincronizar niveles factores en testData ---
factor_vars <- c('gender', 'hypertension', 'heart_disease', 'work_type',
                 'ever_married', 'Residence_type', 'smoking_status')

sync_factor_levels <- function(train_df, test_df, factor_vars) {
  for (fv in factor_vars) {
    test_df[[fv]] <- factor(test_df[[fv]], levels = levels(train_df[[fv]]))
  }
  return(test_df)
}
#------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------
# Diagnóstico de residuos
par(mfrow = c(2, 2))

# 1) GLM stepwise
plot(glm_step, main = "Residuos - GLM Stepwise")

# 2) GLM ROSE
plot(glm_rose, main = "Residuos - GLM ROSE")

par(mfrow = c(1, 1))       # restablece layout

# 3) GAM
# Gráficos de efectos suavizados y residuos
par(mfrow = c(2, 2))
plot(modelo_gam, residuals = TRUE, se = TRUE, pch = 16)
par(mfrow = c(1, 1))
#------------------------------------------------------------------------------------
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

