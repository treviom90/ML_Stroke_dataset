# Instalación y carga de librerías necesarias para análisis y modelado
# install.packages("devtools")
# devtools::install_github("cran/DMwR2")

library(tidyverse)    # Para manipulación y visualización de datos
library(ggplot2)      # Gráficos
library(summarytools) # Reportes descriptivos detallados
library(mice)         # Imputación de datos faltantes
library(ggpubr)       # Funciones gráficas adicionales
library(GGally)       # Paquete para gráficos exploratorios (ggpairs)
library(mgcv)         # Modelos aditivos generalizados (GAM)
library(caret)        # Modelado y evaluación de modelos
library(pROC)         # Curvas ROC y AUC
library(MASS)         # stepAIC para selección de variables
library(ROSE)         # Técnicas para balancear datasets desbalanceados
# library(DMwR)       # SMOTE (no se usa aquí)
library(devtools)     # Para instalar paquetes desde GitHub
library(DMwR2)        # SMOTE actualizado

#------------------------------------------------------------------------------------
# Carga del dataset de accidentes cerebrovasculares (stroke)
data <- read.csv("C:/Users/cecil/OneDrive/Documentos/GitHub_Proyectos/ML_Proyecto_Final/healthcare-dataset-stroke-data.csv", 
                 sep = ";", na.strings = c("N/A"))
print(data)  # Muestra las primeras filas para revisión rápida

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

# Imputación simple: reemplazar NA en BMI con la media de BMI (para evitar perder datos)
data$bmi[is.na(data$bmi)] <- mean(data$bmi, na.rm = TRUE)

#------------------------------------------------------------------------------------
# Conversión de variables categóricas a factores (necesario para modelado)
data$stroke <- factor(data$stroke)
data$gender <- factor(data$gender)
data$hypertension <- factor(data$hypertension)
data$heart_disease <- factor(data$heart_disease)
data$ever_married <- factor(data$ever_married)
data$work_type <- factor(data$work_type)
data$Residence_type <- factor(data$Residence_type)
data$smoking_status <- factor(data$smoking_status)

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
# División de datos en conjunto de entrenamiento (80%) y prueba (20%)
set.seed(123)  # Para reproducibilidad
trainIndex <- createDataPartition(data$stroke, p = .8, list = FALSE)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]

#------------------------------------------------------------------------------------
# --- Modelos con datos originales ---

# Modelo logístico completo (con todas las variables)
glm_fit <- glm(stroke ~ ., data=trainData, family=binomial)

#------------------------------------------------------------------------------------
# Selección de variables por paso hacia adelante y atrás usando AIC
glm_step <- stepAIC(glm_fit, direction = "both")
summary(glm_step)
#------------------------------------------------------------------------------------
# --- Manejo del desbalance en datos con técnica ROSE ---

data_rose <- ROSE(stroke ~ ., data = trainData, seed = 123)$data
table(data_rose$stroke)  # Ver tabla balanceada

#------------------------------------------------------------------------------------
# Modelo logístico usando datos balanceados por ROSE
glm_rose <- glm(stroke ~ ., data = data_rose, family = binomial)
summary(glm_rose)
#------------------------------------------------------------------------------------
# Modelo aditivo generalizado (GAM) Datos originales y sin balance
modelo_gam <- gam(stroke ~ s(age) + s(avg_glucose_level) + s(bmi) + gender + hypertension + heart_disease + ever_married + work_type + Residence_type + smoking_status,
                  data = trainData, family = binomial)
summary(modelo_gam)
plot(modelo_gam, pages=1, se=TRUE)  # Visualización de efectos suavizados con intervalos de confianza
#------------------------------------------------------------------------------------
# --- Diagnóstico de residuos para cada modelo ---

# Diagnóstico para GLM Stepwise
par(mfrow = c(2, 2))
plot(glm_step, main = "Diagnóstico - GLM Stepwise")
par(mfrow = c(1, 1))

# Diagnóstico para GAM
par(mfrow = c(1, 2))
plot(modelo_gam, residuals = TRUE, pch = 19)  # Efectos suavizados con residuos
gam.check(modelo_gam)                        # Verifica supuestos del modelo GAM
par(mfrow = c(1, 1))

# Diagnóstico para GLM balanceado con ROSE
par(mfrow = c(2, 2))
plot(glm_rose, main = "Diagnóstico - GLM ROSE")
par(mfrow = c(1, 1))
#------------------------------------------------------------------------------------
# --- Predicciones sobre conjunto de prueba ---

# Predicciones probabilísticas para cada modelo
glm_probs <- predict(glm_step, newdata=testData, type="response")       # GLM stepwise
gam_probs <- predict(modelo_gam, newdata=testData, type="response")     # GAM
glm_rose_probs <- predict(glm_rose, newdata=testData, type="response")  # GLM ROSE

# Cálculo de curvas ROC y AUC para evaluar desempeño
roc_glm <- roc(testData$stroke, glm_probs)
auc_glm <- auc(roc_glm)

roc_gam <- roc(testData$stroke, gam_probs)
auc_gam <- auc(roc_gam)

roc_rose <- roc(testData$stroke, glm_rose_probs)
auc_rose <- auc(roc_rose)

print(auc_glm)
print(auc_gam)
print(auc_rose)

# Opcional: guardar predicciones en archivo CSV
predicciones <- data.frame(ID = testData$id,
                           Real = testData$stroke,
                           GLM = glm_probs,
                           GAM = gam_probs,
                           ROSE = glm_rose_probs)

write.csv(predicciones, "C:/Users/cecil/OneDrive/Documentos/GitHub_Proyectos/ML_Proyecto_Final/predicciones_modelos.csv", row.names = FALSE)

#------------------------------------------------------------------------------------
# --- Evaluación con matriz de confusión (usando umbral de 0.5 para clasificar) ---

glm_pred_class <- factor(ifelse(glm_probs > 0.5, 1, 0))       # Clases predichas GLM stepwise
gam_pred_class <- factor(ifelse(gam_probs > 0.5, 1, 0))       # Clases predichas GAM
rose_pred_class <- factor(ifelse(glm_rose_probs > 0.5, 1, 0)) # Clases predichas GLM ROSE

# Matrices de confusión y métricas de desempeño (sensibilidad, especificidad, etc.)
conf_glm <- confusionMatrix(glm_pred_class, testData$stroke, positive = "1")
conf_gam <- confusionMatrix(gam_pred_class, testData$stroke, positive = "1")
conf_rose <- confusionMatrix(rose_pred_class, testData$stroke, positive = "1")

print(conf_glm)
print(conf_gam)
print(conf_rose)

#------------------------------------------------------------------------------------
# --- Comparación de la bondad de ajuste con AIC ---

cat("AIC GLM stepwise: ", AIC(glm_step), "\n")
cat("AIC GLM balanceado (ROSE): ", AIC(glm_rose), "\n")

#------------------------------------------------------------------------------------
# --- Curvas ROC comparativas para los tres modelos ---

plot(roc_glm, col = "blue", print.auc = TRUE, main = "Curvas ROC - Modelos Stroke")
lines(roc_gam, col = "green", print.auc = TRUE)
lines(roc_rose, col = "red", print.auc = TRUE)
legend("bottomright", legend = c("GLM Stepwise", "GAM", "GLM ROSE"),
       col = c("blue", "green", "red"), lwd = 2)

