#install.packages("devtools")
#devtools::install_github("cran/DMwR2")
library(tidyverse)
library(ggplot2)
library(summarytools)
library(mice)         # imputación de NA
library(ggpubr)
library(GGally)
library(mgcv)
library(caret)
library(pROC)
library(MASS)         # stepAIC
library(ROSE)         # balanceo ROSE
#library(DMwR)         # SMOTE
library(devtools)
library(DMwR2)
#------------------------------------------------------------------------------------
# Carga
data <- read.csv("C:/Users/cecil/OneDrive/Documentos/GitHub_Proyectos/ML_Proyecto_Final/healthcare-dataset-stroke-data.csv", 
                 sep = ";", na.strings = c("N/A"))
print(data)
#------------------------------------------------------------------------------------
# Vista general
dfSummary(data)
summary(data)
str(data)
#La función \texttt{summary()} proporcionó estadísticas descriptivas básicas, tales como mínimos,
#máximos, medias, medianas y conteos para variables numéricas y categóricas. Esto permitió identificar 
#rápidamente la distribución de las variables y detectar posibles valores faltantes.

#Mediante \texttt{str()} se examinó la estructura del dataset, confirmando el tipo de dato asignado a 
#cada variable (numérico, factor, etc.) y visualizando un pequeño muestreo de sus valores. 

#Esto aseguró que las variables estuvieran correctamente tipificadas para el posterior modelado.
#Finalmente, \texttt{dfSummary()}, del paquete \texttt{summarytools}, generó un reporte enriquecido
#con estadísticas detalladas y mini-gráficos de distribución para cada variable. 

#Este análisis facilitó la identificación de valores atípicos, proporciones de datos faltantes y patrones 
#relevantes en el dataset, contribuyendo a la selección de las técnicas adecuadas de limpieza y preprocesamiento.
#------------------------------------------------------------------------------------
# Revisar NAs
colSums(is.na(data))

# Imputar BMI usando media
data$bmi[is.na(data$bmi)] <- mean(data$bmi, na.rm = TRUE)
#------------------------------------------------------------------------------------
# Conversion de variables categoricas a numericas
data$stroke <- factor(data$stroke)
data$gender <- factor(data$gender)
data$hypertension <- factor(data$hypertension)
data$heart_disease <- factor(data$heart_disease)
data$ever_married <- factor(data$ever_married)
data$work_type <- factor(data$work_type)
data$Residence_type <- factor(data$Residence_type)
data$smoking_status <- factor(data$smoking_status)
#------------------------------------------------------------------------------------
# Analisis descriptivo y grafico
# Histogramas y densidades
ggplot(data, aes(x=age, fill=stroke)) + geom_density(alpha=0.5)
ggplot(data, aes(x=avg_glucose_level, fill=stroke)) + geom_density(alpha=0.5)
ggplot(data, aes(x=bmi, fill=stroke)) + geom_density(alpha=0.5)

# Boxplot
ggplot(data, aes(x=stroke, y=age)) + geom_boxplot()
ggplot(data, aes(x=stroke, y=avg_glucose_level)) + geom_boxplot()

# Gráfico de proporciones
ggplot(data, aes(x=gender, fill=stroke)) + geom_bar(position = "fill")
ggplot(data, aes(x=ever_married, fill=stroke)) + geom_bar(position = "fill")

# Correlación numérica
data_numeric <- data %>%
  dplyr::select(age, avg_glucose_level, bmi) %>%
  na.omit()
ggpairs(data_numeric)
#------------------------------------------------------------------------------------
# Partición para entrenamiento y prueba
set.seed(123)
trainIndex <- createDataPartition(data$stroke, p = .8, list = FALSE)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]
#------------------------------------------------------------------------------------
# --- Manejo del desbalance ---

# ROSE
data_rose <- ROSE(stroke ~ ., data = trainData, seed = 123)$data
table(data_rose$stroke)

# --- Modelos con datos originales ---

# GLM completo
glm_fit <- glm(stroke ~ ., data=trainData, family=binomial)
#------------------------------------------------------------------------------------
# Selección de variables con stepAIC
glm_step <- stepAIC(glm_fit, direction = "both")
summary(glm_step)
#------------------------------------------------------------------------------------
# GAM con variables seleccionadas (ajustar según stepAIC)
modelo_gam <- gam(stroke ~ s(age) + s(avg_glucose_level) + s(bmi) + gender + hypertension + heart_disease + ever_married + work_type + Residence_type + smoking_status,
                  data = trainData, family = binomial)
summary(modelo_gam)
plot(modelo_gam, pages=1, se=TRUE)
#------------------------------------------------------------------------------------
# --- Modelos con datos balanceados (ROSE) ---
glm_rose <- glm(stroke ~ ., data = data_rose, family = binomial)
summary(glm_rose)
#------------------------------------------------------------------------------------
# --- Predicciones en test ---

# GLM paso a paso
glm_probs <- predict(glm_step, newdata=testData, type="response")
roc_glm <- roc(testData$stroke, glm_probs)
auc_glm <- auc(roc_glm)

# GAM
gam_probs <- predict(modelo_gam, newdata=testData, type="response")
roc_gam <- roc(testData$stroke, gam_probs)
auc_gam <- auc(roc_gam)

# GLM balanceado (ROSE)
glm_rose_probs <- predict(glm_rose, newdata=testData, type="response")
roc_rose <- roc(testData$stroke, glm_rose_probs)
auc_rose <- auc(roc_rose)

# --- Evaluación con matriz de confusión (umbral 0.5) ---

glm_pred_class <- factor(ifelse(glm_probs > 0.5, 1, 0))
gam_pred_class <- factor(ifelse(gam_probs > 0.5, 1, 0))
rose_pred_class <- factor(ifelse(glm_rose_probs > 0.5, 1, 0))

conf_glm <- confusionMatrix(glm_pred_class, testData$stroke, positive = "1")
conf_gam <- confusionMatrix(gam_pred_class, testData$stroke, positive = "1")
conf_rose <- confusionMatrix(rose_pred_class, testData$stroke, positive = "1")

print(conf_glm)
print(conf_gam)
print(conf_rose)

# --- Comparación AIC ---

cat("AIC GLM stepwise: ", AIC(glm_step), "\n")
cat("AIC GLM balanceado (ROSE): ", AIC(glm_rose), "\n")

# --- Curvas ROC comparativas ---

plot(roc_glm, col = "blue", print.auc = TRUE, main = "Curvas ROC - Modelos Stroke")
lines(roc_gam, col = "green", print.auc = TRUE)
lines(roc_rose, col = "red", print.auc = TRUE)
legend("bottomright", legend = c("GLM Stepwise", "GAM", "GLM ROSE"),
       col = c("blue", "green", "red"), lwd = 2)

