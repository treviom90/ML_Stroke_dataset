## Lista de paquetes requeridos
#paquetes <- c("tidyverse", "ggplot2", "summarytools", "mice", "ggpubr", "GGally")

## Instalar paquetes que no estén ya instalados
#paquetes_faltantes <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
#if(length(paquetes_faltantes)) install.packages(paquetes_faltantes)

library(tidyverse)
library(ggplot2)
library(summarytools)
library(mice)         # imputación de NA
library(ggpubr)
library(GGally)

# Carga
data <- read.csv("D:/Tetra3/Modelos_Lineales/Proyecto_Final/healthcare-dataset-stroke-data.csv", sep = ";", na.strings = c("N/A"))
print(data)

# Vista general
dfSummary(data)
summary(data)
str(data)

# Revisar NAs
colSums(is.na(data))

# Imputar BMI usando media
data$bmi[is.na(data$bmi)] <- mean(data$bmi, na.rm = TRUE)

#Conversion de variables
data$stroke <- factor(data$stroke)
data$gender <- factor(data$gender)
data$hypertension <- factor(data$hypertension)
data$heart_disease <- factor(data$heart_disease)
data$ever_married <- factor(data$ever_married)
data$work_type <- factor(data$work_type)
data$Residence_type <- factor(data$Residence_type)
data$smoking_status <- factor(data$smoking_status)

#Analisis descriptivo y grafico
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
data_numeric <- data %>% select(age, avg_glucose_level, bmi) %>% na.omit()
ggpairs(data_numeric)

# Modelo1: GLM Logistico
modelo_log <- glm(stroke ~ age + avg_glucose_level + bmi + gender +
                    hypertension + heart_disease + ever_married +
                    work_type + Residence_type + smoking_status,
                  data = data, family = binomial)

summary(modelo_log)

#Modelo 2: GAM no parametrico
library(mgcv)

modelo_gam <- gam(stroke ~ s(age) + s(avg_glucose_level) + s(bmi) +
                    gender + hypertension + heart_disease + ever_married +
                    work_type + Residence_type + smoking_status,
                  data = data, family = binomial)

summary(modelo_gam)
plot(modelo_gam, pages=1, se=TRUE)

# Evaluacion y comparacion de modelos
#Predicciones y validación cruzada
library(caret)
set.seed(123)
trainIndex <- createDataPartition(data$stroke, p = .8, 
                                  list = FALSE, 
                                  times = 1)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]

# Reajustar modelos en entrenamiento
glm_fit <- glm(stroke ~ ., data=trainData, family=binomial)
gam_fit <- gam(stroke ~ s(age) + s(avg_glucose_level) + s(bmi) +
                 gender + hypertension + heart_disease + ever_married +
                 work_type + Residence_type + smoking_status,
               data = trainData, family = binomial)

#Metricas de desempeño
library(pROC)

# GLM
glm_probs <- predict(glm_fit, newdata=testData, type="response")
roc_glm <- roc(testData$stroke, glm_probs)
auc(roc_glm)

# GAM
gam_probs <- predict(gam_fit, newdata=testData, type="response")
roc_gam <- roc(testData$stroke, gam_probs)
auc(roc_gam)

# Comparar AIC
AIC(glm_fit)
AIC(gam_fit)
