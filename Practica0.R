
library("mlr3data")
library("skimr")
library("DataExplorer")
library("corrplot")
library("caret")     
library("pROC")
library(rpart)
library(rpart.plot) 
library(ggplot2)
library(tidyverse)
library(C50)
library(mlr3)
library(mlr3learners)
library(mlr3extralearners)

#-----EDA-------------------------------------------------------------
# Visualización de la Base de Datos
View(ilpd)
summary(ilpd)
skimr::skim(ilpd) 
str(ilpd) # comprobación de tipos de variables
sapply(ilpd, function(x) sum(is.na(x)/length(x)))  # muestra si hay valores nulos

# Correlaciones
numericos<- sapply(ilpd, is.numeric) # escoge las columnas que tiene variables numericas
mat_cor<- cor(ilpd[,numericos]) # calculo de matrices de correlaciones 
corrplot(mat_cor, method = 'ellipse', type= 'upper') # representación grafica de matriz de correlaciones


#Transformaciones
set.seed(0) # establecemos una semilla para que los procesos aleatorios se puedan replicar
datos<-ilpd[,order(names(ilpd))] 
cols_int <- sapply(datos, is.integer)
datos[cols_int] <- lapply(datos[cols_int], as.numeric) #transformar los datos tipo int en float/num
str(datos) # comprobación de la transformación

#Holdout
idx_no  <- which(datos$diseased == "no") #separamos indices respecto a si están enfermos o no
idx_yes <- which(datos$diseased == "yes")

train_no  <- sample(idx_no, size = length(idx_no) * 0.75, replace = FALSE) # selección de individios (sanos) para la muestra train
train_yes <- sample(idx_yes, size = length(idx_yes) * 0.75, replace = FALSE)

indices_train <- sample(c(train_no, train_yes)) # unificamos los dos conjuntos de indices
indices_test  <- setdiff(1:nrow(datos), indices_train) # los indices que no son escogidos para indices_train se introduce a indices_test

datos_train <- datos[indices_train, ] #mediante los indices seleccionados formamos las muestras
datos_test  <- datos[indices_test, ]


#Separo variables predictoras
X_train <- datos_train[, names(datos_train) != "diseased"] 
y_train <- datos_train$diseased             

X_test  <- datos_test[, names(datos_test) != "diseased"]
y_test <- datos_test$diseased   

# Mostrar los 5 primeros índices del conjunto de entrenamiento
indices_train_5 <- head(indices_train, 5)
print(indices_train_5)

# Mostrar los 5 primeros índices del conjunto de test
indices_test_5 <- head(indices_test, 5) # los 5 primeros indices de test 
print(indices_test_5)

#-----rpart--------------------------------


modelo_optimo <- rpart(diseased ~ ., 
                       data = datos_train, 
                       method = "class", 
                       control = rpart.control(maxdepth = 3)) #usamos la función rpart para generar un modelo

# Predicciones para Train y Test
pred_train_opt <- predict(modelo_optimo, newdata = datos_train, type = "class") # calculamos las predicciones para las dos muestras
pred_test_opt  <- predict(modelo_optimo, newdata = datos_test, type = "class")

# Matrices de confusión con caret
cm_train <- confusionMatrix(pred_train_opt, datos_train$diseased, positive = "yes") # construimos dos matrices de confusión para mostrar los aciertos y errores
cm_test  <- confusionMatrix(pred_test_opt, datos_test$diseased, positive = "yes")

# Informe detallado comparativo
cat("\n=======================================================\n")
cat("=== INFORME DETALLADO MODELO ÓPTIMO (maxdepth = 3) ===\n")
cat("=======================================================\n\n")

cat("--- ENTRENAMIENTO (TRAIN) ---\n")
print(cm_train$table)
cat("Accuracy Global (Train)   :", round(cm_train$overall["Accuracy"], 4), "\n")
cat("Sensibilidad (Train)      :", round(cm_train$byClass["Sensitivity"], 4), "\n")
cat("Especificidad (Train)     :", round(cm_train$byClass["Specificity"], 4), "\n\n")

cat("--- EVALUACIÓN (TEST) ---\n")
print(cm_test$table)
cat("Accuracy Global (Test)    :", round(cm_test$overall["Accuracy"], 4), "\n")
cat("Sensibilidad (Test)       :", round(cm_test$byClass["Sensitivity"], 4), "\n")
cat("Especificidad (Test)      :", round(cm_test$byClass["Specificity"], 4), "\n")

#Representación del árbol

rpart.plot(modelo_optimo) # representación en forma de árbol del modelo

#Selección del valor del hiperparámetro

valores_maxdepth <- 1:15 
resultados_maxdepth <- data.frame(
  maxdepth      = numeric(),
  Acc_Train     = numeric(),
  Acc_Test      = numeric()
)

# usamos un bucle para ver los accurasys que obtenemos en función del valor del hiperparámetro
for (depth_i in valores_maxdepth) {
  modelo_temp <- rpart(diseased ~ ., 
                       data = datos_train, 
                       method = "class", 
                       control = rpart.control(maxdepth = depth_i))
  
  pred_train <- predict(modelo_temp, newdata = datos_train, type = "class")
  pred_test  <- predict(modelo_temp, newdata = datos_test, type = "class")
  
  # Accuracy estándar
  acc_tr <- sum(diag(table(datos_train$diseased, pred_train))) / nrow(datos_train)
  acc_te <- sum(diag(table(datos_test$diseased, pred_test))) / nrow(datos_test)
  
  resultados_maxdepth <- rbind(resultados_maxdepth, data.frame(
    maxdepth      = depth_i,
    Acc_Train     = round(acc_tr, 4),
    Acc_Test      = round(acc_te, 4)  ))
}

# Ordenar por mejor Acc_Test
resultados_maxdepth <- resultados_maxdepth[order(-resultados_maxdepth$Acc_Test), ]
print(resultados_maxdepth)

# Representación gráfica del proceso de elección del valor del hiperparámetro

datos_grafico_acc <- resultados_maxdepth %>% 
  pivot_longer(
    cols = c(Acc_Train, Acc_Test),
    names_to = "Conjunto",
    values_to = "Accuracy"
  )

datos_grafico_acc$Conjunto <- factor(
  datos_grafico_acc$Conjunto,
  levels = c("Acc_Train", "Acc_Test"),
  labels = c("Train", "Test")
)

g1 <- ggplot(datos_grafico_acc, aes(x = maxdepth, y = Accuracy, color = Conjunto)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_x_continuous(breaks = valores_maxdepth) +
  scale_color_manual(values = c("Train" = "#2B5C8F", "Test" = "#D95F02")) +
  labs(
    title = "Evolución del Accuracy Estándar según MaxDepth",
    x = "Profundidad máxima del árbol (maxdepth)",
    y = "Accuracy Estándar",
    color = "Conjunto"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 13),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(g1)

# Modelo final con todos los datos
modelo_final_rpart_r <- rpart(diseased ~ ., 
                              data = datos, 
                              method = "class", 
                              control = rpart.control(maxdepth = 3))

rpart.plot(modelo_final_rpart_r)


#-----C50-----

# Hemos aplicado el mismo procedimiento que rpart

mejor_minCases <- 8

modelo_c50_optimo <- C5.0(x = X_train,
                          y = y_train, #indicamos variable respuesta
                          control = C5.0Control(minCases = mejor_minCases))

# Plot del árbol (formato simplificado para que no se superponga el texto)
plot(modelo_c50_optimo, type = "s", main = paste("Árbol C5.0 Óptimo (minCases =", mejor_minCases, ")"))

# Predicciones finales
pred_train_c50 <- predict(modelo_c50_optimo, newdata = X_train)
pred_test_c50 <- predict(modelo_c50_optimo, newdata = X_test)

cm_train_c50   <- confusionMatrix(pred_train_c50, y_train, positive = "yes")
cm_test_c50   <- confusionMatrix(pred_test_c50, y_test, positive = "yes")

cat("\n=======================================================\n")
cat("=== INFORME DETALLADO MODELO ÓPTIMO C5.0 test ===\n")
cat("=======================================================\n")
print(cm_test_c50$table)
cat("Accuracy Global (Test)  :", round(cm_test_c50$overall["Accuracy"], 4), "\n")
cat("Sensibilidad (Test)     :", round(cm_test_c50$byClass["Sensitivity"], 4), "\n")
cat("Especificidad (Test)    :", round(cm_test_c50$byClass["Specificity"], 4), "\n")


cat("\n=======================================================\n")
cat("=== INFORME DETALLADO MODELO ÓPTIMO C5.0 train===\n")
cat("=======================================================\n")
print(cm_train_c50$table)
cat("Accuracy Global (Train)  :", round(cm_train_c50$overall["Accuracy"], 4), "\n")
cat("Sensibilidad (Train)     :", round(cm_train_c50$byClass["Sensitivity"], 4), "\n")
cat("Especificidad (Train)    :", round(cm_train_c50$byClass["Specificity"], 4), "\n")

valores_mincases <- 2:20

resultados_c50 <- data.frame(
  minCases      = numeric(),
  Acc_Train     = numeric(),
  Acc_Test      = numeric()
)


for (mc_i in valores_mincases) {
  # Entrenar C5.0 controlando el mínimo de casos por hoja
  modelo_temp <- C5.0(x = X_train,
                      y = y_train,
                      control = C5.0Control(minCases = mc_i))
  
  pred_train <- predict(modelo_temp, newdata = X_train, type = "class")
  pred_test  <- predict(modelo_temp, newdata = X_test, type = "class")
  
  # Accuracy estándar
  acc_tr <- sum(diag(table(datos_train$diseased, pred_train))) / nrow(X_train)
  acc_te <- sum(diag(table(datos_test$diseased, pred_test))) / nrow(X_test)
  
  resultados_c50 <- rbind(resultados_c50, data.frame(
    minCases      = mc_i,
    Acc_Train     = round(acc_tr, 4),
    Acc_Test      = round(acc_te, 4)
  ))
}

# Ordenar por mejor Accuracy en Test
resultados_c50 <- resultados_c50[order(-resultados_c50$Acc_Test), ]
print(resultados_c50)

datos_grafico_acc_c50 <- resultados_c50 %>%
  pivot_longer(
    cols = c(Acc_Train, Acc_Test),
    names_to = "Conjunto",
    values_to = "Accuracy"
  )

datos_grafico_acc_c50$Conjunto <- factor(
  datos_grafico_acc_c50$Conjunto,
  levels = c("Acc_Train", "Acc_Test"),
  labels = c("Train", "Test")
)

g1_c50 <- ggplot(datos_grafico_acc_c50, aes(x = minCases, y = Accuracy, color = Conjunto)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_x_continuous(breaks = valores_mincases) +
  scale_color_manual(values = c("Train" = "#2B5C8F", "Test" = "#1B9E77")) + # Verde para Test, manteniendo la estética
  labs(
    title = "Evolución del Accuracy Estándar según minCases (C5.0)",
    x = "Mínimo de casos por hoja (minCases)",
    y = "Accuracy Estándar",
    color = "Conjunto"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 13),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(g1_c50)

#Modelo final con todos los datos

modelo_c50_final_r <- C5.0(diseased ~ ., 
                           data = datos,
                           control = C5.0Control(minCases = mejor_minCases))

# Plot del árbol (formato simplificado para que no se superponga el texto)
plot(modelo_c50_final_r, type = "s", main = paste("Árbol C5.0 con todos los datos (minCases =", mejor_minCases, ")"))


# 5 primeras predicciones en Test rpart
head(pred_test_opt, 5)
# 5 primeras predicciones en Test C50
head(pred_test_c50, 5)

#-----Apartado MLR3 ----------------------------------------------------------------

set.seed(100524020)

#tarea

tarea <- as_task_classif(datos, target = "diseased") # creamos una tarea e indicamos la variable respuesta

#HOLDOUT con mlr3 (1/4 test)
tarea$set_col_roles("diseased", add_to = "stratum") # indicamos que los datos esta dividida en dos estratos
particion <- rsmp("holdout", ratio = 3/4)    # realizamos el holdout
particion$instantiate(tarea) #calcula los números de fila específicos para cada partición de entrenamiento y prueba respetando la estratificación

idx_train <- particion$train_set(1) #asignamos los indices
idx_test  <- particion$test_set(1)

cat("Tamaño train:", length(idx_train), "| Tamaño test:", length(idx_test), "\n")
print(round(prop.table(table(tarea$truth(idx_train))), 3))   # proporciones de clase (train)
print(round(prop.table(table(tarea$truth(idx_test))),  3))   # proporciones de clase (test)

#Learners con los hiperparámetros de R
learner_rpart <- lrn("classif.rpart", maxdepth = 3,  predict_type = "response") # Construimos los learners
learner_c50   <- lrn("classif.C50",   minCases = 8,  predict_type = "response")
#3) Entrenamiento con la partición de R 
learner_rpart$train(tarea, row_ids = idx_train) # Construimos el modelo
learner_c50$train(tarea,   row_ids = idx_train)

#4) Predicciones (train y test)
pr_rpart_tr <- learner_rpart$predict(tarea, row_ids = idx_train) #Realizamos las prediciones de rpart y C50 tanto para la muestra train como para la de test
pr_rpart_te <- learner_rpart$predict(tarea, row_ids = idx_test)
pr_c50_tr   <- learner_c50$predict(tarea,   row_ids = idx_train)
pr_c50_te   <- learner_c50$predict(tarea,   row_ids = idx_test)
#5) Métricas
# sensitivity = recall de la clase positiva "yes"; specificity = recall de "no"
medidas <- msrs(c("classif.acc", "classif.ce",
                  "classif.sensitivity", "classif.specificity")) #sacamos las medidas que nos interesan
res_mlr <- rbind(
  rpart_train = pr_rpart_tr$score(medidas),
  rpart_test  = pr_rpart_te$score(medidas),
  C50_train   = pr_c50_tr$score(medidas),
  C50_test    = pr_c50_te$score(medidas))

cat("\n=== RESULTADOS EN MLR3 ===\n"); print(round(res_mlr, 4)) #imprimimos los resultados

cat("\nMatriz de confusión TEST rpart (mlr3):\n"); print(pr_rpart_te$confusion)
cat("\nMatriz de confusión TEST C5.0  (mlr3):\n"); print(pr_c50_te$confusion)

#6) Criterios mínimos para la memoria
cat("\n--- ACCURACIES DE TEST EN MLR ---\n")
cat("Accuracy rpart:", round(pr_rpart_te$score(msr("classif.acc")), 4), "\n")
cat("Accuracy C5.0 :", round(pr_c50_te$score(msr("classif.acc")), 4), "\n")

cat("\n--- 5 PRIMEROS ÍNDICES EN MLR ---\n")
cat("Entrenamiento:", head(idx_train, 5), "\n")
cat("Test         :", head(pr_rpart_te$row_ids, 5), "\n")

cat("\n--- 5 PRIMERAS PREDICCIONES DE TEST EN MLR ---\n")
cat("rpart:", as.character(head(pr_rpart_te$response, 5)), "\n")
cat("C5.0 :", as.character(head(pr_c50_te$response, 5)), "\n")
#7) ¿Coincide con R?
if (exists("pred_test_opt") && exists("pred_test_c50")) {
  cat("\n--- COMPARACIÓN R vs MLR ---\n")
  cat("Índices train idénticos :", identical(sort(as.integer(indices_train)),
                                             sort(as.integer(pr_rpart_tr$row_ids))), "\n")
  cat("Índices test idénticos  :", identical(as.integer(indices_test),
                                             as.integer(pr_rpart_te$row_ids)), "\n")
  cat("Predicciones rpart igual:", identical(as.character(pred_test_opt),
                                             as.character(pr_rpart_te$response)), "\n")
  cat("Predicciones C5.0 igual :", identical(as.character(pred_test_c50),
                                             as.character(pr_c50_te$response)), "\n")
  cat("Acc test rpart  R vs mlr:", round(unname(cm_test$overall["Accuracy"]), 4), "vs",
      round(pr_rpart_te$score(msr("classif.acc")), 4), "\n")
  cat("Acc test C5.0   R vs mlr:", round(unname(cm_test_c50$overall["Accuracy"]), 4), "vs",
      round(pr_c50_te$score(msr("classif.acc")), 4), "\n")
}
#8) Visualización de los modelos entrenados con mlr3
rpart.plot(learner_rpart$model, main = "rpart (mlr3, maxdepth = 3)")

datos_train <- datos[idx_train, ]
c50_para_plot <- C50::C5.0(x = datos_train[, names(datos_train) != "diseased"],
                           y = datos_train$diseased,
                           control = C50::C5.0Control(minCases = 8))
plot(c50_para_plot, type = "s", main = "C5.0 (mismo modelo que en mlr3, minCases = 8)")
print(learner_rpart$model$variable.importance)    # importancia de variables rpart
print(C50::C5imp(learner_c50$model, metric = "usage"))  # importancia C5.0


#Modelo con todos los datos

tarea_full <- as_task_classif(datos, target = "diseased")
learner_rpart_full <- lrn("classif.rpart", maxdepth = 3,  predict_type = "response")
learner_c50_full   <- lrn("classif.C50",   minCases = 8,  predict_type = "response")
learner_rpart_full$train(tarea_full)
learner_c50_full$train(tarea_full)

rpart.plot(learner_rpart_full$model, main = "Modelo Final rpart (Datos completos)")

c50_full_plot <- C50::C5.0(
  x = datos[, names(datos) != "diseased"],
  y = datos$diseased,
  control = C50::C5.0Control(minCases = 8)
)
plot(c50_full_plot, type = "s", main = "Modelo Final C5.0 (Datos completos)")



