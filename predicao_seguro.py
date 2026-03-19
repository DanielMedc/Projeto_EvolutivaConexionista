
'/content/TravelInsurancePrediction.csv'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def aprendizado(coef,treino,epocas,w,dia):
  np.random.seed(42)
  df = pd.read_csv('/content/TravelInsurancePrediction.csv')
  df = df.sample(frac = 1).reset_index(drop=True)
  X_data = df.drop(['Unnamed: 0','TravelInsurance'], axis=1)

  def activation(x):
      return 1 / (1 + np.exp(-x))
  # --- Parametros---
  COEF_APREND = coef
  coef_inicial= COEF_APREND # valor referencial fixo decay, ainda exponecial mas suavizado
  N_TREINO = treino
  EPOCAS =  epocas
  NEURONIOS = 1
  #---Hiperparametro--
  W = w # peso do erro ao errar classe 1, aumentando recall (melhor oferecer seguro a mais)
  dia= dia # decay a cada __ epocas
  # ------ Normalização das features ------
  Seguro = df['TravelInsurance']
  Y_data = Seguro

  X_data['Employment Type'] = X_data['Employment Type'].map({'Government Sector': 0, 'Private Sector/Self Employed': 1})
  X_data['GraduateOrNot'] = X_data['GraduateOrNot'].map({'No': 0, 'Yes': 1})
  X_data['EverTravelledAbroad'] = X_data['EverTravelledAbroad'].map({'No': 0, 'Yes': 1})
  X_data['FrequentFlyer'] = X_data['FrequentFlyer'].map({'No': 0, 'Yes': 1})

  maior_renda = X_data['AnnualIncome'].max()
  X_data['AnnualIncome'] = X_data['AnnualIncome'] / maior_renda

  maior_idade = X_data['Age'].max()
  X_data['Age'] = X_data['Age'] / maior_idade

  membro_familia = X_data['FamilyMembers'].max()
  X_data['FamilyMembers'] = X_data['FamilyMembers'] / membro_familia

  #features + bias
  NN = (np.random.rand((np.size(X_data,axis=1) + 1), NEURONIOS) - 0.5)

  X_data = np.array(X_data)

  X_data = np.append(np.ones((len(X_data), 1)), X_data, axis=1)
  Y_data = np.array(Y_data)

  X_train = X_data[:N_TREINO, :]
  Y_train = Y_data[:N_TREINO]

  X_test  = X_data[N_TREINO:, :]
  Y_test  = Y_data[N_TREINO:]
  MSE = []
  ACCURACY = []
  for epoch in range(EPOCAS):
      for i in range(len(X_train)):
          input_line = X_train[i, :]
          #np.dot = multiplicação
          output = activation(np.dot(input_line, NN))
          peso = W if Y_train[i] == 1 else 1.0
          error = (output - Y_train[i]) * peso
          gradiente = error * output *(1-output)
          NN -= COEF_APREND * 2* gradiente * input_line.reshape(-1,1)
      # a cada "dia" época, coeficiente de aprendizado cai
      if (epoch % dia == 0):
          COEF_APREND = coef_inicial * np.exp(-0.001*epoch)
  # -------------------------------METRICAS ---------------------
      test_outputs = activation(np.dot(X_test, NN))
      a_MSE = np.mean(np.square(test_outputs - Y_test.reshape(-1, 1)))
      MSE.append(a_MSE)
      #lógico binário-> int
      previsoes = (test_outputs >= 0.5).astype(int).flatten()
      acuracia = np.mean(previsoes == Y_test)
      ACCURACY.append(acuracia)

  CM = np.zeros([2,2])
  for instancia in range(len(X_test)):
      input_instancia = X_test[instancia,:]
      output = activation(np.dot(input_instancia, NN))
      previsao = 0
      if (output >= 0.5):
          previsao = 1
      CM[Y_test[instancia],previsao] += 1

  acuracia_final = np.trace(CM) / np.sum(CM)
  recall   = CM[1,1] / (CM[1,1] + CM[1,0])
  precisao   = CM[1,1] / (CM[1,1] + CM[0,1])
  f1 = 2*(precisao*recall)/(precisao+recall)
  print(f"Acurácia Final: {acuracia_final * 100:.2f}%")
  print(f"Recall: {recall * 100:.2f}%")
  print(f"Precisao : {precisao * 100:.2f}%")
  print(f"F1 : {f1 * 100:.2f}%")
  print(f"coeficiente: {coef:.5f}")
  fig, axs = plt.subplots(1, 3, figsize=(18, 5))

  # Matriz de Confusão
  cax = axs[0].matshow(CM, cmap='Blues')
  axs[0].set_title("Matriz de Confusão")
  for i in range(2):
      for j in range(2):
          axs[0].text(j, i, str(CM[i,j]), va='center', ha='center')

  # Acurácia
  axs[1].plot(range(EPOCAS), ACCURACY, color='green')
  axs[1].set_title('Acurácia por Época')
  axs[1].set_ylim(0, 1)

  # MSE
  axs[2].plot(range(EPOCAS), MSE, color='red')
  axs[2].set_title('MSE por Época')

  plt.tight_layout()
  plt.show()

aprendizado(0.23,1500,350,2.5,30)

#Estudo de coeficiente, (+-10 minutos )
'''for i in range(50):
  aprendizado(0.60-(i*0.01),1500,350,2.5,30)'''
