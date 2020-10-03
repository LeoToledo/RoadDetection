#Primeiramente, vamos importar as bibliotecas e o modelo
import torch
import torch.nn as nn
import torchvision
import numpy as np
from matplotlib import pyplot as plt
import model
import load_data as ld
import os
from tqdm import tqdm

#######Variáveis
#Tamanho do lote de imagens a serem carregadas - para validação utilizamos 1
BATCH_SIZE = 1
print_results = 20
total_iou = []
total_recall = []
total_precision = []


#Creating the dataset
dataset = ld.Roads()
training_set, test_set = dataset.load_data()

#Spliting features and labels. Creating DataLoaders
test_X = test_set[0]
test_y = test_set[1]
dataloader_test_X = torch.utils.data.DataLoader(test_X,
                                                 batch_size=BATCH_SIZE)
dataloader_test_y = torch.utils.data.DataLoader(test_y,
                                                 batch_size=BATCH_SIZE)


#########Carregando o modelo treinado
#Path do modelo a ser carregado
path_checkpoint_load = '/home/kodex/Área de Trabalho/Semear/RoadDetection - SqueezeNet/Tentativas/02-10/model1_2020-10-02.tar'

#Checando se a GPU está disponível
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
print(torch.cuda.get_device_name(device))

#Creating the model
model = model.SqueezeNet().to(device)

#Checando se o arquivo existe
if os.path.isfile(path_checkpoint_load):
    print("\nCarregando checkpoint de " + path_checkpoint_load)
    #Carregando informações salvas do modelo
    checkpoint_info = torch.load(path_checkpoint_load)
    epoch_ini = checkpoint_info['epoch']
    loss = checkpoint_info['loss']
    #Carregando modelo
    model.load_state_dict(checkpoint_info['Model_state'])
    
else:
    print("Arquivo não encontrado !")
    
    
#######Começando a validação
#Passando modelo para modo de validação
#É importante para sinalizar para camadas de dropout e batchnorm congelarem
model.eval()

#Carregando os iteradores
iter_X = iter(dataloader_test_X)
iter_y = iter(dataloader_test_y)

with torch.no_grad():
    
    for i in tqdm(range( 0, len(test_X), BATCH_SIZE )):
        
        #Carregando o próximo batch
        batch_X, _ = iter_X.next()
        batch_y, _ = iter_y.next()
        
        #Passando os dados para GPU
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
  
        #Forward prop
        outputs = model(batch_X)
        
        if i % print_results == 0:
            #Plotando resultados 
            #Para plotar no matplotlib, devemos reodenar os canais
            plot_output = outputs[0].permute(1, 2, 0).cpu().detach().numpy()
            plot_batchy = batch_y[0].permute(1, 2, 0).cpu().detach().numpy()
            
            plt.imshow(plot_output, cmap='gray')
            plt.title("Imagem Gerada")
            path = "/home/kodex/Área de Trabalho/Semear/RoadDetection - SqueezeNet/test_imgs/" + str(i) +"_gerada"
            plt.savefig(path)
            plt.close()
            
            plt.imshow(plot_batchy, cmap='gray') 
            plt.title("Imagem Real")
            path = "/home/kodex/Área de Trabalho/Semear/RoadDetection - SqueezeNet/test_imgs/" + str(i) + "_real"
            plt.savefig(path)
            plt.close()
    
    

        ####AVALIAÇÃO DO MODELO
        '''
        Métricas para a avaliação:
          - Verdadeiro Positivo: são os pixels brancos presentes na área de interseção (pixels brancos gerados que também estão presentes na máscara verdadeira)
          - Falso Positivo: seria a área de pixels brancos presentes na máscara gerada, porém não presentes na máscara verdadeira, ou seja, falsos pixels brancos
          - Falso Negativo: seria a área de pixels brancos presentes na máscara verdadeira, porém não presente na gerada, são falsos pixels pretos.
        
        Cálculo da Precisão:
            Verdadeiro_Positivo/(Verdadeiro_Positivo+Falso_Positivo)
            
        Cálculo do Recall:
            Verdadeiro_Positivo/(Verdadeiro_Positivo+Falso_Negativo)
            
        IOU:
            Área_de_Intersecção/Área_de_União
        '''
        
        #Vamos gerar primeiro nossa imagem com a interseção de pixels brancos
        #Para isso, como as imagens são compostas de 0s e 1s, só precisamos multiplicá-las
        intersecao = plot_output * plot_batchy
        
        True_Area = int(np.sum(plot_batchy))
        Gen_Area = int(np.sum(plot_output))
        Inter_Area = int(np.sum(intersecao))
        
        
        FP_image = intersecao - plot_output
        FP_image = torch.from_numpy(FP_image)
        FP_image = np.abs(FP_image)
        FP_image = (FP_image > 0.5).int()  
        FP = int(torch.sum(FP_image))
        
        FN_image = intersecao - plot_batchy
        FN_image = torch.from_numpy(FN_image)
        FN_image = np.abs(FN_image)
        FN_image = (FN_image > 0.5).int()
        FN = int(torch.sum(FN_image))
        
        
        IoU = Inter_Area/(Gen_Area+True_Area-Inter_Area)       
        Precision = Inter_Area/(Inter_Area + FP)
        if(FN == 0):
            FN = 1
        Recall = Inter_Area/(Inter_Area + FN)
    
        #Caso alguma imagem saia bugada, pula ela no cálculo 
        if(IoU != 0 and Precision != 0 and Recall != 0): 
            total_iou.append(IoU)
            total_precision.append(Precision)
            total_recall.append(Recall)
        
            if i % print_results == 0:
                print("\nMétricas:\n\nIoU: {}\nPrecision: {}\nRecall: {}".format(IoU, Precision, Recall))
   
iou_med = np.sum(total_iou)/len(total_iou)
prec_med = np.sum(total_precision)/len(total_precision)
rec_med = np.sum(total_recall)/len(total_recall)
print("\n\n*********************************************************************************\n")
print("Métricas:\n\nIoU: {}\nPrecision: {}\nRecall: {}".format(iou_med, prec_med, rec_med))
print("\n*********************************************************************************\n\n\n")   
    