import load_data as ld
import model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import date
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision

#Variables
BATCH_SIZE = 5
EPOCHS = 250
epoch_checkpoint = 1 #Define de quantas em quantas épocas será feito um checkpoint
model_version = 1.0 #Define a versão atual do modelo
loss_name = "BCELoss" #Loss function utilizada
name_checkpoint = "model1_" + str(date.today()) #Nome do checkpoint
path_model_checkpoints = '/home/kodex/Área de Trabalho/Semear/RoadDetection - SqueezeNet/Checkpoints/' + name_checkpoint + '.tar' #Local de salvamento dos checkpoints
print_results = 300  #Define o período de prints da loss
LEARNING_RATE = 0.0005

#Criando configurações do tensorboard para visualização dos dados
writer_Tensorboard = SummaryWriter('/home/kodex/Área de Trabalho/Semear/RoadDetection - SqueezeNet/runs')

#Creating the dataset
dataset = ld.Roads()
training_set, test_set = dataset.load_data()


#Habilitando GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Rodando na GPU")
else:
    device = torch.device("cpu")
    print("Rodando na CPU")
    
    
#Creating the model
model = model.SqueezeNet().to(device)
# print(model)

#Função utilizada para realizar a inicialização dos pesos da camadas
#convolucionais seguindo distribuição normal com média 0
def _initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean=0, std=0.02)                       
model.apply(_initialize_weights) 
model.cuda()
summary(model, (3, 512, 512))

#Setting the optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.BCELoss()

#Spliting features and labels. Creating DataLoaders
train_X = training_set[0]
train_y = training_set[1]
dataloader_train_X = torch.utils.data.DataLoader(train_X,
                                                 batch_size=BATCH_SIZE)
dataloader_train_y = torch.utils.data.DataLoader(train_y,
                                                 batch_size=BATCH_SIZE)
    

#Training the model
for epoch in range(EPOCHS):
    #Loading dataset iterators
    iter_X = iter(dataloader_train_X)
    iter_y = iter(dataloader_train_y)

    for i in tqdm(range( 0, len(train_X), BATCH_SIZE )):
        
        batch_X, _ = iter_X.next()
        batch_y, _ = iter_y.next()
        
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        #Cleaning the gradient so the loss doesnt accumulate
        model.zero_grad()
        
        #Forward prop
        outputs = model(batch_X)
        # print("OUTPUT: ", outputs.shape)
        # print("REAL: ", batch_y.shape)
        
        #Calculate Cost
        loss = loss_function(outputs, batch_y)
        loss.backward()
        
        #Update weights
        optimizer.step()
        
        if i % print_results == 0:
            #Printando perda e informações
            print("Train Epoch: ", epoch, "   LOSS: ", loss)
            
            #Plotando perda no tensorboard
            writer_Tensorboard.add_scalar('Loss/train',loss.item(),(len(train_X)*epoch)+(i*BATCH_SIZE))
            
            #Selecionando imagens para plotar no tensorboard
            batch_x_tb = batch_X[0]
            batch_y_tb = batch_y[0][0]
            output_tb = outputs[0][0]
            
            img_tb = torch.cat((batch_y_tb, output_tb),1)
            img_grid_mask = torchvision.utils.make_grid(img_tb)
            img_grid_real = torchvision.utils.make_grid(batch_x_tb)
            
            writer_Tensorboard.add_image('Ground Truth vs Output',img_grid_mask,(len(train_X)*epoch)+(i*BATCH_SIZE))
            writer_Tensorboard.add_image('Original Image',img_grid_real,(len(train_X)*epoch)+(i*BATCH_SIZE))
            
            
    #Configurando Checkpoints
    if epoch % epoch_checkpoint == 0:
        torch.save({
            'epoch': epoch,
            'Model_state': model.state_dict(),
            'Model_version': model_version,
            'optimizer': optimizer.state_dict(),
            'Loss Function': loss_name,
            'loss':loss
        }, path_model_checkpoints)
            
            
    #Plotando resultados parciais
    #Para plotar no matplotlib, devemos reodenar os canais
    plot_output = outputs[0].permute(1, 2, 0).cpu().detach().numpy()
    plot_batchy = batch_y[0].permute(1, 2, 0).cpu().detach().numpy()
    
    plt.imshow(plot_output, cmap='gray')
    plt.title("Imagem Gerada")
    path = "/home/kodex/Área de Trabalho/Semear/RoadDetection - SqueezeNet/imgs/" + str(epoch) +"_gerada"
    plt.savefig(path)
    plt.close()
    
    plt.imshow(plot_batchy, cmap='gray') 
    plt.title("Imagem Real")
    path = "/home/kodex/Área de Trabalho/Semear/RoadDetection - SqueezeNet/imgs/" + str(epoch) + "_real"
    plt.savefig(path)
    plt.close()
    