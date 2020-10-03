# RoadDetection
Implementação de algoritmo para detecção de pista utilizando-se a rede SqueezeNet adaptada para uma tarefa de segmentação. 
SqueezeNet Paper: https://arxiv.org/abs/1602.07360

# Etapa de treino
Abaixo, são apresentadas algumas imagens dos resultados obtidos durante o treinamento

Loss Over Epochs:

<p float="left">
  <img src="https://github.com/LeoToledo/RoadDetection/blob/main/imgs/Loss1.png" width="500" />
</p>


Predicted Image vs Real Image:
<p float="left">
  <img src="https://github.com/LeoToledo/RoadDetection/blob/main/imgs/ezgif-6-6f5d9df26f2f.gif" width="500" />
  <img src="https://github.com/LeoToledo/RoadDetection/blob/main/imgs/75_real.png" width="500" /> 
</p>

Predicted/Real:

<p float="center">
  <img src="https://github.com/LeoToledo/RoadDetection/blob/main/imgs/individualImage%20(3).png" width="500" />
</p>

# Etapa de Teste
Para a análise dos resultados, foram analisadas 3 métricas: IoU, Recall e Precision. Abaixo, seguem os valores obtidos para as 3 métricas e algumas imagens.

IoU, Recall e Precision:
<p float="left">
  <img src="https://github.com/LeoToledo/RoadDetection/blob/main/imgs/iou" width="500" />
</p>

Predicted Image vs Real Image:
<p float="left">
  <img src="https://github.com/LeoToledo/RoadDetection/blob/main/imgs/0_gerada.png" width="500" />
  <img src="https://github.com/LeoToledo/RoadDetection/blob/main/imgs/0_real.png" width="500" /> 
</p>
