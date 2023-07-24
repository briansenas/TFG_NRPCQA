[:es: Español](#spanish) | [:gb: English](#english)
---
<h1 align="center"> Estimación de calidad de imágenes médicas 3D por medio de aprendizaje automático </h1>
<h4 align="center">  Curso 2022-2023, Grado en Ingeniería Informática, ETSIIT UGR.</h4>
<h5 align="center"> Trabajo de Fin de Grado </h5><a id='spanish'></a>

<img style="display:block;width:100%;margin:auto;padding-bottom:25px" src="https://github.com/CodeBoy-source/TFG_NRPCQA/blob/main/imgs/local_rotation2.png?raw=True"/>

<div style="display:flex"> 
<div style="flex:50%;max-width:30%"><font size="4"> <emph><strong>Autor</strong></emph>: Brian Sena Simons</font> </div>
<div style="flex:50%;text-align:right"><font size="4"> <emph><strong>Directores</emph></strong>: Dr. Pablo Mesejo Santiago y Dr. Enrique Bermejo Nievas </font></div>
</div>


### :pushpin: Introducción. 
Este TFG trata de la implementación de un meta-modelo capaz de estimar la calidad de 
representaciones 3D biomédicas por medio del uso de un dataset generado 
sintéticamente. Obtiene un resultado promedio de 86\% de <emph>SROCC</emph>, con 
desviación del 0.11 y mediana 88\%.

Para una lectura detallada de los modelos y experimentación pulsar [aquí](https://github.com/CodeBoy-source/TFG_NRPCQA/blob/main/Document/proyecto.tex).


### :gear: Instalación de paquetes.
Se puede importar los directorios en Colab y ejecutar con comandos 
de terminal, ya que todos los paquetes necesarios vienen instalados 
por defecto. No obstante, se ha aportado un `requirements.txt` para instalar 
los paquetes necesario por medio de `python3 -m pip install -r requirements.txt`. 

Otra posibilidad es hacer uso del notebook jupyter con la ejecución del modelo 
por medio de `FastAI` en vez de la implementación con PyTorch. 

### :file_cabinet: Datos
Para los experimentos se utilizaron los siguientes _datasets_:
    
- SJTU [[2]](#2)
    - 10 nubes de puntos de referencia.
    - 7 tipos de distorsiones: Compresión octree (OT), Ruido fotométrico (CN), Submuestreo uniforme (DS), DS + CN, DS + GGN, Ruido geométrico gaussiano GGN, CN + GGN.
    - 6 niveles de distorsiones distintos.
    - Total de 420 ejemplos.
- WPC [[3-4]](#3)
    - 20 nubes de puntos de referencia. 
    - 5 tipos de distorsiones, Submuestreo, Ruido Gaussiano, G-PCC/S-PCC, V-PCC y G-PCC/L-PCC.
    - Total de 740 nubes de puntos. 
- LS-SJTU-PCQA [[5]](#5)
    - 104 nubes de puntos de referencia. 
    - 37 distorsiones diferentes, de las cuales usamos 5: Submuestreo, movimiento 
    y rotación local, ruido geométrico gaussiano y compresión octree. 
    - 7 niveles de distorsiones. 
    - Total de 3640 ejemplos.
- Datos médicos Sintéticos 
    - 11 nubes de puntos de referencia 
    - 5 distorsiones: Submuestreo, movimiento y rotación local, ruido geométrico gaussiano y compresión octree.
    - 7 niveles de intensidad.
    - Total de 385 ejemplos. 

## Entrenamiento

Para ejecutar el modelo es necesario elegir entre la implementación en Pytorch o FastAI. 
A continuación se observan algunos ejemplos de ejecuciones. El código está dependerá del modelo elegido. 

### :nut_and_bolt: Modelo ML: NR3DQA [[6]](#6)

#### Ejemplo de extracción de características con PyntCloud. 
Extracción de características de publicación original y características nuevas propuestas.
```
python -O demo.py --input-dir PCQA-Databases/LS-SJTU-PCQA/ --output-dir ./features/LS-SJTU-PCQA.csv
```
#### Ejemplo de inferencia  
```
python -O nr3dqa.py --input-dir features.csv --splits-dir mos.csv --num_splits 11 --drop-colors 
```
### :building_construction: Modelo DL: VQA-PC [[7]](#7)

#### Ejemplo de comando de test sobre SJTU
```
python -u test.py  \
--pretrained_model_path ''../train/ckpts/ResNet_mean_with_fast_LSPCQA_1_best.pth'' \
--path_imgs ''../train/database/sjtu_2d/'' \
--path_3d_features ''../train/database/sjtu_slowfast''  \
--data_info  ''data_info/sjtu_mos.csv'' \
--num_workers 2 \
--output_csv_path ''sjtu_prediction.csv'
```

#### Ejemplo de comando de entrenamiento sobre SJTU
```
python -u train.py \
 --database SJTU \
 --model_name  Pretrained_mean_with_fast \
 --pretrained_model_path ''ckpts/ResNet_mean_with_fast_LSPCQA_1_best.pth'' \
 --conv_base_lr 0.00004 \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --train_batch_size 32 \
 --num_workers 2 \
 --epochs 2 \
 --split_num 2 \
 --feature_fusion_method 0 \
 --ckpt_path ckpts \
```
Se puede añadir el flag `--Leslie` si desea utilizar el criterio de super convergencia de Leslie Smith[[1]](#1).

#### Ejemplo de ejecución de entrenamiento con FastAI 
```
config = dotdict(config)
dls = get_dataloader(config, 0)
srocc = SpearmanCorrCoef()
learn = Learner(dls, model, metrics=[mse, mae,srocc],
                loss_func = MSELossFlat(),
                opt_func=ranger)
lr = learn.lr_find(show_plot=False)
learn.fit_one_cycle(30, lr_max=lr[0], cbs=cbs)
```

[:es: Español](#spanish) | [:gb: English](#english)
---
<h1 align="center"> Quality Assessment of 3D medical images through machine learning </h1>
<h4 align="center">  2022-2023 Course, Computer Science Degree, ETSIIT UGR.</h4>
<h5 align="center"> Bachelor's thesis </h5><a id='english'></a>

<img style="display:block;width:100%;margin:auto;padding-bottom:25px" src="https://github.com/CodeBoy-source/TFG_NRPCQA/blob/main/imgs/local_rotation2.png?raw=True"/>

<div style="display:flex"> 
<div style="flex:50%;max-width:30%"><font size="4"> <emph><strong>Autor</strong></emph>: Brian Sena Simons</font> </div>
<div style="flex:50%;text-align:right"><font size="4"> <emph><strong>Directores</emph></strong>: Dr. Pablo Mesejo Santiago y Dr. Enrique Bermejo Nievas </font></div>
</div>


### :pushpin: Introduction. 
This bachelor's thesis is about the implementation of a meta-model capable of 
estimating the quality of biomedical 3D representations using a synthetically 
generated dataset. It achieves an average result of 86% <emph>SROCC</emph>, 
with a deviation of 0.11 and a median of 88%.

For a detailed reading of the models and experimentation, click [here](https://github.com/CodeBoy-source/TFG_NRPCQA/blob/main/Document/proyecto.tex).


### :gear: Requirements.
You can import the directories in Colab and execute them with terminal commands 
since all the necessary packages come installed by default. 
However, a `requirements.txt` file has been provided to install the necessary 
packages using `python3 -m pip install -r requirements.txt`.

Another option is to use the Jupyter notebook with the model execution through `FastAI`instead of the implementation with PyTorch.
### :file_cabinet: Datasets
For the experimentation the following datasets have been used: 
    
- SJTU [[2]](#2)
    - 10 reference point clouds.
    - 7 types of distortions: Octree compression(OT), Color noise (CN), Downsampling (DS), DS + CN, DS + GGN, Geometric gaussian noise GGN, CN + GGN.
    - 6 levels of intensity.
    - Total of 420 examples.
- WPC [[3-4]](#3)
    - 20 reference point clouds. 
    - 5 types of distortions: Downsampling, Gassian noise, G-PCC/S-PCC, V-PCC y G-PCC/L-PCC.
    - Total of 740 examples. 
- LS-SJTU-PCQA [[5]](#5)
    - 104 reference poitn clouds. 
    - 37 different distortions, from which we chose: Downsampling, Local rotation 
    and offset, Gaussian geometric shift and octree compression. 
    - 7 levels of intensity. 
    - Total of 3640 examples.
- Medical Synthetic data 
    - 11 reference point clouds.
    - 5 distortions: Downsampling, Local offset and rotation, Gaussian geometric shift and Octree compression.
    - 7 levels of intensity.
    - Total of 385 examples. 

## Training

To run the model, it is necessary to choose between the implementation in PyTorch or FastAI. 
Below are some examples of executions. The code will depend on the chosen model.

### :nut_and_bolt: Modelo ML: NR3DQA [[6]](#6)

#### PyntCloud feature extraction example. 
Extraction of features from the original publication and newly proposed features.
```
python -O demo.py --input-dir PCQA-Databases/LS-SJTU-PCQA/ --output-dir ./features/LS-SJTU-PCQA.csv
```
#### Inference example
```
python -O nr3dqa.py --input-dir features.csv --splits-dir mos.csv --num_splits 11 --drop-colors 
```
### :building_construction: Modelo DL: VQA-PC [[7]](#7)

#### Testing on SJTU example
```
python -u test.py  \
--pretrained_model_path ''../train/ckpts/ResNet_mean_with_fast_LSPCQA_1_best.pth'' \
--path_imgs ''../train/database/sjtu_2d/'' \
--path_3d_features ''../train/database/sjtu_slowfast''  \
--data_info  ''data_info/sjtu_mos.csv'' \
--num_workers 2 \
--output_csv_path ''sjtu_prediction.csv'
```

#### Training on SJTU example
```
python -u train.py \
 --database SJTU \
 --model_name  Pretrained_mean_with_fast \
 --pretrained_model_path ''ckpts/ResNet_mean_with_fast_LSPCQA_1_best.pth'' \
 --conv_base_lr 0.00004 \
 --decay_ratio 0.9 \
 --decay_interval 10 \
 --train_batch_size 32 \
 --num_workers 2 \
 --epochs 2 \
 --split_num 2 \
 --feature_fusion_method 0 \
 --ckpt_path ckpts \
```
The flag `--Leslie` can be added if it's desired to use Leslie Smith[[1]](#1) super convergence.

#### FastAI execution example
```
config = dotdict(config)
dls = get_dataloader(config, 0)
srocc = SpearmanCorrCoef()
learn = Learner(dls, model, metrics=[mse, mae,srocc],
                loss_func = MSELossFlat(),
                opt_func=ranger)
lr = learn.lr_find(show_plot=False)
learn.fit_one_cycle(30, lr_max=lr[0], cbs=cbs)
```


## References
<a id="1">[1]</a> 
Leslie N. Smith, Nicholay Topin.
Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates, 2018. \
<a id="2">[2]</a> 
Q. Yang, H. Chen, Z. Ma, Y. Xu, R. Tang y J. Sun, "Predicting
the Perceptual Quality of Point Cloud: A 3D-to-2D Projection-Based
Exploration" IEEE Transactions on Multimedia, 2020.\
<a id="3">[3]</a> 
Q. Liu, H. Su, Z. Duanmu, W. Liu y Z. Wang, "Perceptual Quality Assessment of Colored
3D Point Clouds," IEEE Transactions on
Visualization and Computer Graphics, págs. 1-1, 2022. \
<a id="4">[4]</a> 
H. Su, Z. Duanmu, W. Liu, Q. Liu y Z. Wang, "Perceptual quality
assessment of 3D point clouds" en 2019 IEEE International Confe-
rence on Image Processing (ICIP), 2019, págs. 3182-3186.\
<a id="5">[5]</a> 
Y. Liu, Q. Yang, Y. Xu y L. Yang, "Point Cloud Quality Assessment:
Dataset Construction and Learning-based No-Reference Metric" \
<a id="6">[6]</a> 
Z. Zhang, W. Sun, X. Min, T. Wang, W. Lu y G. Zhai, ((No-Reference
Quality Assessment for 3D Colored Point Cloud and Mesh Models,))
IEEE Transactions on Circuits and Systems for Video Technology,
vol. 32, n.o 11, pags. 7618-7631, 2022.\
<a id="7">[7]</a> 
Z. Zhang et al., Treating Point Cloud as Moving Camera Videos: A
No-Reference Quality Assessment Metric, 2022.
