# CFAESVD:Predicting Microbe-Diseases Associations Based on Multi-Kernel Autoencoder and Singular Value Decomposition
Extensive research has shown that microbial communities are closely linked to the development of many complex human diseases. Therefore, identifying potential microbe-disease associations is essential for improving diagnosis, prognosis, and treatment strategies. Traditional biomedical experiments,however, tend to be costly, time-consuming, and labor-intensive. To address these limitations, we propose a novel computational model, CFAESVD, for predicting potential microbe-diseases associations.The model begins by integrating data from three different databases and employs a Centered Kernel Alignment-based Multi-Kernel Learning (CKA-MKL) algorithm to fuse multi-source features of microbes and diseases based on known associations. It then utilizes three distinct modules to extract feature representations: a four-layer autoencoder to capture the nonlinear features of diseases,Singular Value Decomposition (SVD) module to extract linear features for both microbes and diseases, and Graph Attention Encoder (GATE) module for the nonlinear features of microbes. These features are concatenated into feature vectors representing microbe-disease pairs. Finally, these vectors are input into an modifyed Cascade Forest model for prediction. Experimental results show that under five-fold cross-validation, the model achieves an AUC of 0.9748 and an AUPR of 0.9732, surpassing several advanced prediction models. Additionally, case studies confirm the CFAESVD model’s effectiveness in predicting microbial associations with conditions such as obesity and Crohn’s disease, further validating its reliability.CFAESVD is publicly available at https://github.com/senliyang/CFAESVD.
# Flowchart
![image](https://github.com/senliyang/CFAESVD/blob/main/CFAESVD/CFAESVD终.png)
# Requirements
numpy                     1.19.2          
pandas                    1.1.5           
python                    3.9               
scipy                     1.5.2            
tensorflow                2.6.2  
# Usage
Calculate the integrated similarity between microbes and diseases  　&ensp;                  python get_weight_MKL_CKA.py          
Extract the linear features of microbes and diseases             　&ensp;        python matrix_svd.py                 
Run this model          　&ensp;      python main.py
