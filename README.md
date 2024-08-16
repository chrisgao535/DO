# Remote sensing inversion of dissolved oxygen concentration in Baiyang Lake based on machine learning<br>

This is the research that corresponds the manuscript authored by Leilei Shi, Chen Gao, Tuo Wang, Lixiang Liu, Yue Wu, and Xiaogang You, submitted to Ecological Informatics and waiting for review. The study focused on the dissolved oxygen (DO) of Baiyangdian Lake using 251 sets of empirically measured water quality data and corresponding Sentinel-2 satellite images. Nine machine learning algorithms were then used  to develop a rapid detection algorithm for the spatial distribution of the DO concentration in Baiyangdian Lake.<br>

Seven ground stations: Gudingdian (GDD), Nanliuzhuang (NLZ), Shaohedian (SCD), Guangdianzhangzhuang (GDZZ), Quantou (QT), Caiputai (CPT), and Zaolinzhuang (ZLZ).<br>

Nine machine learning models: support vector machines (SVM), artificial neural networks (ANN), Bayesian ridge regression (BRR), decision tree regression (DTR), K-nearest-neighbor regression (KNR), random forest regression (RFR), extra tree regression (ETR), AdaBoost regression (ABR), and gradient boosting regression (GBR).<br>

## What you can find<br>
* `Research data`: dissolved oxygen (DO) concentration data, latitude and longitude data, and pixel data of Sentinel-2 MSI images. The B01, B02 to B12 of tables represents bands 1, 2 to 12 of Sentinel-2 MSI image, respectively.
* `code`: Python scripts for training nine machine learning models.