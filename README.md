# Avocado_Research
Estimation of Avocado Ripening Day By HyperSpectral Imaging
# Introduction
## Main Steps
The Project contains 3 main steps:
1. Data Pre-processing
2. Channels selection models & methods
3. Analyzing the results

## Notes
Before getting started and in order to run all steps, follow those steps:
1. clone repository  
2. Arrange folders so that this will be the tree in your main folder (important)
```bash
├───analyze
│   ├───plantations
│   │   ├───train12_test3
│   │   ├───train13_test2
│   │   └───train23_test1
│   └───random
├───best_models
├───best_models_plantations
│   ├───train12_test3
│   ├───train13_test2
│   └───train23_test1
├───channels
│   ├───add_one
│   ├───drop_more
│   ├───drop_one
│   └───random_select
├───channels_plantations
│   ├───train12_test3
│   │   ├───add_one
│   │   ├───drop_more
│   │   ├───drop_one
│   │   └───random_select
│   ├───train13_test2
│   │   ├───add_one
│   │   ├───drop_more
│   │   ├───drop_one
│   │   └───random_select
│   └───train23_test1
│       ├───add_one
│       ├───drop_more
│       ├───drop_one
│       └───random_select
├───graphs
│   ├───best_channel_number
│   └───total_best_wl
├───graphs_plantations
│   ├───train12_test3
│   │   ├───best_channel_number
│   │   └───total_best_wl
│   ├───train13_test2
│   │   ├───best_channel_number
│   │   └───total_best_wl
│   └───train23_test1
│       ├───best_channel_number
│       └───total_best_wl
├───ref
```

* Note 1 - when working on hpc, uplaod the same folder tree. add the data folder by the name `data vnir`
* Note 2 - sub folders in the data folder should countain a `capture` folder with the RAW file of the image and HDR file
* Note 3 -  the sub folders name should resamble the format: "vnir_avocado_avocado_1-4_2019-12-03_10-53" (where 1-4 represnets samples 1 to 4, and 2019-12-03_10-53 is the date of the image). If your subfolders are not named in this format, modify the line `self.samples = (re.search(r'avocado_avocado_(.*?)_2019', img_fold).group(1)).split('-')` in `data_loader.py` so that self.samples will contain an array of the sample numbers in the image.

3.Change those parameters in `params_update.py`:
| Parameter | Description |
| --- | --- |
| hpc | change to True if your working on hpc |
| data_folder | path to your data folder- where saved sub-folders for every hyperspectral image |

# Data Pre-Processing
## Segmentation
The segmentation process is done by subtracting 2 images in 2 different wavelengths by the NDVI index. then extracting contours from the image and selecting the largest ones to avoid outliers.

Run `segmentation.py` - This function saves dataframe for every image as a feather file in the `segmentation` folder. The dataframe contains:

| Column | Description |
| --- | --- |
| avocado_num | avocado sample numbers in the image |
| bounding_boxes | bounding boxes equivilent for every sample number |
| ndvi_img | ndvi img (flatten array) |
| countours | countours equivilent for every sample number (flatten array) |
| cont_shapes | shapes of the original countours array |
| centers | center pixel equivilent for every bounding box |
| avocodo_pixs | pixels in the image classified as avocado (not background)|

Example for the segmantation:
<p align="center">

<img src="https://github.com/adico103/Avocado_Research/blob/main/segmentation_101-104.png" width=30%>
</p>


## Feature Extraction
Run `feature_extraction.py` - The features that are calculated for each avocado sample and saves dataframe for every avocado sample as a feather file in `ref` folder :

| Column | Description |
| --- | --- |
| bands | list of the bands the features were calculated for |
| ref | Reflectance for the whole avocado for each wavelength (normalized by one of the wavelengths) |
| partial_ref | Partial Reflectance of 3 parts of the avocado -  up,middle,down, for each wavelength (normalized by one of the wavelengths) |
| size_ratio | The ratio between the hight and the width of the elipse fitted to the avocado |

* If you wish to change/add features saved, do so in `feature_extraction` function in `data_loader.py` and **Notice to do so also in** `get_data_by_bands` **function that is in** `cahnnel_test.py` **to load those features to the model**


# Channels selection models & methods
## Theoretical Introduction
This section is evaluating 4 different methods for band selection, tested on 3 different models (Random Forest, Xgboost and SVM).
The models are receiving input data that includes the features calculated, **for a specific set of wavelengths**, determined by each method.

The 4 methods are:
| Method | Description |
| --- | --- |
| drop_one | start with 84 equaly spaced wavelengths (out of total 840). Drop one of the bands at each time, and evaluate the model result with the remaining bands. save the best 83 wavelengths and so forth until reaching only 2 bands. |
| drop_more | start with 84 equaly spaced wavelengths (out of total 840). Drop N number of the bands at each time, and evaluate the model result with the remaining bands. save the best (84-N) wavelengths and so forth until reaching only 2 bands. |
| add_one | select 2 wavelengths from 84 equaly spaced wavelengths (that provided the best result for the model out of all possible combinations). add one wavelength at a time (that gives the best results), until reaching 84 bands. |
| random_select | for N changing between 2 to 84: 100 times randomly select N bands, and pick the best combination by the model results  |

The data is splitted in 4 different ways to train and test sets:

| Data Split type | Description |
| --- | --- |
| Random     | randomly split avocado samples in to train and test sets |
| Plantation -1 | 1. test set include : 1-40 avocado samples from plantation1 |
|  Plantation -2  | 2. test set include : 41-80 avocado samples from plantation2  |
|  Plantation -3 | 3. test set include : 81-120 avocado samples from plantation3  |



## Run Program (Preferable on hpc)
Run `channels_test.py` to run all of the methods tested on all methods on all models and on all different data split types.
The script is saving two different types of files:

1. Parameters_file = parameters for **every** model evaluated in the process. include dataframe with the follwing features:

| Feature | Description |
| --- | --- |
| bands     | bands inds out of 840, used in the model |
| train_score | model score for train set |
| test_score | model score for test set  |
|  train_samples | avocado sample numbers of the training set  |
|  test_samples | avocado sample numbers of the test set  |
| train_labels | avocado sample labels of the training set |
|  test_labels | avocado sample labels of the test set  |
|  cv_score | cv score of the model (standard deviation / avarge) |

2. Best_Model = the best model for every "fixed" band number - 'joblib' file:

The location of all files is described in this main folders (and are splitted between sub-folders for each method):

| File Type | Data Split Type |Main Folder | 
| --- | --- |--- |
| Parameters_file     | Random |`channels`     | 
| Parameters_file | Plantation -1 |`channels_plantation\train23_test1`      |
|  Parameters_file | Plantation -2  |`channels_plantation\train13_test2`     |
|  Parameters_file | Plantation -3 |`channels_plantation\train12_test3`      |  
| Best_Model     | Random | `best_models`     |
| Best_Model | Plantation -1 | `best_models_plantations\train23_test1`     |  
|  Best_Model  | Plantation -2  |`best_models_plantations\train13_test2`      |   
|  Best_Model | Plantation -3 |`best_models_plantations\train12_test3`      | 


# Analyzing the results
After saving the best models for every band number in the process, we need to understand which of the methods/models/data split type/bands are the best for us.

## Channels analysys
Run `channels_analysys_all.py`, for every model, method and data split combination, this script:
1. saves a "summery" feather file in `analyze` folder. For every method+model+data split type combination, the analysys saves the results for all of models saved,as well as for only 5 best models. **5 best models are selected by a value describe the "worst case"- the mean error of the test set+ the standard deviation.**

The files combines all of the parameters of the selected models as saved in the training process.
2. plot graphs in `graphs` and `grpahs_plantations` folders in sub-folder: `best_channel_number` - presenting the test error and the standard devieation of this error, as the number of bands is changing- for example:

<p align="center">
<img src="https://github.com/adico103/Avocado_Research/blob/main/xgboost_random_select_methodch_num_test.png" width=50%>
</p>

 and also the test vs train as a function of number of bands- for example::
<p align="center">

<img src="https://github.com/adico103/Avocado_Research/blob/main/svm_drop_one_methodch_num_train_test.png" width=50%>
</p>

3. plot graphs in `graphs` and `grpahs_plantations` folders in sub-folder: `total_best_wl` - presenting the weighted score of the importance of the wavelengths by all of the models saved- for example:
 
<p align="center">
<img src="https://github.com/adico103/Avocado_Research/blob/main/random_forest_drop_one_methodtot_best_wl83models.png" width=50%>
</p>

Also the weighted score of the importance of the wavelengths by the best 5 models- for example:
<p align="center">
<img src="https://github.com/adico103/Avocado_Research/blob/main/random_forest_drop_one_methodtot_best_wl5models.png" width=50%>
</p>

Also the weighted score of the importance of the wavelengths by the best model- for example:
<p align="center">
<img src="https://github.com/adico103/Avocado_Research/blob/main/random_forest_drop_one_methodtot_best_wl1models.png" width=50%>
</p>

## Final Results

Run `results_random_split.ipynb`,`results_plantations-1.ipynb`,`results_plantations-2.ipynb`,`results_plantations-2.ipynb`. The notebooks plot the best models scores selected for every combination of model+method. 

It presnts a table for the test scores- for example:
<p align="center">
<img src="https://github.com/adico103/Avocado_Research/blob/main/Avarage%20Test%20Error.png" width=50%>
</p>






