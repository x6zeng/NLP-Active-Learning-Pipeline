# NLP-Active-Learning-Pipeline
This is the repository for our DSC180B section A12 Group B Quarter 2 Project, which consists of 2 machine learning models, with Active Learning approaches, that can be used to predict the relevance and sentiment toward China of the tweets posted by the members of the U.S. Congress, given the tweet's text content.

## Main Content
- __data__: folder to store data, including test data and other data. It is also used to store the results data for the author
  - test: folder to store the test data
  - raw: empty, folder to store the raw data
  - result: folder to store the result data
- __notebook__: folder to store the pre-development notebooks
  - analyses: notebooks containing the active learning results analyses.
  - explorations: code explorations for active learning pipeline.
  - model_comparison: all the pre-development code for relevance and sentiment model, as well as the comparison between models using different hyperparameters.
- __src__: folder to store the files of obtaining the dataset, building the features, and the code for the 2 models
  - `utilities.py` - script to preprocess the raw data
  - `Relevance_AL_Committee.py` - script to train the relevance model
  - `Sentiment_AL_Committee.py` - script to train the sentiment model

## Data Source
The data used in this project was provided by the staffs from the China Data Lab at UC San Diego. Click [here](https://drive.google.com/drive/folders/1VSYdGh12UNVNhfxbSeHRdANvHr5xF8Ea?usp=sharing) for data. If running the models with the raw data, please place the `SentimentLabeled_10112022.csv` in the folder `data/raw`.

## Important Files
- `Dockerfile`: contains the information for building the docker image
- `run.py`: the script to run the models. To run the models on test data, use the following command: 
  - `python3 run.py test`
- `submission.json`: contains the submission information