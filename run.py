import os
import sys
import json

from src.utilities import *
from src.Relevance_AL_Committee import * 
from src.Sentiment_AL_Committee import * 


def main(targets):

    try:
        if 'test' in targets:
            filepath = os.path.join('data/test', 'test_data.csv')
        elif 'raw' in targets:
            filepath = os.path.join('data/raw', 'tweet_data_2022.csv')
        print(filepath)

        df_relevance = get_data(filepath, 1)
        df_relevance_result = run_relevance_model(df_relevance)
        
        df_sentiment = get_data(filepath, 2)
        df_sentiment_result = run_sentiment_model(df_sentiment)

        df_relevance_result.to_csv('data/result/df_relevance_result.csv')
        df_sentiment_result.to_csv('data/result/df_sentiment_result.csv')

    except Exception as e:
        print(e)


if __name__ == "__main__":
    targets = sys.argv[1]
    main(targets)