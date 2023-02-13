from src.utilities import *

def trial_sentiment(df, model_names, training_method, balance, sampling_size, sort_by_time, partition_ratio):
    output = {}
    output['model_names'] = model_names
    output['training_method'] = training_method #random_sampling, active_learning
    output['balance'] = balance
    output['sampling_size'] = sampling_size
    output['sort_by_time'] = sort_by_time
    output['partition_ratio'] = partition_ratio
        
    # 1. Balance dataset 
    df_1, df_2, df_3 = df[df.SentimentScore=='Negative'], df[df.SentimentScore=='Neutral'], df[df.SentimentScore=='Positive']

    # 1.1 Balance the label distribution  (33% Negative vs. 33% Neutral vs. 33% Positive)
    if balance:
        sample_size = min(df_1.shape[0], df_2.shape[0], df_3.shape[0])
        if df_1.shape[0] > sample_size:
            df_1 = resample(df_1, replace=False, n_samples=sample_size, random_state=random_state)
        if df_2.shape[0] > sample_size:
            df_2 = resample(df_2, replace=False, n_samples=sample_size, random_state=random_state)
        if df_3.shape[0] > sample_size:
            df_3 = resample(df_3, replace=False, n_samples=sample_size, random_state=random_state)

    # 1.2 Keep the natural label distribution
    seed_1, unlabeled_1, test_1 = partition_data(df_1, partition_ratio, sort_by_time)
    seed_2, unlabeled_2, test_2 = partition_data(df_2, partition_ratio, sort_by_time)
    seed_3, unlabeled_3, test_3 = partition_data(df_3, partition_ratio, sort_by_time)
    seed, unlabeled, test = pd.concat([seed_1, seed_2, seed_3]), pd.concat([unlabeled_1, unlabeled_2, unlabeled_3]), pd.concat([test_1, test_2, test_3])
    output['seed_size'], output['unlabeled_size'], output['test_size'] = seed.shape[0], unlabeled.shape[0], test.shape[0]
    
    initial_seed = seed.copy()
    initial_unlabeled = unlabeled.copy()
    
    # 2. Train the model
    initial_pipelines, initial_accuracy = train_model(initial_seed,2)
    
    # 3. Active Learning
    if sampling_size == 0:
        pipelines, accuracy = initial_pipelines, initial_accuracy
        
    # 3.1 Initial Model + Random Sampling
    elif training_method == 'random_sampling':
        if initial_unlabeled.shape[0] >= sampling_size:
            sample_unlabeled = initial_unlabeled.sample(n=sampling_size, replace=False, random_state=random_state)
        else:
            sample_unlabeled = initial_unlabeled.sample(n=sampling_size, replace=True, random_state=random_state)
        seed_and_sample_unlabeled_df = pd.concat([initial_seed, sample_unlabeled])
        pipelines, accuracy = train_model(seed_and_sample_unlabeled_df,2)
        
    # 3.2 Initial Model + Active Learning
    else:
        pipelines, accuracy = active_learning(initial_pipelines, initial_seed, initial_unlabeled, sampling_size,2)

    # 4. Report Model Accuracy
    X_test, Y_test = test[['text_cleaned']], test[['SentimentScore']]

    pl_preds = []
    for pl in pipelines:
        # compute the testing accuracy
        prediction = pl.predict(pd.DataFrame(X_test))
        pl_preds.append([pl, prediction])
        
    #Saves all the model predictions
    all_preds = np.array([x[1] for x in pl_preds]).transpose()
    #Find the mode in all preds
    prediction = [mode(i) for i in all_preds]
    accuracy = round(accuracy_score(Y_test, prediction), round_number)
    f1_micro = round(f1_score(np.array(Y_test), prediction, average='micro'), round_number)
    f1_macro = round(f1_score(np.array(Y_test), prediction, average='macro'), round_number)
    f1_weighted = round(f1_score(np.array(Y_test), prediction, average='weighted'), round_number)
    
    precision_micro = round(precision_score(np.array(Y_test), prediction, average='micro'), round_number)
    precision_macro = round(precision_score(np.array(Y_test), prediction, average='macro'), round_number)
    precision_weighted = round(precision_score(np.array(Y_test), prediction, average='weighted'), round_number)
        
    recall_micro = round(recall_score(np.array(Y_test), prediction, average='micro'), round_number)
    recall_macro = round(recall_score(np.array(Y_test), prediction, average='macro'), round_number)
    recall_weighted = round(recall_score(np.array(Y_test), prediction, average='weighted'), round_number)
    
    output['accuracy'] = accuracy
    output['f1_micro'], output['f1_macro'], output['f1_weighted'] = f1_micro, f1_macro, f1_weighted
    output['precision_micro'], output['precision_macro'], output['precision_weighted'] = precision_micro, precision_macro, precision_weighted
    output['recall_micro'], output['recall_macro'], output['recall_weighted'] = recall_micro, recall_macro, recall_weighted
    return output


def run_sentiment_model(df):
    training_method = ['random_sampling']
    balanced = [True]
    sampling_size = [100]
    sort_by_time = [True]
    partition_ratio = [[0.5, 0.25, 0.25]]

    model_result_df = pd.DataFrame()
    index = 1
    model_name = "SVC, KNN, Decision Tree, Random Forest, AdaBoost"
    for tm in training_method:
        for b in balanced:
            for ss in sampling_size:
                for t in sort_by_time:
                    for r in partition_ratio:
                        model_output = trial_sentiment(df, model_name, tm, b, ss, t, r)
                        if index == 0:
                            model_result_df = pd.DataFrame(model_output, index=index)
                        else:
                            model_result_df = model_result_df.append(pd.DataFrame([model_output],index=[index]))
                        index += 1

    return model_result_df