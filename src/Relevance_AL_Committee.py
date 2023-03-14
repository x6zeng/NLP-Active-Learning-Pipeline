from src.utilities import *

def trial_relevance(df, model_names, training_method, balance, sampling_size, sort_by_time, partition_ratio):
    output = {}
    output['model_names'] = model_names
    output['training_method'] = training_method #random_sampling, active_learning
    output['balance'] = balance
    output['sampling_size'] = sampling_size
    output['sort_by_time'] = sort_by_time
    output['partition_ratio'] = partition_ratio
    accuracy_lst, f1_lst, precision_lst, recall_lst, specificity_lst = [], [], [], [], []

    for i in range(5):  
        # 1. Balance dataset 
        df_1, df_2_3 = df[df.Bucket=='1'], df[df.Bucket=='2 or 3']
        df_lst = [df_1, df_2_3]
        
        # 1.1 Balance the label distribution  (50% Bucket 1 vs. 50% Non-Bucket 1)
        if balance:
            sample_size = min(df_1.shape[0], df_2_3.shape[0])
            if df_1.shape[0] > sample_size:
                df_1 = resample(df_1, replace=False, n_samples=sample_size, random_state=random_state)
            if df_2_3.shape[0] > sample_size:
                df_2_3 = resample(df_2_3, replace=False, n_samples=sample_size, random_state=random_state)
                
        # 1.2 Keep the natural label distribution
        seed_1, unlabeled_1, test_1 = partition_data(df_1, partition_ratio, sort_by_time)
        seed_2_3, unlabeled_2_3, test_2_3 = partition_data(df_2_3, partition_ratio, sort_by_time)
        seed, unlabeled, test = pd.concat([seed_1, seed_2_3]), pd.concat([unlabeled_1, unlabeled_2_3]), pd.concat([test_1, test_2_3])
        output['seed_size'], output['unlabeled_size'], output['test_size'] = seed.shape[0], unlabeled.shape[0], test.shape[0]
        
        initial_seed = seed.copy()
        initial_unlabeled = unlabeled.copy()
        
        # 2. Train the model
        initial_pipelines, initial_accuracy = train_model(initial_seed,1)
        
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
            pipelines, accuracy = train_model(seed_and_sample_unlabeled_df,1)
            
        # 3.2 Initial Model + Active Learning
        else:
            pipelines, accuracy = active_learning(initial_pipelines, initial_seed, initial_unlabeled, sampling_size, 1)

        # 4. Report Model Accuracy
        X_test, Y_test = test[['text_cleaned']], test[['Bucket']]
        

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
        f1 = round(f1_score(np.array(Y_test), prediction, pos_label='1'), round_number)
        precision = round(precision_score(np.array(Y_test), prediction, pos_label='1', average='binary'), round_number)
        recall = round(recall_score(np.array(Y_test), prediction, pos_label='1', average='binary'), round_number)
        specificity = round(recall_score(np.array(Y_test), prediction, pos_label='2 or 3', average='binary'), round_number)

        accuracy_lst.append(accuracy)
        f1_lst.append(f1)
        precision_lst.append(precision) 
        recall_lst.append(recall) 
        specificity_lst.append(specificity) 

    output['accuracy'] = accuracy
    output['f1_score'] = f1
    output['precision'] = precision
    output['recall'] = recall
    output['specificity'] = specificity
    return output


def run_relevance_model(df):
    training_method = ['random_sampling']
    balanced = [True]
    sampling_size = [10]
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
                        model_output = trial_relevance(df, model_name, tm, b, ss, t, r)
                        if index == 0:
                            model_result_df = pd.DataFrame(model_output, index=index)
                        else:
                            model_result_df = model_result_df.append(pd.DataFrame([model_output],index=[index]))
                        index += 1

    return model_result_df