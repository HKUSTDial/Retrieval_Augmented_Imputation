import jsonlines

with jsonlines.open('/home/yangchenyu/Data_Imputation/imputation/results/show_movie/gpt-4o-mini_show_movie_evidence_confidence_reranker_top5.jsonl') as reader:
    for line in reader:
        prediction, ground_truth = line['prediction'], line['ground_truth']
        
        if prediction is None:
            print(line['tuple_id'])
            print(line['input'])
            print(line['output'])