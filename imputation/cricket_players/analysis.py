import jsonlines

with jsonlines.open('/home/yangchenyu/Data_Imputation/imputation/results/cricket_players/gpt-4o-mini_cricket_players_evidence_confidence_reranker_top5.jsonl') as reader:
    for line in reader:
        prediction, ground_truth = line['prediction'], line['ground_truth']
        
        try:
            if prediction['National Side'].lower() != ground_truth['National Side'].lower():
                print(line['tuple_id'])
        except:
            print(line['tuple_id'])
        try:
            if ('right' in prediction['Batting Style'].lower() and 'left' in ground_truth['Batting Style'].lower()) or ('left' in prediction['Batting Style'].lower() and 'right' in ground_truth['Batting Style'].lower()):
                print(line['tuple_id'])
        except:
            print(line['tuple_id'])