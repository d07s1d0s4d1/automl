python3 app/train_stack.py --mode $1 --train-csv /data/check_${2}/train.csv --model-dir /models &&
python3 app/predict_stack.py --test-csv /data/check_${2}/test.csv --test-target-csv /data/check_${2}/test-target.csv --prediction-csv /models/prediction_${2}.csv --model-dir /models
