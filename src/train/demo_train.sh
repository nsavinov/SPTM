EXPERIMENT_OUTPUT_FOLDER=demo_L python train_action_predictor.py > action_log.txt && ACTION_EXPERIMENT_ID=demo_L python resave_weights.py action &
EXPERIMENT_OUTPUT_FOLDER=demo_R python train_edge_predictor.py > edge_log.txt && EDGE_EXPERIMENT_ID=demo_R python resave_weights.py edge &
