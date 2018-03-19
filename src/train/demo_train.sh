EXPERIMENT_OUTPUT_FOLDER=demo_L python train.py action > action_log.txt && ACTION_EXPERIMENT_ID=demo_L python resave_weights.py action &
EXPERIMENT_OUTPUT_FOLDER=demo_R python train.py edge > edge_log.txt && EDGE_EXPERIMENT_ID=demo_R python resave_weights.py edge &
