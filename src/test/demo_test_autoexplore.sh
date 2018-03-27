PREFIX=demo_test
python run_eval.py --max-num-procs 4 --methods ours --doom-envs columns_autoexplore deepmind_small_autoexplore office1_autoexplore --params "$1" --exp-folder-prefix $PREFIX
bash plot_all.sh ../../experiments/${PREFIX}*/log.out
