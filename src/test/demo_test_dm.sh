PREFIX=demo_test
python run_eval.py --max-num-procs 4 --methods ours --doom-envs columns_dm deepmind_small_dm office1_dm deepmind_large_dm --params "$1" --exp-folder-prefix $PREFIX
bash plot_all.sh ../../experiments/${PREFIX}*/log.out
