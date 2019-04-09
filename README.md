## Semi-parametric topological memory for navigation ##
#### In ICLR 2018 [[Project Website]](https://sites.google.com/view/sptm/) [[Demo Video]](https://youtu.be/vRF7f4lhswo)

[Nikolay Savinov¹](http://people.inf.ethz.ch/nsavinov/), [Alexey Dosovitskiy²](https://dosovits.github.io/), [Vladlen Koltun²](http://vladlen.info/)<br/>
¹ETH Zurich, ²Intel Labs<br/>

<p align="center">
  <img src="misc/sptm.gif" width="480">
</p>

This is Tensorflow/Keras implementation of our [ICLR 2018 paper on semi-parametric topological memory for navigation](https://arxiv.org/abs/1803.00653).
If you use this work, please cite:

    @inproceedings{SavinovDosovitskiyKoltun2018_SPTM,
        Author = {Savinov, Nikolay and Dosovitskiy, Alexey and Koltun, Vladlen},
        Title = {Semi-parametric topological memory for navigation},
        Booktitle = {International Conference on Learning Representations ({ICLR})},
        Year = {2018}
    }
    
### Requirements
The code was tested for Linux only. It requires around 12Gb of GPU memory to run the demo. All versions of packages used in this code are fixed and specified in the following installation scripts - so the results for the demo should exactly reproduce the results in the paper. The code assumes that the command "python" invokes python2.

### Installation
```Shell
# install vizdoom dependencies as explained in https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_deps
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev libboost-all-dev

# install anaconda2 if you don't have it yet
wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
bash Anaconda2-4.4.0-Linux-x86_64.sh
source ~/.profile
# or use source ~/.bashrc - depending on where anaconda was added to PATH as the result of the installation
# now anaconda is assumed to be in ~/anaconda2

# create conda environment named sptm and install all dependencies into it
git clone https://github.com/nsavinov/SPTM.git
cd SPTM
./setup.sh

# download models
cd experiments
./download_models.sh
```
### Demo
The scripts below produce navigation videos, trajectories, success rate plots and success@5000 metrics:
```Shell
source activate sptm
cd src/test
# test downloaded models on all environments 
nohup bash demo_test.sh &
# in case you need to re-run demo_test.sh, run the clean-up script first
# we require the experiment folder to be unique, so the script below will remove the results from the previous run
bash cleanup_demo_test.sh
```
### Demo results progress tracking and interpretation
The following steps happen when you run demo_test.sh:
* A memory graph is computed and saved. A log is written into ../../experiments/demo_test*/graph_shortcuts.out . The graph is visualized in ../../experiments/demo_test*/evaluation/*_graph.pdf . The blue lines are temporal edges and the red lines are visual shortcut edges.
* A navigation test is run. A log is written into ../../experiments/demo_test*/log.out , other output data is written into ../../experiments/demo_test*/evaluation/* . The test consists of many sub-tests, one for every maze/starting point/goal/random seed. For every sub-test, we produce a navigation video and a navigation track of the agent. Their naming is given by a suffix _TRIAL'INDEX_STARTING'POINT'INDEX_GOAL'NAME.mov[.pdf].
* Plots similar to Figure 5 in the paper are produced from the log file and saved in the current directory as pdf-files named after test environments (one plot for each environment). The x-axis is the time limit to reach the goal (measured in VizDoom simulator steps) and the y-axis is the success rate of the agent for this time limit (the higher the better, measured in percents).
* Metrics success@5000 (used for Table 1 in the paper) are computed and saved to table_results.txt in the current directory.

### Changing SPTM hyperparameters for demo
You can try out non-default memory parameters for the demo by
```Shell
nohup bash demo_test.sh "PARAMETER_NAME_1 PARAMETER_VALUE_1 ... PARAMETER_NAME_N PARAMETER_VALUE_N" &
```
All parameters are described below in the format "PARAMETER_NAME DEFAULT_PARAMETER_VALUE", their relation to the paper is also explained:
```python
# basic or temporally consistent (smoothed) localization; see Section 3.1, "Finding the waypoint"
SMOOTHED_LOCALIZATION 0
# every n-th frame from walkthrough video left in memory; see Section 4.1.1 Hyperparameters
MEMORY_SUBSAMPLING 4
# loops are not closed if start/end are temporally closer than this value; \delta T_l in the paper
MIN_SHORTCUT_DISTANCE 5
# s_local in the paper
WEAK_INTERMEDIATE_REACHABLE_GOAL_THRESHOLD 0.7
# s_reach in the paper
INTERMEDIATE_REACHABLE_GOAL_THRESHOLD 0.95
# number of shortcuts in the graph; see Section 4.1.1 Hyperparameters
SMALL_SHORTCUTS_NUMBER 2000
# for temporally consistent shortcuts -- median is computed over this window; \delta T_w in the paper
SHORTCUT_WINDOW 10
# min step to take on the shortest path; H_min in the paper
MIN_LOOK_AHEAD 1
# max step to take on the shortest path; H_max in the paper
MAX_LOOK_AHEAD 7
```
### Environment naming
Here we describe how environment names in the paper map into the names in the code:
```python
# format:
# NAME_PAPER NAME_CODE
Test-1 deepmind_small
Test-2 open_space_five
Test-3 star_maze
Test-4 office1
Test-5 columns
Test-6 office2
Test-7 topological_star_easier
Val-1 open_space_two
Val-2 branching
Val-3 deepmind_large
```
We also used suffix '_dm' for homogenious textures experiments and '_autoexplore' for automatic exploration experiments (see the paper supplementary).

### Training
```Shell
# train models: both R and L networks
# R-network is called 'edge' and L-network is called 'action' throughout the code
cd src/train
nohup bash demo_train.sh &
# in case you need to re-run demo_train.sh, run the clean-up script first
# we require the experiment folder to be unique, so the script below will remove the results from the previous run
bash cleanup_demo_train.sh
# test newly trained models
cd ../test
nohup bash demo_test.sh "ACTION_EXPERIMENT_ID demo_L EDGE_EXPERIMENT_ID demo_R" &
```
While training, you can track the progress via tensorboard logs in ../../experiments/demo_L/logs and ../../experiments/demo_R/logs. The training takes a week and requires approximately 10Gb of RAM.

### Supplementary experiments
```Shell
# test on automatic exploration mazes
cd src/test
nohup bash demo_test_autoexplore.sh &
# test pre-trained homogenious textures models on homogenious textures mazes
nohup bash demo_test_dm.sh "ACTION_EXPERIMENT_ID 0104_L EDGE_EXPERIMENT_ID 0105_R" &
# train new homogenious textures models
cd src/train
TRAIN_WAD_PATH=Train-DM/D3_battle_navigation_split.wad_manymaps_test.wad nohup bash demo_train.sh &
# test newly trained homogenious textures models
cd src/test
nohup bash demo_test_dm.sh "ACTION_EXPERIMENT_ID demo_L EDGE_EXPERIMENT_ID demo_R" &
```

### Maze generation
The code below creates mazes with the same distribution of textures as those in the paper:
```Shell
cd src/data_generation
# training
# normal mazes
python apply_random_textures.py --input ../../data/Train/D3_battle_navigation_split.wad --texture-sparsity dense --mode train
# homogenious textures mazes
python apply_random_textures.py --input ../../data/Train-DM/D3_battle_navigation_split.wad --texture-sparsity sparse --mode train
# test
# normal mazes
python apply_random_textures.py --input ../../data/Val/deepmind_large/deepmind_large.wad --texture-sparsity dense --mode test
# after running script above, move the starting points manually
# homogenious textures mazes
python apply_random_textures.py --input ../../data/Val/deepmind_large_dm/deepmind_large.wad --texture-sparsity sparse --mode test
# after running script above, move the starting points manually
```

### Related projects
Check out the code for [Episodic Curiosity](https://github.com/google-research/episodic-curiosity), which addresses the exploration problem using episodic memory and R-network (could be used as a source of exploration video for SPTM).

### Acknowledgements
We would like to thank Anastasia Savinova for helping with the demo video.
