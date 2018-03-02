## Semi-parametric topological memory for navigation ##
#### In ICLR 2018 [[Project Website]](https://sites.google.com/view/sptm/) [[Demo Video]](https://youtu.be/vRF7f4lhswo)

[Nikolay Savinov¹](http://people.inf.ethz.ch/nsavinov/), [Alexey Dosovitskiy²](https://dosovits.github.io/), [Vladlen Koltun²](http://vladlen.info/)<br/>
¹ETH Zurich, ²Intel Labs<br/>

<p align="center">
  <img src="misc/sptm.gif" width="480">
</p>

This is Tensorflow/Keras implementation of our [ICLR 2018 paper on semi-parametric topological memory for navigation](https://openreview.net/pdf?id=SygwwGbRW).
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
git clone -b 2018_iclr_release --single-branch https://github.com/nsavinov/mapping.git
cd mapping
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
nohup bash demo.sh &
# in case you need to re-run demo.sh, run the clean-up script first
# we require the experiment folder to be unique, so the script below will remove the results from the previous run
bash cleanup_demo.sh
```
### Demo results progress tracking and interpretation
The following steps happen when you run demo.sh:
* A memory graph is computed and saved. A log is written into ../../experiments/demo*/graph_shortcuts.out . The graph is visualized in ../../experiments/demo*/evaluation/*_graph.pdf . The blue lines are temporal edges and the red lines are visual shortcut edges.
* A navigation test is run. A log is written into ../../experiments/demo*/log.out , other output data is written into ../../experiments/demo*/evaluation/* . The test consists of many sub-tests, one for every maze/starting point/goal/random seed. For every sub-test, we produce a navigation video and a navigation track of the agent. Their naming is given by a suffix _TRIAL'INDEX_STARTING'POINT'INDEX_GOAL'NAME.mov[.pdf].
* Plots similar to Figure 5 in the paper are produced from the log file and saved in the current directory as pdf-files named after test environments (one plot for each environment). The x-axis is the time limit to reach the goal (measured in VizDoom simulator steps) and the y-axis is the success rate of the agent for this time limit (the higher the better, measured in percents).
* Metrics success@5000 (used for Table 1 in the paper) are computed and saved to table_results.txt in the current directory.

### Changing SPTM hyperparameters for demo
You can try out non-default memory parameters for the demo by
```Shell
nohup bash demo.sh "PARAMETER_NAME_1 PARAMETER_VALUE_1 ... PARAMETER_NAME_N PARAMETER_VALUE_N" &
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
Coming soon! (by mid-March)

### Map generation
Coming soon (by mid-March)

### Acknowledgements
We would like to thank Anastasia Savinova for helping with the demo video.
