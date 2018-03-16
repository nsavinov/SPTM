from __future__ import print_function
import subprocess
import os
import time
import argparse
import shutil
import itertools

CHECK_FOLDER_EXISTENCE = True

def parse_param_string(param_string):
    param = {}
    tokens = param_string.split()
    if len(tokens):
        assert (len(tokens)%2 == 0), "Parameters string should have an even number of words"
        for name,val in zip(tokens[0::2],tokens[1::2]):
            param[name] = val
    return param

def make_params_command(command_dict):
    params_command = ''
    for k,v in command_dict.items():
        params_command += 'export {k}={v} ; '.format(k=k,v=v)
    return params_command

def make_eval_command(method, doom_env):
    if method == 'ours':
        eval_command = "nohup python -u test_navigation_quantitative.py {doom_env} policy".format(doom_env=doom_env)
    elif method == 'teach_and_repeat':
        eval_command = "nohup python -u test_navigation_quantitative.py {doom_env} teach_and_repeat".format(doom_env=doom_env)
    return eval_command

def make_graph_command(method, doom_env):
    graph_command = "nohup python -u build_graph.py {doom_env}".format(doom_env=doom_env)
    return graph_command

def make_command(method, doom_env, param_string='', exp_out_folder_prefix='eval'):
    param = parse_param_string(param_string)
    param['EXPERIMENT_OUTPUT_FOLDER'] = exp_out_folder_prefix + '_' + method + '_' + doom_env
    exp_folder = os.path.join('../../experiments', param['EXPERIMENT_OUTPUT_FOLDER'])
    outfile = os.path.join(exp_folder, 'log.out')
    graph_outfile = os.path.join(exp_folder, 'graph_log.out')
    if CHECK_FOLDER_EXISTENCE:
        assert (not os.path.exists(exp_folder)), "Experiment folder {} already exists".format(exp_folder)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
        os.makedirs(os.path.join(exp_folder, 'evaluation/graph_shortcuts'))
    param_command = make_params_command(param)
    eval_command =  make_eval_command(method, doom_env)
    graph_command = make_graph_command(method, doom_env)
    command = param_command + graph_command + ' > ' + graph_outfile + ' ; ' + eval_command + ' > ' + outfile
    with open(os.path.join(exp_folder, 'command.log'), 'w') as f:
        f.write(command)
    shutil.copy('../common/constants.py', exp_folder)
    return command

def wait_for_free_slot(procs, max_num_procs):
    while len(procs) >= max_num_procs:
        for proc in procs:
            if proc.poll() is not None:
                print(proc.pid, 'done')
                procs.remove(proc)
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run evaluation of goal-directed anvigation')
    parser.add_argument('--methods', metavar='METHODS', type=str, nargs='+', help='Methods')
    parser.add_argument('--doom-envs', metavar='DOOM_ENVS', type=str, nargs='+', help='Environments')
    parser.add_argument('--params',  metavar='PARAMS', type=str, nargs='+', help='Parameters')
    parser.add_argument('--exp-folder-prefix',  metavar='EXP_FOLDER_PREFIX', type=str, help='Prefix for the output folder', default='eval')
    parser.add_argument('--max-num-procs',  metavar='MAX_NUM_PROCS', type=int, help='Maximum number of evaluation runs to start in parallel', default=5)
    args = parser.parse_args()
    num_experiments = len(args.doom_envs)*len(args.methods)
    if len(args.params):
        assert (len(args.params) == num_experiments or len(args.params) == 1), "Number of params should equal to 1 or to the number of envs"
        if len(args.params) == 1:
            args.params *= num_experiments
    print('\nArgs:\n', args, '\n')
    commands = []
    for ne, (method, doom_env) in enumerate(itertools.product(args.methods,args.doom_envs)):
        commands.append(make_command(method,doom_env, param_string=args.params[ne], exp_out_folder_prefix='{}_{:02}'.format(args.exp_folder_prefix,ne)))
    procs = []
    for command in commands:
        wait_for_free_slot(procs, args.max_num_procs)
        proc = subprocess.Popen(command, shell=True)
        print('Starting PID: ', proc.pid, '\nCommand: ', command, '\n')
        procs.append(proc)
    print('\n *** Started all processes. Waiting to complete *** \n')
    wait_for_free_slot(procs, 1)
    print('\n *** All processes done *** \n')
