from collections import OrderedDict

import matplotlib.pyplot as plt
from pathlib import Path
import os
import math
import argparse

import numpy as np
from scipy.special import expit, erf
from numpy.fft import fft, ifft
from scipy.interpolate import UnivariateSpline, CubicSpline

import torch
import torch.nn.functional as F

sq2 = math.sqrt(2)


##########################################
############ PARAMS STRUCTURE ############
##########################################

class Pars:
    def __init__(self):
        
        ### TASK PARAMS
        ## general
        self.dataset = "RANDOM"        # dataset name
        self.alpha_train = 1.5         # fraction of training patterns wrt N in a random task
        self.alpha_test = 0            # fraction of test patterns wrt N in a random task
        self.span_h = False            # generate all h in range (only makes sense for N=2 or N=3)
        self.exp_input = False         # exponentially distributed input (default is gaussian)
        self.rho = 0.                  # input covariance
        self.pars_teacher = {}
        ## NLGP specific options
        self.torus = False             # whether to use a torus topology
        self.g_nonlin = 1e-10          # gain parameter for nonlinearity
        self.xis = [1e-10, 2]          # lenght-scales of mixtures
        ## spatial options
        self.dim = 1                   # input dimension
        self.lN = 40                   # linear input dimension
        self.normalize = False         # normalize inputs (some dataset are already normalized)
        ## misc options
        self.cuda = True
        self.seed_data = 1
        self.data_dir = f'data'
        self.save_dir = f'results_notebook'
        
        ## NETWORK PARAMS
        ## general
        self.seed_net = 1
        self.nonlinearity = "linear"    # activation function: relu | tanh | linear
        self.discrete_time = False      # time discretization: True | False
        self.recurrent_type = "rate"    # recurrent 
        self.K = 40                     # number of hidden units in the recurrent network
        self.mean_field_init = True     # init readout weight with std ~ 1/K
        ## structural
        self.no_self = False            # no self connection
        self.g = 0.5                    # initial spectral radius
        self.fe = -1                    # fraction of E neurons, -1 for uncontrained weights
        self.positive_currents = False  # impose posivity constraints on input currents
        self.out_sign = False           # when True applies sign constraint to projective synapses
        self.out_act = False            # sets the readouts as real neurons with "nonlinearity" activation function
        self.x_out = False              # take w_out = Id and don't train it
        self.maxweight = 10.            # maximum allowed absolute value of the weights
        self.eye_in = False             # identity input matrix
        self.eye_J = False              # identity recurrent matrix
        self.soft_w = False             # readouts set to 1/sqrt(K)
        self.embed_model = False        # generate positions to embed model in space
        self.square_dist = False        # whether to square distances
        self.tau = 1.                   # neuron time scale
        self.dt = 0.05                  # integration time step
        ## implicit rnn
        self.implicit = True            # use implicit optimization
        self.rule_type = "dw1"          # type of Rosenbaum's rule: dw1 | {dw2 | dw3} : rsbm rules
        self.newton = False             # use newton method in implicit model
        self.tol_implicit = 1e-4        # tolerance for reaching a fixed point
        self.max_iter_implicit = 1000   # max number of iterations within implicit model
        
        ## GENERAL TASK OPTIONS
        self.classifier = False         # use network to classify inputs at fixed point
        ## explicit only options
        self.titot = 1                  # total number of iterations (only valid in the explicit case)
        self.init_zero = True           # init network on zero activity
        self.input_dur = 1              # input duration in timesteps
        self.amp_input = 1.             # input amplitude
        self.last_times = 10            # number of final timesteps to generate output

        ### TRAINING OPTIONS
        ## set loss
        self.loss_name = "mse"          # loss function name: mse | ce
        ## set training of each layer and biases
        self.train_w_in = False
        self.train_J = True
        self.train_w = False
        self.train_bias_in = False
        self.train_bias_w = False
        ## regularization for position embedding
        self.gamma_reg_pos = [0., 0., 0.]
        self.reg_dist_type = "power1"
        ## training and regularization options
        self.batch_size = 512
        self.num_epochs = 1
        self.test_every = 1
        self.L1 = 0.                    # L1 weight regularization
        self.L2 = 0.                    # L2 weight regularization
        self.alpha = 0.                 # energy cost
        self.en_power_J = 2.            # J exponent for energy cost
        self.en_power_act = 1.          # act exponent for energy cost
        self.cost_bias_in = 0.          # bias_in cost
        self.cost_bias_w = 0.           # bias_w cost
        self.gamma_soft_sign = 0.       # enforcing sign constraints with a potential
        self.gamma_sign_reg = 0.        # synaptic sign regularization
        self.sign_reg_quadratic = False # whether to use quadratic sign reg
        self.out_sign_reg = False       # synaptic sign regularization on projective synapses
        self.optimizer = 'SGD'          # SGD and Adam currently available
        self.lr = 0.05                  # learning rate
        self.momentum = 0.
        self.max_acc_train = 1+1
        self.max_acc_test = 1+1
        ## options for GPU use, keep true for now
        self.train_set_on_device = True
        self.test_set_on_device = True
        ## storage options
        self.write_to_allresults = True # whether to write final results in a common file
        
        

##########################################################
############ READ ARGUMENTS FROM COMMAND LINE ############
##########################################################
        
def read_args():

    parser = argparse.ArgumentParser()

    # whether to read some parameters from par_vec
    parser.add_argument("--par_vec", default=False, action="store_true", help="read pars from par_vec")
    # dataset
    parser.add_argument("--dataset", type=str, default="RANDOM", help="dataset")
    parser.add_argument("--normalize", default=False, action="store_true", help="normalize dataset")
    # teacher-student options
    parser.add_argument("--K_teacher", type=int, default=1, help="K teacher")
    parser.add_argument("--nonlinearity_teacher", default="tanh", help="activation function : sigmoid | tanh | erf | relu | linear")
    parser.add_argument("--discrete_time_teacher", default=False, action="store_true")
    parser.add_argument("--dt_teacher", type=float, default=0.1, help="dt teacher")
    parser.add_argument("--eye_in_teacher", default=False, action="store_true")
    parser.add_argument("--g_bias_in_teacher", type=float, default=0., help="g bias_in teacher")
    parser.add_argument("--soft_w_teacher", default=False, action="store_true")
    parser.add_argument("--num_out_teacher", type=int, default=1, help="K teacher")
    parser.add_argument("--g_teacher", type=float, default=1, help="g teacher")
    # random dataset options
    parser.add_argument("--alpha_train", type=float, default=1., help="alpha_train for random dataset")
    parser.add_argument("--alpha_test", type=float, default=0., help="alpha_test for random dataset")
    parser.add_argument("--span_h", default=False, action="store_true", help="span_h input")
    parser.add_argument("--exp_input", default=False, action="store_true", help="exp distributed input")
    parser.add_argument("--rho", type=float, default=0., help="input covariance")
    # NLGP options
    parser.add_argument("--torus", default=False, action="store_true", help="torus topology")
    parser.add_argument("--g_nonlin", type=float, default=2., help="NLGP gain")
    parser.add_argument("--xi1", type=float, default=1e-5, help="NLGP xi1")
    parser.add_argument("--xi2", type=float, default=4., help="NLGP xi2")
    # spatial options
    parser.add_argument("--dim", type=int, default=1, help="dimension")
    parser.add_argument("--lN", type=int, default=10, help="linear dimension")
    # model options
    parser.add_argument("--nonlinearity", type=str, default="relu", help="activation function")
    parser.add_argument("--discrete_time", default=False, action="store_true", help="discrete time")
    parser.add_argument("--recurrent_type", type=str, default="rate", help="recurrent_type")
    parser.add_argument("--num_hidden", type=int, default=10, help="number of hidden units")
    parser.add_argument("--mean_field_init", default=False, action="store_true", help="mean field iniitalization for readout weights")
    parser.add_argument("--no_self", default=False, action="store_true", help="no self connection")
    parser.add_argument("--g", type=float, default=0.4, help="g")
    parser.add_argument("--fe", type=float, default=0.5, help="fraction of E neurons")
    parser.add_argument("--positive_currents", default=False, action="store_true", help="positive currents")
    parser.add_argument("--out_sign", default=False, action="store_true", help="projective sign constraints")
    parser.add_argument("--out_act", default=False, action="store_true", help="readout activation function")
    parser.add_argument("--x_out", default=False, action="store_true", help="set output as x")
    parser.add_argument("--eye_in", default=False, action="store_true", help="eye_in")
    parser.add_argument("--eye_J", default=False, action="store_true", help="eye_J")
    parser.add_argument("--soft_w", default=False, action="store_true", help="soft w")
    parser.add_argument("--embed_model", default=False, action="store_true", help="embed model")
    parser.add_argument("--square_dist", default=False, action="store_true", help="square_dist")
    parser.add_argument("--dt", type=float, default=0.1, help="dt")
    parser.add_argument("--implicit", default=False, action="store_true", help="implicit model")
    parser.add_argument("--rule_type", type=str, default="dw1", help="rule type")
    parser.add_argument("--newton", default=False, action="store_true", help="newton model for implicit model")
    # task options
    parser.add_argument("--classifier", default=False, action="store_true", help="classifier")
    parser.add_argument("--titot", type=int, default=10, help="titot")
    parser.add_argument("--input_dur", type=int, default=10, help="input duration")
    parser.add_argument("--amp_input", type=float, default=1., help="input amplitude")
    parser.add_argument("--last_times", type=int, default=10, help="last times")
    # learning options
    parser.add_argument("--loss", type=str, default="ce", help="loss function")
    parser.add_argument("--train_w_in", default=False, action="store_true", help="train w_in")
    parser.add_argument("--train_J", default=False, action="store_true", help="train J")
    parser.add_argument("--train_w", default=False, action="store_true", help="train w")
    parser.add_argument("--train_bias_in", default=False, action="store_true", help="trainbias_in")
    parser.add_argument("--train_bias_w", default=False, action="store_true", help="train bias_w")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=1000, help="num_epochs")
    parser.add_argument("--test_every", type=int, default=10, help="test_every")
    parser.add_argument("--L1", type=float, default=0., help="L1 reg")
    parser.add_argument("--L2", type=float, default=0., help="L2 reg")
    parser.add_argument("--alpha", type=float, default=0., help="energy regularization")
    parser.add_argument("--en_power_J", type=int, default=2, help="energy exponent J")
    parser.add_argument("--en_power_act", type=int, default=2, help="energy exponent act")
    parser.add_argument("--sign_reg_quadratic", default=False, action="store_true", help="sign_reg_quadratic")
    parser.add_argument("--gamma_reg_pos_in", type=float, default=0., help="gamma_reg_pos_in")
    parser.add_argument("--gamma_reg_pos_J", type=float, default=0., help="gamma_reg_pos_J")
    parser.add_argument("--gamma_reg_pos_out", type=float, default=0., help="gamma_reg_pos_out")
    parser.add_argument("--reg_dist_type", default=None, help="reg disttype")
    parser.add_argument("--cost_bias_in", type=float, default=0., help="cost bias in")
    parser.add_argument("--cost_bias_w", type=float, default=0., help="cost bias w")
    parser.add_argument("--gamma_soft_sign", type=float, default=0., help="gamma soft sign")
    parser.add_argument("--gamma_sign_reg", type=float, default=0., help="sign regularization")
    parser.add_argument("--out_sign_reg", default=False, action="store_true", help="projective sign regularization")
    parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0., help="momentum")
    # general options
    parser.add_argument("--cuda", default=False, action="store_true", help="use cuda")
    parser.add_argument("--data_dir", type=str, default="data", help="data directory")
    parser.add_argument("--save_dir", type=str, default="results", help="save directory")
    parser.add_argument("--seed_data", type=int, default=1, help="seed data")
    parser.add_argument("--seed_net", type=int, default=1, help="seed net")
    parser.add_argument("--allresults", default=False, action="store_true", help="write to allresults file")

    # EXTRACT COMMAND LINE OPTIONS
    args = parser.parse_args()
#     print(vars(args))
    
    return args
    

    
####################################################
############ SET PARS FROM COMMAND LINE ############
####################################################
    
def set_pars(pars, args):
    # dataset options
    pars.dataset = args.dataset
    # teacher-student options
    pars.pars_teacher = {}
    pars.pars_teacher["K"] = args.K_teacher
    pars.pars_teacher["nonlinearity"] = args.nonlinearity_teacher
    pars.pars_teacher["discrete_time"] = args.discrete_time_teacher
    pars.pars_teacher["dt"] = args.dt_teacher
    pars.pars_teacher["eye_in"] = args.eye_in_teacher
    pars.pars_teacher["g_bias_in"] = args.g_bias_in_teacher
    pars.pars_teacher["soft_w"] = args.soft_w_teacher
    pars.pars_teacher["num_out"] = args.num_out_teacher
    pars.pars_teacher["g"] = args.g_teacher
    pars.pars_teacher["J"] = None
    # random dataset options
    pars.alpha_train = args.alpha_train
    pars.en_power_J = args.en_power_J
    pars.en_power_act = args.en_power_act
    pars.sign_reg_quadratic = args.sign_reg_quadratic
    pars.gamma_reg_pos = [args.gamma_reg_pos_in, args.gamma_reg_pos_J, args.gamma_reg_pos_out]
    pars.reg_dist_type = args.reg_dist_type
    pars.alpha_test = args.alpha_test
    pars.span_h = args.span_h
    pars.exp_input = args.exp_input
    pars.rho = args.rho
    # NLGP specific options
    torus = args.torus
    pars.g_nonlin = args.g_nonlin
    pars.xis = [args.xi1, args.xi2]
    pars.dim = args.dim
    pars.lN = args.lN
    pars.normalize = args.normalize
    # model options
    pars.nonlinearity = args.nonlinearity
    pars.discrete_time = args.discrete_time
    pars.recurrent_type = args.recurrent_type
    pars.K = args.num_hidden
    pars.mean_field_init = args.mean_field_init
    pars.no_self = args.no_self
    pars.g = args.g
    pars.fe = args.fe
    pars.positive_currents = args.positive_currents
    pars.out_sign = args.out_sign
    pars.out_act = args.out_act
    pars.x_out = args.x_out
    pars.eye_in = args.eye_in
    pars.eye_J = args.eye_J
    pars.soft_w = args.soft_w
    pars.embed_model = args.embed_model
    pars.square_dist = args.square_dist
    pars.dt = args.dt
    pars.implicit = args.implicit
    pars.rule_type = args.rule_type
    pars.newton = args.newton
    # task options
    pars.classifier = args.classifier
    pars.titot = args.titot
    pars.input_dur = args.input_dur
    pars.amp_input = args.amp_input
    pars.last_times = args.last_times
    # learning options
    pars.loss_name = args.loss
    pars.train_w_in = args.train_w_in
    pars.train_J = args.train_J
    pars.train_w = args.train_w
    pars.train_bias_in = args.train_bias_in
    pars.train_bias_w = args.train_bias_w
    pars.batch_size = args.batch_size
    pars.num_epochs = args.num_epochs
    pars.test_every = args.test_every
    pars.L1 = args.L1
    pars.L2 = args.L2
    pars.alpha = args.alpha
    pars.cost_bias_in = args.cost_bias_in
    pars.cost_bias_w = args.cost_bias_w
    pars.gamma_soft_sign = args.gamma_soft_sign
    pars.gamma_sign_reg = args.gamma_sign_reg
    pars.out_sign_reg = args.out_sign_reg
    pars.optimizer = args.optimizer
    pars.lr = args.lr
    momentum = args.momentum
    # general options
    pars.cuda = args.cuda
    pars.data_dir = args.data_dir
    pars.save_dir = args.save_dir
    pars.seed_data = args.seed_data
    pars.seed_net = args.seed_net
    pars.write_to_allresults = args.allresults
    dtype = torch.float
    device = set_cuda(args.cuda)
    
    return pars, dtype, device


    
###########################################    
############ GENERAL FUNCTIONS ############
###########################################

def set_cuda(cuda):
    if cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        print(f'determinstic cuda? {torch.backends.cudnn.deterministic}')
    else:
        device = torch.device("cpu")
    print("...using", device)
    return device


def makedir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def watch_dataset(train_dataset, test_dataset, pars):
    
    if "CIFAR" in pars.dataset:
        raise ValueError("doesn't work with CIFAR yet! Sorry!")

    cmap = 'gray'
    upto_dim1 = 40
    upto_dim2 = 5

    X_to_plot_train, y_to_plot_train = extract_tensors(train_dataset)
    if pars.num_test is not None:
        X_to_plot_test, y_to_plot_test = extract_tensors(test_dataset)

    if pars.dim == 1:
        rows = 2 if pars.num_test is not None else 1
        for m in range(pars.num_label):
            plt.subplot(rows, pars.num_label, m+1)
            plt.imshow(X_to_plot_train.cpu()[y_to_plot_train==m][:upto_dim1], cmap=cmap)
        plt.tight_layout()
        plt.show()

        if pars.num_test is not None:
            for y in range(pars.num_label):
                plt.subplot(rows, pars.num_label, pars.num_label+y+1)
                plt.imshow(X_to_plot_test.cpu()[y_to_plot_test==y][:upto_dim1], cmap=cmap)
            plt.tight_layout()
            plt.show()
    else:
        nrows = 2
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 5))
        perm = np.random.randint(len(X_to_plot_train), size=len(X_to_plot_train))

        count = 0
        for i in range(nrows):
            for j in range(ncols):
                y_to_plot_train
                ax[i,j].set_title(str(y_to_plot_train[perm][count].item()))
                if len(X_to_plot_train[perm][count].shape) == 1:
                    X = X_to_plot_train[perm][count].reshape(pars.lN, pars.lN)
                else:
                    X = X_to_plot_train[perm][count]
                ims = ax[i,j].imshow(X, cmap=cmap)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

                count += 1
        plt.tight_layout()
    
    
def set_flags(pars):

    ## set directory flag
    flag_dir = f"N{pars.nls[0]}_K{pars.nls[1]}"
    if pars.embed_model:
        flag_dir += f"_d{pars.dim}"
    if "RANDOM" in pars.dataset:
        flag_dir += "_at%g" % (pars.alpha_train)
        flag_dir += "_ex" if pars.exp_input else ""

    flag_dir += f"_Kt{pars.pars_teacher['K']}" if "Teacher" in pars.dataset else ""
    flag_dir += "_%s_%s_g%g_fe%g_%s" % (pars.nonlinearity,
                                            pars.recurrent_type,
                                            pars.g,
                                            pars.fe,
                                            pars.rule_type)
    flag_dir += "_ns" if pars.no_self else ""
    flag_dir += "_pc" if pars.positive_currents else ""
    flag_dir += "_os" if pars.out_sign else ""
    flag_dir += "_oa" if pars.out_act else ""
    flag_dir += "_xo" if pars.x_out else ""
    flag_dir += "_ey" if pars.eye_in else ""
    flag_dir += "_sw" if pars.soft_w else ""
    flag_dir += "_em" if pars.embed_model else ""
    flag_dir += "_sd" if pars.square_dist else ""

    ## set flag for dump file
    flag_dump = f"{pars.loss_name}"
    flag_dump += "_in" if pars.train_w_in else ""
    flag_dump += "_J" if pars.train_J else ""
    flag_dump += "_w" if pars.train_w else ""
    flag_dump += "_bin" if pars.train_bias_in else ""
    flag_dump += "_bw" if pars.train_bias_w else ""
    if pars.embed_model:
        flag_dump += "_grp_%g_%g_%g_%s" % (pars.gamma_reg_pos[0],
                                           pars.gamma_reg_pos[1],
                                           pars.gamma_reg_pos[2],
                                           pars.reg_dist_type)
    flag_dump += "_L1%g_L2%g_a%g_epJ%d_epa%d_cbi%g_cbw%g" % (pars.L1,
                                                          pars.L2,
                                                          pars.alpha,
                                                          pars.en_power_J,
                                                          pars.en_power_act,
                                                          pars.cost_bias_in,
                                                          pars.cost_bias_w)
    flag_dump += "_gss%g_gsr%g" % (pars.gamma_soft_sign,
                                   pars.gamma_sign_reg)
    flag_dump += "_srq" if pars.sign_reg_quadratic else ""
    flag_dump += "_osr" if pars.out_sign_reg else ""
    flag_dump += "_%s_lr%f_sd%d_sn%d" % (pars.optimizer,
                                         pars.lr,
                                         pars.seed_data,
                                         pars.seed_net)

    return flag_dir, flag_dump
    
    
def write_allresults_title(pars):
    
    ## extact params
    save_dir = pars.save_dir
    dataset = pars.dataset
    num_test = pars.num_test
    allresults_filename = pars.allresults_filename
    
    print(f"will write results to {allresults_filename}")
    if not os.path.isfile(allresults_filename):
        title = "# loss \t train_w_in \t train_J \t train_w \t train_bias_in \t train_bias_w \t gamma_reg_pos_in \t "
        title += "gamma_reg_pos_J \t gamma_reg_pos_w \t reg_dist_type \t L1 \t L2 \t alpha \t en_power_J \t en_power_act \ t cost_bias_in \t "
        title += "cost_bias_w \t gamma_soft_sign \t gamma_sign_reg \t sign_reg_quadratic \t out_sign_reg \t "
        title += "optimizer \t lr \t seed_data \t seed_net \t "
        title += "best_train \t best_test \t best_active_train \t best_energy_train \t best_active_test \t best_energy_test"
        with open(allresults_filename, "w", buffering=1) as allresults_file:
            log(title, allresults_file, print_to_out=False)

            
def write_allresults(pars, results):
        
    ind_loss = 0 if pars.loss_name == "mse" else 1
    to_allresults_file = "%s \t %d \t %d \t %d \t %d \t %d \t" % (ind_loss,
                                                               pars.train_w_in * 1,
                                                               pars.train_J * 1,
                                                               pars.train_w * 1,
                                                               pars.train_bias_in * 1,
                                                               pars.train_bias_w * 1)
    
    ind_reg_dist = pars.reg_dist_type[-1]
    to_allresults_file += "%g \t %g \t %g \t %s \t %g \t %g \t" % (pars.gamma_reg_pos[0],
                                                               pars.gamma_reg_pos[0],
                                                               pars.gamma_reg_pos[0],
                                                               ind_reg_dist,
                                                               pars.L1,
                                                               pars.L2)
                                                               
    to_allresults_file += "%g \t %d \t %d \t %g \t %g \t %g \t %g \t" % (pars.alpha,
                                                               pars.en_power_J,
                                                               pars.en_power_act,
                                                               pars.cost_bias_in,
                                                               pars.cost_bias_w,
                                                               pars.gamma_soft_sign,
                                                               pars.gamma_sign_reg)
                                                               
    ind_optimizer = 0 if pars.optimizer == "SGD" else 1
    to_allresults_file += "%d \t %d \t %d \t %g \t %d \t %d \t" % (pars.sign_reg_quadratic * 1,
                                                               pars.out_sign_reg * 1,
                                                               ind_optimizer,
                                                               pars.lr,
                                                               pars.seed_data,
                                                               pars.seed_net)

    to_allresults_file += "%g \t %g \t %g \t %g \t %g \t %g" % (results['best_train'],
                                                                results['best_test'],
                                                                results['best_active_train'],
                                                                results['best_energy_train'],
                                                                results['best_active_test'],
                                                                results['best_energy_test'])
    
    print(f"writing to {pars.allresults_filename}")
    
    with open(pars.allresults_filename, "a", buffering=1) as allresults_file:    
        log(to_allresults_file, allresults_file, print_to_out=False)
        

def extract_tensors(dataset):
    if hasattr(dataset, "data"):
        return dataset.data, dataset.targets
    else:
        return dataset.tensors
    

def check_equal(a1, a2, name, atol = 1e-8):
    if not np.allclose(a1, a2, atol=atol):
        raise ValueError(f"WFT {name}!")
        
    
def array_nan_equal(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.array_equal(a[m], b[m])


def sigmoid(h):
    return 0.5 * (1. + np.tanh(h))


def argmin_mat(matrix, irange, jrange = None, verbose = False):
    jrange = irange if jrange is None else jrange
    Ni = irange.shape[0]
    Nj = jrange.shape[0]
    ii = np.argmin(matrix)
    if verbose:
        print(ii, ii//Nj, ii%Nj, Ni, Nj)
    return irange[ii//Nj], jrange[ii%Nj], matrix[ii//Nj, ii%Nj]


def argmin_array(array, xrange):
    lin_ind = np.nanargmin(array)
    ind = np.unravel_index(lin_ind, array.shape)
    xmin = xrange[np.array(ind)]
    arraymin = array[ind]
    return ind, xmin, arraymin


def argmax_span_eb(eb1, eb2, Jrange, alpha, maximize=True):
    if maximize:
        lin_ind = np.nanargmax(eb1 - alpha * eb2)
    else:
        lin_ind = np.nanargmin(eb1 + alpha * eb2)
    ind = np.unravel_index(lin_ind, eb1.shape)
    Jmax = Jrange[np.array(ind)]
    eb1max = eb1[ind]
    eb2max = eb2[ind]
    return Jmax, eb1max, eb2max


# WARNING: NOT ALL DERIVATIVES HAVE BEEN IMPLEMENTED YET
activations_and_derivatives = {
    "sigmoid": (torch.sigmoid,
                lambda x: x),
    "tanh": (torch.tanh,
             lambda x: 1. / torch.cosh(x)**2),
    "erf": lambda x: (torch.erf(x / sq2),
                      lambda x: x),
    "erf+": lambda x: (0.5 * (1 + torch.erf(x / sq2)),
                             lambda x: x),
    "relu": (F.relu,
             lambda x: 1. * (x > 0)),
    "relu2": (lambda x: F.relu(x)**2,
             lambda x: 2 * F.relu(x) * x),
    "linear": (lambda x: x,
               lambda x: torch.ones_like(x)),
    "softplus": (lambda x: 1/5 * torch.log(1. + torch.exp(5 * x)),
                 lambda x: x)
}

def get_act_dact(act_type):
    try:
        return activations_and_derivatives[act_type]
    except:
        raise ValueError('Unknow activation function')

        
activations = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "erf": lambda x: torch.erf(x / sq2),
    "erf+": lambda x: 0.5 * (1 + torch.erf(x / sq2)),
    "relu": F.relu,
    "linear": lambda x: x,
}


def get_act(act_type):
    return activations[act_type]


activations_np = {
    "sigmoid": expit,
    "tanh": np.tanh,
    "erf": lambda x: erf(x / sq2),
    "erf+": lambda x: 0.5 * (1 + erf(x / sq2)),
    "relu": lambda x: (x > 0) * x,
    "linear": lambda x: x,
}


def get_act_np(act_type):
    return activations_np[act_type]


def minmax(tensor, tmin, tmax):
    return torch.min(torch.max(tensor.data, tmin, out=tensor.data), tmax, out=tensor.data)


def to_device_ifnotnone(tensor, device):
    return tensor.to(device) if tensor is not None else None


def kurtosis(weights, dim=-1, excess=True):
    """
    Parameters:
    -----------
    weights (...) : tensor
    dim: dimension over which the kurtosis should be measured, default=-1
    excess : if True, compute the excess kurtosis. Default is True.
    """
    D = weights.shape[dim]
    kurt = D * weights.pow(4).sum(dim=dim) / weights.pow(2).sum(dim=dim)**2
    if excess:
        kurt -= 3
    return kurtosis


class BatchGenerator:
    def __init__(self, D, samplers, batch_size):
        self.D = D
        self.samplers = samplers
        self.batch_size = batch_size
        self.M = len(samplers)

    def sample(self, P=1):
        ys = torch.randint(self.M, (self.batch_size,))
        xs = torch.zeros((self.batch_size, self.D))
        for iy, y in enumerate(ys):
            xs[iy] = self.samplers[y].sample(P=P)[0]
        return xs, ys


def get_dict(net, to_numpy=False):
    if to_numpy:
        return OrderedDict(
            {k: v.detach().clone().to("cpu").numpy() for k, v in net.state_dict().items()}
        )
    else:
        return OrderedDict(
            {k: v.detach().clone().to("cpu") for k, v in net.state_dict().items()}
        )


def get1hot(ys, num_classes):
    """
    Transform an array with class labels into an array with one-hot encodings of
    these classes.
    """
    ys1hot = ys.unsqueeze(-1) == torch.arange(num_classes).reshape(1, num_classes)
    return ys1hot.float()


def getCorrelationLength(Ts):
    """
    Returns the correlation length of an Ising model at the given temperature.

    The correlation length xi is defined such that the covariance between spins is
    given by

        E x_i x_j = exp(−|i−j|/xi).

    Parameters:
    -----------
    Ts : the temperature(s), either as scalar, numpy array or pyTorch tensor.
    """
    # Turn Ts into a tensor
    if type(Ts) in [float, int]:
        Ts = torch.tensor([Ts])
    elif isinstance(Ts, np.ndarray):
        Ts = torch.from_numpy(Ts)
    elif isinstance(Ts, list):
        Ts = torch.tensor(Ts)

    return -1 / torch.log(torch.tanh(1.0 / Ts))


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def is_singular(A):
    return np.linalg.matrix_rank(A) < len(A)


def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real


def chebfft(v, x):
    N = len(v) - 1
    if N == 0:
        return 0
    ii = np.arange(0, N)
    iir = np.arange(1 - N, 0)
    iii = np.array(ii, dtype=int)
    V = np.hstack((v, v[N - 1 : 0 : -1]))
    U = np.real(fft(V))
    W = np.real(ifft(1j * np.hstack((ii, [0.0], iir)) * U))
    w = np.zeros(N + 1)
    w[1:N] = -W[1:N] / np.sqrt(1 - x[1:N] ** 2)
    w[0] = sum(iii ** 2 * U[iii]) / N + 0.5 * N * U[N]
    w[N] = (
        sum((-1) ** (iii + 1) * ii ** 2 * U[iii]) / N + 0.5 * (-1) ** (N + 1) * N * U[N]
    )
    return w


def log(msg, logfile, print_to_out=True):
    """
    Print log message to  stdout and the given logfile.
    """
    logfile.write(msg + "\n")

    if print_to_out:
        print(msg)


def roll_batch_(data, vec=None):
    if vec is not None:
        vec = np.array(vec)
        shifts = np.repeat(vec[None], len(data), axis=0)
    else:
        shifts = np.random.randint(data.shape[-1], size=[len(data), 2])
    for i in range(len(data)):
        data[i] = torch.roll(data[i], tuple(shifts[i]), (0, 1))


def roll_batch(data, vec=None):
    data_rolled = data.clone()
    roll_batch_(data_rolled, vec)
    return data_rolled


def roll_dataset(X, y, dim = 1):

    delta_transl = 1
    max_transl = X.shape[-1]
    transl_x = range(0, max_transl, delta_transl)
    if dim == 2:
        transl_y = range(0, max_transl, delta_transl)
    else:
        transl_y = [0]

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=len(X),
                                         shuffle=False,
                                         num_workers=0,
                                         pin_memory=True)

    X_rolled, y_rolled = [], []
    for batch_idx, (data, target) in enumerate(loader):
        for ix, tx in enumerate(transl_x):
            for iy, ty in enumerate(transl_y):
                if dim == 2:
                    batch_rolled = torch.roll(data.squeeze(), (tx,ty), dims=(-2,-1))
                else:
                    batch_rolled = torch.roll(data.squeeze(), tx, dims=(-1))
                X_rolled.append(batch_rolled)
                y_rolled.append(target)

    X_rolled = torch.cat(X_rolled)
    y_rolled = torch.cat(y_rolled)

    perm = np.random.permutation(len(X_rolled))
    X_rolled = X_rolled[perm]
    y_rolled = y_rolled[perm]

    return X_rolled, y_rolled


def IPR(w):
    return ((w**2).sum(-1))**2 / (w**4).sum(-1)


def get_width_spline(l, D):
    spline = UnivariateSpline(np.arange(D), l - np.max(l)/2, s=0)
    r1, r2 = spline.roots()
    return np.abs(r2 - r1)


def circular_mean_std(l, D):
    ps = np.exp(1j * 2 * np.pi * np.arange(D)/D)
    m1 = (l * ps).sum()
    anglem = np.angle(m1)
#     R = np.abs(m1)
#     return anglem / (2 * np.pi) * D, np.sqrt(np.log(1/R**2))
    return anglem / (2 * np.pi) * D, None


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n