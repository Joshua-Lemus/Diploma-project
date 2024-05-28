# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np
import pickle
import itertools
from scipy.stats import expon, multivariate_normal

import torch
import torchvision
import torchvision.transforms as transforms

import inputs
from myutils import get_act_np


##############################################################################
###################### GENERAL FUNCTION TO GET DATASET #######################
##############################################################################

def get_data(pars, dtype = torch.float, device = torch.device("cpu")):
    
    # extract params
    dataset = pars.dataset
    alpha_train = pars.alpha_train
    alpha_test = pars.alpha_test
    span_h = pars.span_h
    exp_input = pars.exp_input
    rho = pars.rho
    dim = pars.dim
    lN = pars.lN
    torus = pars.torus
    g_nonlin = pars.g_nonlin
    xis = pars.xis
    pars_teacher = pars.pars_teacher
    normalize = pars.normalize
    data_dir = pars.data_dir
    seed_data = pars.seed_data
    
    np.random.seed(seed_data);
    torch.manual_seed(seed_data);

    if "RANDOM" in dataset:
        outs = get_data_random(dataset, alpha_train, alpha_test,
                            span_h=span_h,
                            exp_input=exp_input,
                            rho=rho,
                            dim=dim,
                            lN=lN,
                            pars_teacher=pars_teacher,
                            normalize=normalize,
                            data_dir=data_dir,
                            dtype=dtype,
                            device=device)
        
    elif "GP" in dataset:
        outs = get_data_nlgp(dataset, alpha_train, alpha_test,
                             dim=dim,
                             lN=lN,
                             torus=torus,
                             g_nonlin=g_nonlin,
                             xis=xis,
                             dtype=dtype,
                             device=device)
        outs = *outs, None, None # these two are teacher_weights and x_teacher
        
    elif dataset == "MNIST1D":
        outs = get_data_mnist1d(data_dir=data_dir)
        outs = *outs, None, None
        
    else:
        outs = get_data_torch(dataset, normalize, data_dir=data_dir)
        outs = *outs, None, None
        
    ## setting data parameters
    train_dataset, test_dataset, num_train, num_test, num_label, dim, lN, N, teacher_weights, x_teacher = outs
    pars.num_train = num_train
    pars.num_test = num_test
    pars.num_label = num_label
    pars.dim = dim
    pars.lN = lN
    pars.N = N
    pars.reshape_data = dim == 2

    return train_dataset, test_dataset, teacher_weights, x_teacher


################################################################################
################################## DATASETS ####################################
################################################################################

# TO DIRECTLY DOWNLOAD MNIST1D USE THE FOLLOWING:
# import requests
# url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
# r = requests.get(url, allow_redirects=True)
# open('data/mnist1d_data.pkl', 'wb').write(r.content)

datasets = ["MNIST", "MNIST10", "FashionMNIST", "CIFAR10", "CIFAR100"]


datasets_torch = {"MNIST"        : torchvision.datasets.MNIST,
                  "MNIST10"      : torchvision.datasets.MNIST,
                  "FashionMNIST" : torchvision.datasets.FashionMNIST,
                  "CIFAR10"      : torchvision.datasets.CIFAR10,
                  "CIFAR100"     : torchvision.datasets.CIFAR100
                 }

meanstds = {"MNIST"        : ((0.1307,), (0.3081,)),
            "MNIST10"      : ((0.1307,), (0.2873,)),
            "FashionMNIST" : ((0.2860,), (0.3530,)),
            "CIFAR10"      : ((0.4734,), (0.2393,)),
            "CIFAR100"     : ((0.4782,), (0.2499,)),
           }

grayscales = {"MNIST"        : False,
              "MNIST10"      : False,
              "FashionMNIST" : False,
              "CIFAR10"      : True,
              "CIFAR100"     : True,
           }

resizes = {"MNIST"        : (False, None),
           "MNIST10"      : (True, 10),
           "FashionMNIST" : (False, None),
           "CIFAR10"      : (False, None),
           "CIFAR100"     : (False, None),
           }


def get_data_random(dataset, alpha_train, alpha_test,
                    span_h = False,
                    exp_input = False,
                    rho = 0.,
                    dim = 1,
                    lN = 10,
                    pars_teacher = None,
                    normalize = True,
                    data_dir = "data",
                    dtype = torch.float,
                    device = torch.device("cpu")):

    ## set data properties
    N = lN**dim
    num_train = int(alpha_train * N)
    num_test = int(alpha_test * N)

    ## get input dataset
    if span_h: # whole hs range
        hrange = np.arange(-3, 3.2, 0.2)
        if N > 3:
            raise ValueError(f"Are you crazy?! You're asking me to generate {len(hrange)**N} patterns!")
        hs = np.zeros((len(hrange)**N, N))
        for ie, e in enumerate(itertools.product(*[list(hrange) for i in range(N)])):
            hs[ie] = e
    else:
        if exp_input: # correlated exp
            hs = expon.rvs(scale=1, size=(num_train+num_test, N))
            if rho > 0:
                z = expon.rvs(scale=1, size=(num_train+num_test, 1))
                hs = rho * z + (1. - rho) * hs                
        else: # correlated gaussian
            Ch = rho * np.ones((N, N))
            Ch[np.diag_indices(N)] = 1.
            hs = multivariate_normal.rvs(mean=np.zeros(N), cov=Ch, size=num_train+num_test)
            
    # small fix in case num_train+num_test = 1
    if len(hs.shape) == 1:
        hs = hs[None]
    
    ## get output dataset
    if "Teacher" in dataset: # use teacher-generated output
        y, teacher_weights, x_teacher = get_data_teacher(dataset, hs, N, pars_teacher)
        if "-Y" in dataset:
            num_label = pars_teacher["num_out"]
        elif "-X" in dataset:
            num_label = N
        else:
            num_label = 2
    else:
        num_label = 2
        teacher_weights, x_teacher = None, None
        num_test = None
        if "-Y" in dataset: # random uniformly distributed output
            y = np.random.randn(num_train)
        else: # random binary output
            y = np.random.randint(2, size = num_train)

    ## wrap in torch dataset
    X_train = torch.tensor(hs[:num_train], dtype=dtype, device=device)
    y_train = torch.tensor(y[:num_train], dtype=dtype, device=device)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    if num_test is not None:
        X_test = torch.tensor(hs[num_train:], dtype=dtype, device=device)
        y_test = torch.tensor(y[num_train:], dtype=dtype, device=device)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    else:
        X_test, y_test = None, None
        test_dataset = None
        
    return train_dataset, test_dataset, num_train, num_test, num_label, dim, lN, N, teacher_weights, x_teacher


def get_data_teacher(dataset, hs, N, pars_teacher = None):
                     
    K_teacher = pars_teacher["K"]
        
    ## recurrent teacher
    if "RNN" in dataset:
        
        # set main params
        discrete_time = pars_teacher["discrete_time"]
        dt = pars_teacher["dt"]
        
        # gen input weights
        if pars_teacher["eye_in"]:
            if K_teacher != N:
                print(f"K_teacher must be consistent with input dimension! Setting K_teacher = {N}")
                K_teacher = N
            w_in = np.eye(N)
        else:
            w_in = np.random.randn(N, K_teacher) / np.sqrt(N)
        # read or generate recurrent weights
        if pars_teacher["J"] is not None:
            J = pars_teacher["J"]
        else:
            J = pars_teacher["g"] * np.random.randn(K_teacher, K_teacher) / np.sqrt(K_teacher)
        # generate input biases
        bias_in = pars_teacher["g_bias_in"] * np.random.randn(K_teacher)
        # gen output weights
        if pars_teacher["soft_w"]:
            v = np.ones((pars_teacher["num_out"], K_teacher)) / np.sqrt(K_teacher)
        else:
            v = np.random.randn(pars_teacher["num_out"], K_teacher) / np.sqrt(K_teacher)
        act = get_act_np(pars_teacher["nonlinearity"])
        teacher_weights = [w_in, J, v, bias_in]
        
        # converge to fixed points
        tol = 1e-4
        max_iter = 500
        x_teacher = np.zeros((hs.shape[0], K_teacher))
        inp = hs @ w_in
        iterations = 0
        while iterations < max_iter:
            h = x_teacher @ J.T + inp + bias_in
            if discrete_time:
                xnew = act(h)
            else:
                xnew = (1. - dt) * x_teacher + dt * act(h)
            err = np.linalg.norm(xnew - x_teacher)
            x_teacher = xnew
            iterations += 1
            if err < tol:
                break
        # generate outputs
        if "X" in dataset:
            y = x_teacher
        elif "Y" in dataset:
            y = x_teacher @ v.T
        else:
            y = (x_teacher @ v[0]) > 0 # standard is a single binary output
        
    ## feed-forward teacher
    elif "FF" in dataset:
        
        # set network
        act = get_act_np(pars_teacher["nonlinearity"])
        if pars_teacher["J"] is not None:
            J = pars_teacher["J"]
        else:
            J = pars_teacher["g"] * np.random.randn(N, K_teacher) / np.sqrt(N)
        b = pars_teacher["g_bias_in"] * np.random.randn(K_teacher)
        if pars_teacher["soft_w"]:
            v = np.ones(K_teacher) / np.sqrt(K_teacher)
        else:
            v = np.random.randn(K_teacher) / np.sqrt(K_teacher)
        teacher_weights = [J, b, v]
        
        # generate output
        x_teacher = act(hs @ J + b) @ v
        if "Y" in dataset:
            y = x_teacher
        else:
            y = x_teacher > 0
    else:
        raise ValueError("What kind of teacher do you want?")
        
    return y, teacher_weights, x_teacher


def get_data_nlgp(dataset, alpha_train, alpha_test,
                dim = 1,
                lN = 10,
                xis = [1, 4],
                torus = True,
                g_nonlin = 2,
                dtype = torch.float,
                device = torch.device("cpu")):

    # set data properties
    N = lN**dim
    num_train = int(alpha_train * N)
    num_test = int(alpha_test * N)
    num_label = len(xis)
    
    perturbation = 1e-4

    mixture_params = [None] * num_label
    if g_nonlin == 0:
        mixture_params[0] = {"distribution" : "gp",
                             "dim" : dim,
                             "torus" : torus,
                             "exponent" : 2,
                             "xi" : xis[0],
                             "perturbation" : perturbation}
        mixture_params[1] = mixture_params[0].copy()
        mixture_params[1]["xi"] = xis[1]
        task_generator = build_mixture(N, mixture_params)
    elif dataset == "GPNLGP":
        distributions = build_nlgp_mixture(["gp", "nlgp"],
                                           [xis[0], xis[1]],
                                           lN,
                                           torus,
                                           g_nonlin,
                                           dim=dim,
                                           xi_pow_pi=True,
                                           perturbation=perturbation)
        task_generator = Mixture(distributions)
    else:
        input_name = "gp" if dataset == "GP" else "nlgp" 
        input_names = [input_name] * num_label
        distributions = build_nlgp_mixture(input_names,
                                           xis,
                                           lN,
                                           torus,
                                           g_nonlin,
                                           dim,
                                           perturbation=perturbation)
        task_generator = Mixture(distributions)

    X_train, y_train = task_generator.get_dataset(P=num_train, train=True)
    if torch.isnan(X_train).sum().item() > 0:
        raise ValueError("AHI! NANS!!!")
    num_train = len(y_train)
    X_test, y_test = task_generator.get_dataset(P=num_test, train=False)
    num_test = len(y_test)
    num_label = len(y_train.unique())
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    return train_dataset, test_dataset, num_train, num_test, num_label, dim, lN, N


def get_data_mnist1d(data_dir = "data"):
    
    # get dataset
    with open(f"{data_dir}/mnist1d_data.pkl", 'rb') as handle:
        data = pickle.load(handle)
    X_train = torch.tensor(data['x'], dtype=torch.float)
    y_train = torch.tensor(data['y'], dtype=torch.float)
    X_test = torch.tensor(data['x_test'], dtype=torch.float)
    y_test = torch.tensor(data['y_test'], dtype=torch.float)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # set data properties
    normalize = True
    num_train = len(train_dataset.tensors[0])
    num_test = len(test_dataset.tensors[0])
    num_label = 10
    dim = 1
    lN = test_dataset.tensors[0].shape[-1]
    N = lN**dim
    
    return train_dataset, test_dataset, num_train, num_test, num_label, dim, lN, N


def get_data_torch(dataset, normalize = False, data_dir = "data"):
    
    # extract dataset properties
    dataset_torch = datasets_torch[dataset]
    grayscale = grayscales[dataset]
    resize, size = resizes[dataset]
    meanstd = meanstds[dataset]
    
    # set transformations
    transforms_list = [transforms.ToTensor()]
    if grayscale:
        transforms_list += [transforms.Grayscale()]
    if resize:
        transforms_list += [transforms.Resize(size)]
    if normalize:
        transforms_list += [transforms.Normalize(*meanstd)]
    
    # get dataset
    train_dataset = dataset_torch(data_dir,
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose(transforms_list))
    test_dataset = dataset_torch(data_dir,
                                 train=False,
                                 download=True,
                                 transform=transforms.Compose(transforms_list))
    # set data properties
    num_train = len(train_dataset.data)
    num_test = len(test_dataset.data)
    num_label = len(test_dataset.classes)
    dim = 2
    lN = test_dataset.data.shape[1] if not resize else size
    N = lN**dim
    
    return train_dataset, test_dataset, num_train, num_test, num_label, dim, lN, N



#########################################
############# MIXTURE TASKS #############
#########################################

class Task(metaclass=ABCMeta):
    """
    Abstract class for all the tasks used in these experiments.
    """

    @abstractmethod
    def input_dim(self):
        """
        Dimension of the vectors for 1D models, width of inputs for 2D models.
        """

    @abstractmethod
    def get_dataset(
        self, train=True, P=None, dtype=torch.float, device=torch.device("cpu"),
    ):
        """
        Retrieve stored dataset.

        Parameters:
        -----------
        train : True for training data set, else a test data set is loaded.
        P : number of samples
        """

    @abstractmethod
    def sample(self, P):
        """
        Samples P samples from the task.

        Returns:
        --------

        xs : (P, D)
             P inputs in D dimensions
        ys : (P)
             P labels
        """


class Mixture(Task):
    """
    Mixture of distributions, one for each label.
    """

    def __init__(self, distributions):
        """
        Parameters:
        -----------

        distributions: array
            a set of inputs, each of which will correspond to one label.
        """
        super().__init__()

        self.distributions = distributions
        self.num_classes = len(distributions)

        self.D = self.distributions[0].input_dim

    def input_dim(self):
        return self.D

    def get_dataset(
        self, train=True, P=None, dtype=torch.float, device=torch.device("cpu"),
    ):
        num_p = P // self.num_classes
        dataset_desc = "training" if train else "testing"
#         print(f"will generate {dataset_desc} set with {num_p} patterns per class")
        X = torch.empty((0, self.D), dtype=dtype, device=device)
        y = torch.empty(0, dtype=dtype, device=device)
        for p, distribution in enumerate(self.distributions):
            Xtemp = distribution.get_dataset(
                train=train, P=num_p, dtype=dtype, device=device
            )
            X = torch.cat([X, Xtemp])
            y = torch.cat([y, p * torch.ones(len(Xtemp), dtype=dtype, device=device)])

        return X, y

    def sample(self, P=1):
        ys = torch.randint(self.num_classes, (P,))
        xs = torch.zeros(P, self.D)

        for m in range(self.num_classes):
            num_samples = torch.sum(ys == m).item()
            xs[ys == m] = self.distributions[m].sample(num_samples)

        return xs, ys

    def __str__(self):
        dist_names = [str(dist) for dist in self.distributions]
        name = "_".join(dist_names)
        return name


def build_nlgp_mixture(
    input_names, xis, D, torus, gain, dim=1, xi_pow_pi=True, perturbation=1e-3
):
    """
    Constructs a mixture of distributions (NLGP / GP) with the given xis and gain.
    """
    distributions = [None] * len(input_names)

    for idx, input_name in enumerate(input_names):
        # create the covariance for the given correlation length
        xi = xis[idx]

        covariance = inputs.trans_inv_var(
            D,
            torus=torus,
            p=2,
            xi=xi,
            perturbation=perturbation,
            dim=dim,
            xi_pow_pi=xi_pow_pi,
        )
        # create the non-linear GP
        nlgp = inputs.NLGP("erf", covariance, gain=gain)
        if input_name == "gp":
            # create a Gaussian process with the same covariance
            distributions[idx] = inputs.GaussianProcess(nlgp.covariance())
        elif input_name == "nlgp":
            distributions[idx] = nlgp
        else:
            raise ValueError("Did not recognise input name (gp | nlgp)")

    return distributions


def build_mixture(D, params):
    """
    Factory method to create mixtures of distributions.

    Parameters:
    -----------
    D : input dimension
    param:
        params ...
    distribution : gp | ising | phi4
    """
    num_distributions = len(params)

    distributions = [None] * num_distributions
    for m, par in enumerate(params):
        distribution = par["distribution"]
        if distribution == "gp":
            dim = par["dim"]
            torus = par["torus"]
            exponent = par["exponent"]
            perturbation = par["perturbation"]
            xi = par["xi"]

            covariance = inputs.trans_inv_var(
                D, torus=torus, p=exponent, xi=xi, perturbation=perturbation, dim=dim
            )

            distributions[m] = inputs.GaussianProcess(covariance)
        elif distribution == "ising":
            dim = par["dim"]
            T = par["T"]
            num_steps_eq = par["num_steps_eq"]
            sampling_rate = par["sampling_rate"]
            load_dir = par["load_dir"]
            distributions[m] = inputs.Ising(
                dim=dim,
                N=D,
                T=T,
                num_steps_eq=num_steps_eq,
                sampling_rate=sampling_rate,
                load_dir=load_dir,
            )
        elif distribution == "phi4":
            dim = (par["dim"],)
            lambd = par["lambd"]
            musq = par["musq"]
            zscore = par["zscore"]
            normalize = par["normalize"]
            sampling_rate = par["sampling_rate"]
            buffer_size = par["buffer_size"]
            load_dir = par["load_dir"]
            distributions[m] = inputs.Phi4(
                dim=dim,
                D=D,
                lambd=lambd,
                musq=musq,
                zscore=zscore,
                normalize=normalize,
                sampling_rate=sampling_rate,
                buffer_size=buffer_size,
                load_dir=load_dir,
            )
        else:
            raise ValueError("What the fuck are you talking about?!")

    mixture = Mixture(distributions)

    return mixture


class MNIST(Task):
    def __init__(
        self, task, load_dir="data/MNIST_ROLLED",
    ):

        # override topological options
        self.dim = 2
        self.load_dir = load_dir
        self.task = task

        if task == "evenodd" or task == "passionforfashion" or isinstance(task, list):
            self.M = 2
        else:
            self.M = 10

    def input_dim(self):
        return self.D ** 2

    def get_dataset(
        self, train=True, P=None, dtype=torch.float, device=torch.device("cpu")
    ):
        dataset_type = "training" if train else "test"
        filename = f"{self.load_dir}/{dataset_type}.pt"
        print(f"...will read {dataset_type} data from {filename}")
        X, y = torch.load(filename)
        if P is not None:
            if P > len(X):
                raise ValueError("Not enough data in the stored dataset")
            X = X[:P]
            y = y[:P]
        print(f"...{len(X)} patterns read from file")

        self.D = X.shape[-1]
        self.lens = [self.D, self.D]

        # reduce MNIST
        if isinstance(self.task, list):
            X_temp = torch.empty((0,*X.shape[1:]))
            y_temp = []

            for il, label in enumerate(self.task):
                XX = X[y==label]
                X_temp = torch.vstack([X_temp, XX])
                y_temp += [il] * XX.shape[0]
            X = X_temp
            y = torch.tensor(y_temp, dtype=torch.long)
            print(f"...filtered classes {self.task}")

        # even vs odd
        if self.task == "evenodd":
            is_even = (y == 0) + (y == 2) + (y == 4) + (y == 6) + (y == 8)
            is_odd = (y == 1) + (y == 3) + (y == 5) + (y == 7) + (y == 9)
            y[is_even] = 0
            y[is_odd] = 1
            print("...filtered even vs odd")

        # passion for fashion
        if self.task == "passionforfashion":
#             0: magliette
#             1: pantaloni
#             2: felpe
#             3: vestiti lunghi da donna
#             4: felpe piu lunghe
#             5: sandali
#             6: blazer (simili a 2 e 3 e 4)
#             7: scarpe chiuse
#             8: borse (simili a 7)
#             9: stivali
            is_passion = (y == 0) + (y == 1) + (y == 5) + (y == 6) + (y == 8)
            is_fashion = (y == 2) + (y == 3) + (y == 4) + (y == 7) + (y == 9)
            y[is_passion] = 0
            y[is_fashion] = 1
            print("...filtered passion for fashion")

        return X, y

    def sample(self, P=None):
        print(
            """
            You want to use a DataLoader for this.
            I could very well sample from MNIST
            but won't as a matter of principle."""
        )
