import torch
import torch.nn as nn
import numpy as np

from pyhessian import hessian


from itertools import product
import scipy.optimize as opt
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import pyvista as pv


def get_params(model):
    parameters = []
    for param in model.parameters():
        parameters.append(param.clone().detach().flatten())
    return parameters


def set_params(model, p):
    for mpar, vpar in zip(model.parameters(), p):
        mpar.data = nn.Parameter(vpar.reshape(mpar.shape))


class ParamList():
    '''
    Additional utility class for handling operations on lists of torch Tensors.
    '''
    def __init__(self, parameters):
        self.params = parameters
        
    def to(self, device):
        self.params = [p.to(device) for p in self.params]
        return self

    def __add__(self, other):
        new_params = [p1 + p2 for p1, p2 in zip(self.params, other.params)]
        return ParamList(new_params)
    
    def __sub__(self, other):
        new_params = [p1 - p2 for p1, p2 in zip(self.params, other.params)]
        return ParamList(new_params)
    
    def __mul__(self, k):
        new_params = [p * k for p in self.params]
        return ParamList(new_params)
    
    def __truediv__(self, k):
        new_params = [p / k for p in self.params]
        return ParamList(new_params)

    def __getitem__(self, i):
        return self.params[i]


class LossLandscapePlotting():
    def __init__(
            self, 
            model,
            device,
            criterion, 
            data, 
            parameters_history, 
            loss_history=None,
            theta0=None,
            mean_theta0=False
            ):
        
        # Model for which we want to make loss landscape plot, and
        # corresponding loss function with certain data.
        self.model = model
        self.device = device
        self.criterion = criterion
        self.data = data

        # History of model's parameters and losses during training.
        # This needed for making training trace on landscape.
        self.parameters_history = parameters_history
        self.loss_history = loss_history

        # 'Basic' parameters, used for calculation of eigenvecs
        # of model's Hessian, and as center point for plotting.
        # Usually - model parameters after training.
        if mean_theta0:
            self.theta0 = parameters_history[0]
            for i in range(1, len(parameters_history)):
                self.theta0 = self.theta0 + parameters_history[i]
            self.theta0 = self.theta0 / len(parameters_history)
        else:
            self.theta0 = parameters_history[-1] if theta0 is None else theta0

        # Note that eigenvectors may differ from each iteration of 
        # landscape plotting, and this behavior is not understood
        # yet. Maybe this happens because of random number generator
        # (but manual seed didn't help), maybe because of different
        # directions of eigenvectors (but this isn't core of the 
        # problem, as was seen after setting only positive directions), 
        # and maybe because of some floating-point errors, but this 
        # isn't seems as core of problem either. Fortunately, there 
        # only finite number of different plots (4-5), so recalculating
        # the plot may help (if it doesn't took too much time). Todo.
        self.eigvals, self.eigvecs = self.get_eigvecs()

    def get_eigvecs(self):
        '''
        Computes eigenvectors of Hessian for model at certain point, provided as 
        self.theta0, and returns top n(=2) eigenvalues and eigenvectors.
        '''

        print('Computing eigenvectors at providen data...')
        hessian_comp = hessian(self.model.to('cpu'), self.criterion, data=(self.data[0].to('cpu'), self.data[1].to('cpu')), cuda=False)
        eigval, eigvec = hessian_comp.eigenvalues(maxIter=100, top_n=2)
        
        eigvecs = []
        for vec in eigvec:
            pars = []
            for param in vec:
                pars.append(param.clone().detach().flatten().float().to(self.device))
            eigvecs.append(ParamList(pars))
            
        self.model = self.model.to(self.device)
        return eigval, eigvecs

    def L(self, x, loss, theta1):
        '''
        Objective loss function to minimize for training trace approximation.
        '''
        
        newvec = (self.theta0 + self.eigvecs[0] * x[0] + self.eigvecs[1] * x[1])
        par_diff = (theta1 - newvec).params
        
        new_parameters = (self.theta0 + self.eigvecs[0] * x[0] + self.eigvecs[1] * x[1]).params
        set_params(self.model, new_parameters)

        out = self.model(self.data[0])
        loss_diff = (self.criterion(out, self.data[1]).item() - loss)**2

        S = 0
        for param in par_diff:
            S += torch.norm(param)
        S += loss_diff
        return S.item()
    
    def compute_trace(self, every_ith=1, method='nelder-mead'):
        '''
        Function to approximate alpha and beta (guess[0] and guess[1] respectively)
        for every point in parameters history.
        '''
        
        print('Computing trace...')
        trace = []

        parameters = list(self.parameters_history[::every_ith])
        if self.parameters_history[-1] != parameters[-1]:
            parameters.append(self.parameters_history[-1])
        losses = list(self.loss_history[::every_ith])
        if self.loss_history[-1] != losses[-1]:
            losses.append(self.loss_history[-1])
            
        # print(parameters, losses)
            
        self.data = [self.data[0].to(self.device), self.data[1].to(self.device)]

        total_iters = len(parameters)
        for theta1, loss in tqdm(zip(parameters, losses), total=total_iters - 1):
            guess = opt.minimize(self.L, (0, 0), args=(loss, theta1.to(self.device)), method=method).x

            new_par = (self.theta0 + self.eigvecs[0] * guess[0] + self.eigvecs[1] * guess[1]).params

            set_params(self.model, new_par)
            
            out = self.model(self.data[0])
            loss = self.criterion(out, self.data[1])

            trace.append([guess[0], guess[1], loss.item()])
        set_params(self.model, self.theta0.params)
        
        trace = np.array(trace)
        return trace
    
    def compute_landscape(self, trace, grid_density=50, coef=1, arange=None, brange=None, make_equal=True):

        if arange is None:
            alpha_min, alpha_max = trace[:, 0].min(), trace[:,0].max()
            beta_min, beta_max = trace[:, 1].min(), trace[:,1].max()
            a_d = np.abs(alpha_max - alpha_min) * coef
            b_d = np.abs(beta_max - beta_min) * coef

            amin, amax = alpha_min - a_d, alpha_max + a_d
            bmin, bmax = beta_min - b_d, beta_max + b_d

            if make_equal:
                amax = bmax = max(amax, bmax)
                amin = bmin = min(amin, bmin)

        else:
            amin, amax = arange
            bmin, bmax = brange

        ralpha = np.linspace(amin, amax, grid_density)
        rbeta = np.linspace(bmin, bmax, grid_density)
        
        surface = []
        self.data = [self.data[0].to(self.device), self.data[1].to(self.device)]

        print('Making surface...')
        for alpha, beta in tqdm(product(ralpha, rbeta), total=grid_density*grid_density):
            # f(a, b) = L(theta + a * eig_0 + b * eig_0)
            new_params = (self.theta0 + self.eigvecs[0] * alpha + self.eigvecs[1] * beta).params
            set_params(self.model, new_params)
            
            out = self.model(self.data[0])
            loss = self.criterion(out, self.data[1])
            
            surface.append(loss.item())
        set_params(self.model, self.theta0.params)

        surface = np.array(surface).reshape((grid_density, grid_density)).T
        
        if surface.max() == surface.min():
            print('ERROR: Invalid calculations (surface is completely flat)')
            return 0, 0, 0
        
        return ralpha, rbeta, surface
    
    def plot(self, trace, ralpha, rbeta, surface, colormap='cividis', k=1):
        
        # new_trace = []
        # for point in trace:
        #     alphaidx = np.abs(point[0] - ralpha).argmin()
        #     betaidx = np.abs(point[1] - rbeta).argmin()
        #
        #     new_trace.append([
        #         ralpha[alphaidx],
        #         rbeta[betaidx],
        #         surface[betaidx, alphaidx]
        #         ])
        new_trace = trace # np.array(new_trace)

        pl = pv.Plotter()
        
        x, y = np.meshgrid(ralpha, rbeta)
        
        training_trace = pv.PolyData(new_trace)
        pl.add_mesh(training_trace,
                    color='#faedcd',
                    point_size=15.0,
                    render_points_as_spheres=True)
        
        grid = pv.StructuredGrid(x, y, surface)
        pl.add_mesh(grid,
                    scalars=surface.T,
                    cmap=plt.cm.get_cmap(colormap),
                    lighting=True)
        
        coef = np.abs(rbeta.max() - rbeta.min()) / np.abs(surface.max() - surface.min()) * k
        pl.set_scale(zscale=coef)
        
        pl.show_grid()
        pl.show()

    def plot_contour(self, ax_in, trace, ralpha, rbeta, surface, colormap='cividis', k=1, label="loss landscape"):
        x, y = np.meshgrid(ralpha, rbeta)

        coef = np.abs(rbeta.max() - rbeta.min()) / np.abs(surface.max() - surface.min()) * k

        # fig, ax = plt.subplots()
        CS = ax_in.contour(
            x,
            y,
            surface,
            cmap=colormap,
            linewidths=0.75)
        ax_in.clabel(CS, inline=True, fontsize=10, fmt="%1.2f")
        normalize = mpl.colors.Normalize(vmin=-coef, vmax=coef)
        ax_in.scatter(trace[:, 0], trace[:, 1], c=trace[:, 2], s=4, cmap="cividis", marker='o', norm=normalize)
        ax_in.plot(trace[:, 0], trace[:, 1])
        ax_in.set_title(label)

        ax_in.grid()
        # plt.savefig('filename.svg')
        return ax_in
