# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

'''
Function Library for anabatic_flow_main ver 6
----------------------------------------
Author: Andrew Loeppky
Course: ATSC 409, Apr2/20
Professor: Susan Allan, Phil Austin
''';

'''
NEW THIS VERSION

''';

# +
#usual imports
import context;
import numpy as np

#plotters/display
from IPython.display import Image
import IPython.display as display
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns

#for loading ICs/BCs
import yaml
from collections import namedtuple


# -
class Integrator():
    '''
    class that performs the integration of eq (1) and (2) referenced in anabatic flow paper. 
    Largely copied from ATSC 409 daisyworld code
    '''
    
    def __init__(self, coeffFileName):
        with open(coeffFileName, 'rb') as f:
            config=yaml.load(f, Loader=yaml.FullLoader)
        self.config = config
        
        # read in dt tstart tend
        timevars = namedtuple('timevars',config['timevars'].keys())
        self.timevars = timevars(**config['timevars'])
        
        # read in grid size dn nstart nend
        nvars = namedtuple('nvars',config['nvars'].keys())
        self.nvars = nvars(**config['nvars'])
        
        # read in user specified parameters
        # g Km Kh theta_0 gamma beta
        constants = namedtuple('constants',config['constants'].keys())
        self.constants = constants(**config['constants'])
        
        # read in ICs and BCs
        boundaryconds = namedtuple('boundaryconds',config['boundaryconds'].keys())
        self.boundaryconds = boundaryconds(**config['boundaryconds'])
        
        initialconds = namedtuple('initialconds',config['initialconds'].keys())
        self.initialconds = initialconds(**config['initialconds'])
     
    
    def __str__(self):
        out = 'integrator instance with attributes timevars, constants'
        return out

    
    def gen_grids(self):   
        '''make a grid t x n
        '''
        timevars = self.timevars
        nvars = self.nvars
        
        tsteps = int((timevars.tend - timevars.tstart) / timevars.dt)
        nsteps = int((nvars.nend - nvars.nstart) / nvars.dn)
        u_grid = np.ones([tsteps, nsteps + 1])
        theta_grid = np.ones([tsteps, nsteps  + 1])
        
        return u_grid, theta_grid

    
    def apply_bc(self, u_grid, theta_grid, tstep):
        '''ver 1, enforce a constant temperature anomaly and velocity
        at surface, zero anomaly/vel at highest cell (crudely representing n_{\infty}),          
        '''
        boundaryconds = self.boundaryconds
        
        u_grid[tstep][0] = boundaryconds.u_surf
        u_grid[tstep][-1] = boundaryconds.u_sky
        theta_grid[tstep][0] = boundaryconds.theta_surf
        theta_grid[tstep][-1] = boundaryconds.theta_sky

        
    def deriv_test(self, u_old, theta_old):
        '''
        Dummy function that adds 1 each timestep 
        '''
        consts = self.constants
        timevars = self.timevars
        nvar = self.nvars
        
        u_new = np.empty_like(u_old)
        theta_new = np.empty_like(theta_old)
        
        for n in range(len(theta_old)):
            u_new[n] = u_old[n] + 1
            theta_new[n] = theta_old[n] + 1
            
        return u_new, theta_new
    
    
    def deriv_dfsn_only(self, u_old, u_old2, theta_old, theta_old2):
        '''
        solve uncoupled diffusion terms in (1) and (2) to assess numerical stability
        of DFF or FTCS
        '''
        #define relevant constants
        dt = self.timevars.dt                  #s
        dn = self.nvars.dn                     #m
        g = self.constants.g                   #m/s2
        Km = self.constants.Km                 #m2/s
        Kh = self.constants.Kh                 #m2/s
        
        #empty vectors to hold u_new, theta_new
        u_new = np.empty_like(u_old)
        theta_new = np.empty_like(theta_old)
        
        #compute derivative 
        for n in np.arange(1,len(u_new) -1, 1):
            #FTCS (7.9) -- stable for dt <= dn**2 / 2Km
            u_new[n] = u_old[n] + (dt*Km/dn**2) * (u_old[n+1] - 2*u_old[n] + u_old[n-1])
            theta_new[n] = theta_old[n] + (dt*Kh/dn**2) * (theta_old[n+1] - 2*theta_old[n] + theta_old[n-1])
                        
            #DFF (7.12)
            #u_new[n] = ((1-C)/(1+C))*u_old2[n] + (C/(1+C))*(u_old[n+1] + u_old[n-1])
            #theta_new[n] = ((1-D)/(1+D))*theta_old2[n] + (D/(1+D))*(theta_old[n+1] + theta_old[n-1])
        
        return u_new, theta_new
 

    def deriv_FTCSv1(self, u_old, theta_old):
        '''
        Calculates t(j+1)th element of (1) and (2) with the forward time centered space method
        
        **hacky version not mathematically verified!**
        
        uses convention _old, _new for j and j+1 time elements 
        '''
        #define constants
        dt = self.timevars.dt                  #s
        dn = self.nvars.dn                     #m
        g = self.constants.g                   #m/s2
        Km = self.constants.Km                 #m2/s
        Kh = self.constants.Kh                 #m2/s
        theta_0 = self.constants.theta_0       #K
        theta_surf = self.constants.theta_surf #K
        gamma = self.constants.gamma           #K/m
        beta = self.constants.beta             #degrees
        
        N2 = gamma * g / theta_0               #s^-2 (brunt-vaisala freq^2)
        
        #empty vectors to hold u_new, theta_new
        u_new = np.empty_like(u_old)
        theta_new = np.empty_like(theta_old)
        
        #compute derivatives FTCS
        for n in np.arange(1,len(u_new) -1, 1):
            #FTCS (7.9) -- stable for dt <= dn**2 / 2Km
            u_new[n] = 2*10**(-2)*theta_old[n] + u_old[n] + (dt*Km/dn**2) * (u_old[n+1] - 2*u_old[n] + u_old[n-1])
            theta_new[n] = 2*10**(-2)*u_old[n] + theta_old[n] + (dt*Kh/dn**2) * (theta_old[n+1] - 2*theta_old[n] + theta_old[n-1])
                  
        return u_new, theta_new
 

    def deriv_FTCS(self, u_old, theta_old):
        '''
        Calculates t(j+1)th element of (1) and (2) with the forward time centered space method
        
        uses convention _old, _new for j and j+1 time elements 
        '''
        #define constants
        dt = self.timevars.dt                  #s
        dn = self.nvars.dn                     #m
        g = self.constants.g                   #m/s2
        Km = self.constants.Km                 #m2/s
        Kh = self.constants.Kh                 #m2/s
        theta_0 = self.constants.theta_0       #K
        gamma = self.constants.gamma           #K/m
        beta = self.constants.beta             #degrees
        
        #print('magnitude of theta cross term:')
        #print(np.sin((np.pi/180)*beta)*dt*g/theta_0)
        #print('\nmagnitude of u cross term:')
        #print(gamma*dt*np.sin((np.pi/180)*beta))
        
        #empty vectors to hold u_new, theta_new
        u_new = np.empty_like(u_old)
        theta_new = np.empty_like(theta_old)
       
        #compute derivatives DFF
        for n in np.arange(1,len(u_new) -1, 1):
            
            u_new[n] = u_old[n] + theta_old[n]*np.sin((np.pi/180)*beta)*dt*g/theta_0 \
                       + (Km*dt/dn**2)*(u_old[n+1] - 2*u_old[n] + u_old[n-1])
            
            theta_new[n] = theta_old[n] - u_old[n]*gamma*dt*np.sin((np.pi/180)*beta) \
                       + (Kh*dt/dn**2)*(theta_old[n+1] -2*theta_old[n] + theta_old[n-1])
        
        return u_new, theta_new   
            
        
    def solve_eqns(self):
        '''
        Time loop for solving (1) and (2). Choose which derivs function you would like to use
        '''
        timevars = self.timevars
        initialconds = self.initialconds
        time = np.arange(timevars.tstart,timevars.tend,timevars.dt)
        tsteps = int((timevars.tend - timevars.tstart) / timevars.dt)
        
        nvars = self.nvars
        height = np.arange(nvars.nstart,nvars.nend,nvars.dn)
        
        #build the grid and set initial conds
        u_grid, theta_grid = self.gen_grids()
        
        u_grid[0][:] = initialconds.initvel
        u_grid[1][:] = initialconds.initvel

        theta_grid[0][:] = initialconds.init_th_prof 
        theta_grid[1][:] = initialconds.init_th_prof

        #loop through time and apply derivative function
        #use convention new, old, old2 for t(j+1), t(j), t(j-1)
        for the_time in range(2,tsteps,1):            
            
            u_old = u_grid[the_time - 1][:]
            u_old2 = u_grid[the_time - 2][:]
        
            theta_old = theta_grid[the_time - 1][:]
            theta_old2 = theta_grid[the_time - 2][:] 
            
            #Test function, adds 1 each timestep
            #u_grid[the_time][:], theta_grid[the_time][:] = self.deriv_test(u_old, theta_old)
            
            #Diffusion only, no coupling - can be toggled bw FTCS and DFF (see function)
            #u_grid[the_time][:], theta_grid[the_time][:] = self.deriv_dfsn_only(u_old, u_old2, theta_old, theta_old2)
            
            #Forward in Time, Centered in Space 
            u_grid[the_time][:], theta_grid[the_time][:] = self.deriv_FTCS(u_old, theta_old)
            
            #DuFort-Frankel Scheme
            #u_grid[the_time][:], theta_grid[the_time][:] = self.deriv_DFF(u_old, u_old2, theta_old, theta_old2)
                       
            self.apply_bc(u_grid, theta_grid, the_time) 
        
        return u_grid, theta_grid   
    
    def make_plot2(self, u_vel, theta):
        '''
        Plots 100 velocity profiles along t and n axes
        '''
        tskips = u_vel.shape[0] // 10
        nskips = u_vel.shape[1] // 15
        w = np.zeros_like(u_vel)
        u = np.empty_like(u_vel) 
        u[:] = np.nan
        u[::tskips,::nskips] = u_vel[::tskips,::nskips]   #u[t][n] for reference
        print(f'Grid size: {u.shape}')
        
        plt.figure()
        fig, ax = plt.subplots(figsize=(20,5))
        ax = sns.heatmap(theta.T, cmap="coolwarm")
        Q = ax.quiver(u.T,w.T, pivot='tail', units='width', scale=50)

        #make it pretty
        ax.invert_yaxis()
        ax.set_title("the title")
        ax.set_xlabel('time')
        ax.set_ylabel('position')
        
        plt.show(fig)

if __name__ == '__main__':
    anabatic_run = Integrator('anabatic_FTCS.yaml')
    u_grid, theta_grid = anabatic_run.solve_eqns()
    anabatic_run.make_plot2(u_grid, theta_grid)
    
    print(u_grid[-1,10:20])






