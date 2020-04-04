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
Cleaned up and streamlined (still not 'fast'), uncoupled and FTCS, exact modes for Prandtl and Defant solutions
''';

# +
#usual imports
import context;
import numpy as np

#plotters/display
import IPython.display as display
import matplotlib.pyplot as plt
# %matplotlib inline
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
    
    def __init__(self, coeffFileName, mode, forcing='const'):
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
        
        self.mode = mode
        self.forcing = forcing
     
    
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

    
    def apply_bc(self, u_grid, theta_grid, tstep, forcing='const'):
        '''ver 2, set BCs to either correspond to Prandtl problem (constant
        temp anomaly) or Defant (sinusiodal temp anomaly)
        Zero anomaly/vel at highest cell (crudely representing n_{\infty}),          
        '''
        boundaryconds = self.boundaryconds
        timevars = self.timevars
        forcing = self.forcing
        
        omega = 2*(np.pi/((timevars.tend - timevars.tstart)/timevars.dt)) 
        
        u_grid[tstep][0] = boundaryconds.u_surf 
        u_grid[tstep][-1] = boundaryconds.u_sky
        theta_grid[tstep][-1] = boundaryconds.theta_sky

        if forcing == 'diurnal':
            #note diurnal forcing will assume scale tend to be 24h ie oscillate 1 period 
            theta_grid[tstep][0] = boundaryconds.theta_surf * np.sin(omega*tstep) 
        elif forcing == 'const':
            theta_grid[tstep][0] = boundaryconds.theta_surf
        else:
            pass
           
    
    def deriv_dfsn_only(self, u_old, theta_old):
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
        
        #empty vectors to hold u_new, theta_new
        u_new = np.empty_like(u_old)
        theta_new = np.empty_like(theta_old)
       
        #compute derivatives FCTS
        for n in np.arange(1,len(u_new) -1, 1):
            
            u_new[n] = u_old[n] + theta_old[n]*np.sin((np.pi/180)*beta)*dt*g/theta_0 \
                       + (Km*dt/dn**2)*(u_old[n+1] - 2*u_old[n] + u_old[n-1])
            
            theta_new[n] = theta_old[n] - u_old[n]*gamma*dt*np.sin((np.pi/180)*beta) \
                       + (Kh*dt/dn**2)*(theta_old[n+1] -2*theta_old[n] + theta_old[n-1])
        
        return u_new, theta_new   
    
    
    def exact_prandtl(self, u_old, theta_old):
        '''
        Computes the exact solution for constant $\Theta$ surface condition
        Prandtl 1942
        '''
        #define constants
        g = self.constants.g                       #m/s2
        Km = self.constants.Km                     #m2/s
        Kh = self.constants.Kh                     #m2/s
        theta_0 = self.constants.theta_0           #K
        theta_surf = self.boundaryconds.theta_surf #K
        gamma = self.constants.gamma               #K/m
        beta = self.constants.beta                 #degrees
        
        N = (gamma*g/theta_0)**0.5                 #s^-1 Brunt Vaisala Frq 
        L = ((4*Km*Kh) / (N * np.sin((np.pi/180)*beta))**2)**0.25 #m length scale
        
        #empty vectors to hold u_new, theta_new
        u_ext = np.empty_like(u_old)
        theta_ext = np.empty_like(theta_old)
        
        #compute 
        for n in np.arange(1,len(u_ext) -1, 1):
            
            u_ext[n] = theta_surf * N / gamma * (Km/Kh)**0.5 * np.exp(-n/L) * np.sin(n/L)
            theta_ext[n] = theta_surf * np.exp(-n/L) *  np.cos(n/L)
        
        return u_ext, theta_ext
    
    def exact_defant(self, u_old, theta_old, time):
        '''
        Computes the exact solution for $\Theta = sin(\omega t)$
        Defant 1949
        
        default parameters are sin(t/24h + 0) diurnal heating cycle
        '''
        #define constants
        dt = self.timevars.dt                      #s
        tstart = self.timevars.tstart              #s
        tend = self.timevars.tend                  #s
        g = self.constants.g                       #m/s2
        Km = self.constants.Km                     #m2/s
        Kh = self.constants.Kh                     #m2/s
        theta_0 = self.constants.theta_0           #K
        theta_surf = self.boundaryconds.theta_surf #K
        gamma = self.constants.gamma               #K/m
        beta = self.constants.beta                 #degrees
        
        N = (gamma*g/theta_0)**0.5                 #s^-1 Brunt Vaisala Frq 
        L = ((4*Km*Kh) / (N * np.sin((np.pi/180)*beta))**2)**0.25 #m length scale
        
        #empty vectors to hold u_new, theta_new
        u_ext = np.empty_like(u_old)
        theta_ext = np.empty_like(theta_old)
        
        omega = 2*(np.pi/((tend - tstart)/dt))
        
        #compute 
        for n in np.arange(1,len(u_ext) -1, 1):
            
            u_ext[n] = theta_surf * N / gamma * (Km/Kh)**0.5 * np.exp(-n/L) * np.sin(n/L) \
            * np.sin(omega*time)
            
            theta_ext[n] = theta_surf * np.exp(-n/L) *  np.cos(n/L) * np.sin(omega*time)
        
        return u_ext, theta_ext
        
    def solve_eqns(self):
        '''
        Time loop for solving (1) and (2). Choose which derivs function 
        to use when initializing Integrator class
        '''
        mode = self.mode
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
        #use convention new, old for t(j+1), t(j)
        for the_time in range(2,tsteps,1):            
            
            u_old = u_grid[the_time - 1][:]
            theta_old = theta_grid[the_time - 1][:]
            
            
            #Diffusion only, no coupling 
            if mode == 'dfsn':
                u_grid[the_time][:], theta_grid[the_time][:] = \
                self.deriv_dfsn_only(u_old, theta_old)
            
            #Forward in Time, Centered in Space 
            elif mode == 'ftcs':
                u_grid[the_time][:], theta_grid[the_time][:] = self.deriv_FTCS(u_old, theta_old)
            
            #Exact solution, constant surface forcing
            elif mode == 'ext_prandtl':
                u_grid[the_time][:], theta_grid[the_time][:] = self.exact_prandtl(u_old, theta_old)
           
            #Exact solution, sin(wt) surface forcing
            elif mode == 'ext_defant':
                u_grid[the_time][:], theta_grid[the_time][:] = self.exact_defant(u_old, theta_old, the_time)
                
            else:
                raise Exception("no valid integration mode specified in arg 2 of 'Integrator'")
                       
            self.apply_bc(u_grid, theta_grid, the_time) 
        
        return u_grid, theta_grid   

    
    def make_plot(self, u_vel, theta):
        '''
        Plots 100 velocity profiles along t and n axes
        '''
        tskips = u_vel.shape[0] // 7
        nskips = u_vel.shape[1] // 15
        w = np.zeros_like(u_vel)
        u = np.empty_like(u_vel) 
        u[:] = np.nan
        u[::tskips,::nskips] = u_vel[::tskips,::nskips]   #u[t][n] for reference
        print(f'Grid size: {u.shape}')
        
        plt.figure()
        fig, ax = plt.subplots(figsize=(20,5))
        ax = sns.heatmap(theta.T, cmap="coolwarm")
        Q = ax.quiver(u.T,w.T, pivot='tail', units='width', scale=120)

        #make it pretty
        ax.invert_yaxis()
        ax.set_title("the title")
        ax.set_xlabel('time')
        ax.set_ylabel('position')
        
        plt.show(fig)



if __name__ == '__main__':
    defant_run = Integrator('anabatic_FTCS.yaml','ext_defant')
    u_grid, theta_grid = defant_run.solve_eqns()
    
    defant_run.make_plot(u_grid, theta_grid)

# +
remake_plot = Integrator('anabatic_lowres.yaml','ext_prandtl')
u_grid, theta_grid = remake_plot.solve_eqns()
    
remake_plot.make_plot(u_grid, theta_grid)
# -




