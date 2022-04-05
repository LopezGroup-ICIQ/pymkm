"""electroMKM class for electrocatalysis."""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from natsort import natsorted
from constants import *
from functions import *
from parser import preprocess, get_NGR_NR

class electroMKM:
    """
    A class for representing microkinetic models for electrocatalysis. 
    It provides functions for obtaining information as reaction rates (current density),
    steady-state surface coverage.
    Attributes: 
        name(string): Name of the system under study.
        rm_input_file(string): Path to the .mkm file containing the reaction mechanism.
        g_input_file(string): Path to the .mkm file containing the energy values for TS, surface
                              and gas species.   
        reactor_model(string): reactor model used for representing the system under study. 
                               Two available options:
                                   "differential": differential PFR, zero conversion model
                                   "dynamic": dynamic CSTR, integral model (finite conversion)
        t_ref(float): Reference temperature at which entropic contributions have been calculated [K].
        inerts(list): Inert species in the system under study. The list contains the species name as strings.
    """
    def __init__(self,
                 name,
                 rm_input_file,
                 g_input_file,
                 t_ref=298.15,
                 reactor='differential',
                 inerts=[]):

        self.name = name
        self.input_rm = rm_input_file
        self.input_g = g_input_file
        self.t_ref = t_ref
        self.reactor_model = reactor
        self.inerts = inerts
        ############################################################################
        # rm.mkm -> Global reactions, stoichiometric matrix, species, reaction types
        ############################################################################
        lines = preprocess(rm_input_file, 3)
        self.NGR, self.NR = get_NGR_NR(lines)
        global_reaction_label = []
        global_reaction_index = []
        global_reaction_string = []
        for reaction in range(self.NGR):
            global_reaction_label.append(lines[reaction].split()[0])
            global_reaction_index.append(int(lines[reaction].split()[1][:-1]))
            global_reaction_string.append(
                " ".join(lines[reaction].split()[2:]))
        self.target = global_reaction_index[0]
        self.target_label = global_reaction_label[0]
        self.by_products = global_reaction_index[1:]
        self.by_products_label = global_reaction_label[1:]
        self.grl = dict(zip(global_reaction_label, global_reaction_index))
        self.gr_string = global_reaction_string
        self.gr_dict = dict(zip(global_reaction_label, global_reaction_string))
        # Convention: global and elementary reaction are separated by 3 blank lines in rm.mkm
        reaction_type_list = []
        species_label = []
        species_sur_label = []
        species_gas_label = []
        charge_transfer_reactions = []

        for reaction in range(self.NR):
            line = lines[reaction + 3 + self.NGR]
            line_list = line.split()
            arrow_index = line_list.index('->')
            try:  # Extraction of reaction type
                gas_index = line_list.index(
                    [element for idx, element in enumerate(line_list) if '(g)' in element][0])
            except IndexError:
                reaction_type_list.append('sur')  # Surface reaction
            else:
                if gas_index < arrow_index:
                    reaction_type_list.append('ads')  # Adsorption
                else:
                    reaction_type_list.append('des')  # Desorption
            if "H(e)".format() in line:
                charge_transfer_reactions.append("R{}".format(reaction+1)) # Charge transfer steps
                reaction_type_list[reaction] += "+e"
                # check correctness of the redox balance
            for element in line_list:  # Extraction of all species
                if (element == '+') or (element == '->'):
                    pass
                elif (element[0] in {'2', '3', '4'}):
                    element = element[1:]
                    if element in species_label:
                        pass
                    else:
                        species_label.append(element)
                elif element in species_label:
                    pass
                else:
                    species_label.append(element)

        if inerts != None:
            for i in inerts:
                species_label.append(i+'(g)')
        v_matrix = np.zeros((len(species_label), self.NR))

        for element in species_label:  # Classification of species (sur/gas)
            if '(g)' in element:
                species_gas_label.append(element)
            else:
                species_sur_label.append(element)

        species_sur_label = natsorted(species_sur_label)
        species_label = species_sur_label + species_gas_label

        for reaction in range(self.NR):  # Construction of stoichiometric matrix
            line = lines[self.NGR + 3 + reaction].split()
            arrow_index = line.index('->')
            for species in range(len(species_label)):
                if species_label[species] in line:
                    species_index = line.index(species_label[species])
                    if species_index < arrow_index:
                        v_matrix[species, reaction] = -1
                    else:
                        v_matrix[species, reaction] = 1
                elif '2' + species_label[species] in line:
                    species_index = line.index('2' + species_label[species])
                    if species_index < arrow_index:
                        v_matrix[species, reaction] = -2
                    else:
                        v_matrix[species, reaction] = 2
        ###########################################################################
        self.NC_sur = len(species_sur_label)     # Number of surface species
        self.NC_gas = len(species_gas_label)     # Number of gaseous species
        self.NC_tot = self.NC_sur + self.NC_gas  # Total number of species
        self.v_matrix = v_matrix.astype(int)     # Stoichiometric matrix
        self.species_sur = species_sur_label     # List of surface intermediates labels
        self.species_gas = species_gas_label     # List of gaseous species labels
        self.species_tot = self.species_sur + self.species_gas  # List of all species
        # List with description of each elementary step
        self.reaction_type = reaction_type_list
        self.masses = []
        for i in self.species_gas:
            mw = 0.0
            MWW = []
            i = i.strip('(g)')
            for j in range(len(i)):
                if j != (len(i)-1):
                    if i[j+1].islower():  # next char is lower case (example: Br, Ar)
                        x = i[j:j+2][0] + i[j:j+2][1]
                        MWW.append(x)
                    else:
                        if i[j] in int_set:  # CH3, NH2
                            for k in range(int(i[j]) - 1):
                                MWW.append(MWW[-1])
                        elif i[j].islower():
                            pass
                        else:
                            MWW.append(i[j])
                else:  # last string char
                    if i[j] in int_set:  # CH3
                        for k in range(int(i[j]) - 1):
                            MWW.append(MWW[-1])
                    elif i[j].islower():  # CH3Br
                        pass
                    else:  # H3N
                        MWW.append(i[j])
            for i in MWW:
                mw += m_dict[i]
            self.masses.append(mw)
        self.MW = dict(zip(self.species_gas, self.masses))
        #----------------------------------------------------------------------------------
        self.m = [0]*self.NR   # List needed for adsorption kinetic constants
        for i in range(self.NR):
            if self.reaction_type[i] == 'sur':
                pass
            else:
                for j in range(self.NC_gas):
                    if self.v_matrix[self.NC_sur+j, i] == 0:
                        pass
                    else:
                        self.m[i] = self.masses[j] / (N_AV*1000)
        ###########################################################################
        # g.mkm -> System energetics (H, S, G)
        ###########################################################################
        lines = preprocess('./{}'.format(g_input_file), 6)
        for i in range(len(lines)): 
            lines[i] = lines[i].strip("\n")
        E_ts = lines[:self.NR]
        E_species = [i for i in lines[self.NR+3:] if i != ""]
        H_ts = np.zeros(self.NR)
        S_ts = np.zeros(self.NR)
        self.alfa = np.zeros(self.NR)  # charge transfer coefficient
        H_species = np.zeros(self.NC_tot)
        S_species = np.zeros(self.NC_tot)
        keys_R = []
        keys_species = []
        for j in range(len(E_ts)):
            keys_R.append(E_ts[j].split()[0])
        for j in range(len(E_species)):
            keys_species.append(E_species[j].split()[0].strip(':'))
        for i in range(self.NR):
            index = keys_R.index('R{}:'.format(i+1))
            H_ts[i] = float(E_ts[index].split()[1])
            S_ts[i] = float(E_ts[index].split()[2]) / t_ref
            self.alfa[i] = float(E_ts[index].split()[3])
        for i in range(self.NC_tot):
            if self.species_tot[i].strip('(g)') not in inerts:
                index = keys_species.index(self.species_tot[i])
                H_species[i] = float(E_species[index].split()[1])
                S_species[i] = float(E_species[index].split()[-1]) / t_ref
            else:
                H_species[i] = 0.0
                S_species[i] = 0.0
        self.h_species = H_species
        self.s_species = S_species
        self.g_species = H_species - t_ref * S_species
        self.h_ts = H_ts
        self.s_ts = S_ts
        self.g_ts = H_ts - t_ref * S_ts
        ###########################################################################
        # Convert global reaction string to stoichiometric vectors
        ###########################################################################
        self.v_global = np.zeros((self.NC_tot, self.NGR))
        for i in range(self.NC_tot):
            for j in range(self.NGR):
                reaction_list = self.gr_string[j].split()
                arrow_index = reaction_list.index('->')
                if ((self.species_tot[i] == "H(e)") and (self.species_tot[i] in reaction_list)):
                    if reaction_list.index(self.species_tot[i]) < arrow_index:
                        self.v_global[i, j] = -1
                    else:
                        self.v_global[i, j] = 1
                elif ((self.species_tot[i] == "H(e)") and ("2{}".format(self.species_tot[i]) in reaction_list)):
                    if reaction_list.index("2{}".format(self.species_tot[i])) < arrow_index:
                        self.v_global[i, j] = -2
                    else:
                        self.v_global[i, j] = 2                 
                elif self.species_tot[i].strip('(g)') in reaction_list:
                    if reaction_list.index(self.species_tot[i].strip('(g)')) < arrow_index:
                        self.v_global[i, j] = -1
                    else:
                        self.v_global[i, j] = 1 
                else:
                    if '2'+self.species_tot[i].strip('(g)') in reaction_list:
                        if reaction_list.index('2'+self.species_tot[i].strip('(g)')) < arrow_index:
                            self.v_global[i, j] = -2
                        else:
                            self.v_global[i, j] = 2
                    elif '3'+self.species_tot[i].strip('(g)') in reaction_list:
                        if reaction_list.index('3'+self.species_tot[i].strip('(g)')) < arrow_index:
                            self.v_global[i, j] = -3
                        else:
                            self.v_global[i, j] = 3
        for i in range(self.NC_tot):
            for j in range(self.NGR):
                if (i < self.NC_sur) and (self.species_tot[i]+'(g)' in self.species_gas):
                    self.v_global[i, j] = 0
        #############################################################################
        # Stoichiometric vector for global reactions
        #############################################################################
        self.stoich_numbers = np.zeros((self.NR, self.NGR))
        for i in range(self.NGR):
            sol = np.linalg.lstsq(self.v_matrix, self.v_global[:, i], rcond=None)
            self.stoich_numbers[:, i] = np.round(sol[0], decimals=2)
        #############################################################################
        # Reaction energy profiles (H, S and G)
        #############################################################################
        self.dh_reaction = np.zeros(self.NR)
        self.ds_reaction = np.zeros(self.NR)
        self.dg_reaction = np.zeros(self.NR)
        self.dh_barrier = np.zeros(self.NR)
        self.ds_barrier = np.zeros(self.NR)
        self.dg_barrier = np.zeros(self.NR)
        self.dh_barrier_rev = np.zeros(self.NR)
        self.ds_barrier_rev = np.zeros(self.NR)
        self.dg_barrier_rev = np.zeros(self.NR)
        for i in range(self.NR):
            self.dh_reaction[i] = np.sum(
                self.v_matrix[:, i]*np.array(self.h_species))
            self.ds_reaction[i] = np.sum(
                self.v_matrix[:, i]*np.array(self.s_species))
            self.dg_reaction[i] = self.dh_reaction[i] - \
                t_ref * self.ds_reaction[i]
            condition1 = self.g_ts[i] != 0.0
            ind = list(np.where(
                self.v_matrix[:, i] == -1)[0]) + list(np.where(self.v_matrix[:, i] == -2)[0])
            his = sum([self.h_species[j]*self.v_matrix[j, i]*(-1)
                      for j in ind])
            sis = sum([self.s_species[j]*self.v_matrix[j, i]*(-1)
                      for j in ind])
            gis = sum([self.g_species[j]*self.v_matrix[j, i]*(-1)
                      for j in ind])
            condition2 = self.g_ts[i] > max(gis, gis+self.dg_reaction[i])
            if condition1 and condition2:  # Activated elementary reaction
                self.dh_barrier[i] = self.h_ts[i] - his
                self.dh_barrier_rev[i] = self.dh_barrier[i] - \
                    self.dh_reaction[i]
                self.ds_barrier[i] = self.s_ts[i] - sis
                self.ds_barrier_rev[i] = self.ds_barrier[i] - \
                    self.ds_reaction[i]
                self.dg_barrier[i] = self.g_ts[i] - gis
                self.dg_barrier_rev[i] = self.dg_barrier[i] - \
                    self.dg_reaction[i]
            else:  # Unactivated elementary reaction
                if self.dg_reaction[i] < 0.0:
                    self.dh_barrier[i] = 0.0
                    self.dh_barrier_rev[i] = -self.dh_reaction[i]
                    self.ds_barrier[i] = 0.0
                    self.ds_barrier_rev[i] = -self.ds_reaction[i]
                    self.dg_barrier[i] = 0.0
                    self.dg_barrier_rev[i] = -self.dg_reaction[i]
                else:
                    self.dh_barrier[i] = self.dh_reaction[i]
                    self.dh_barrier_rev[i] = 0.0
                    self.ds_barrier[i] = self.ds_reaction[i]
                    self.ds_barrier_rev[i] = 0.0
                    self.dg_barrier[i] = self.dg_reaction[i]
                    self.dg_barrier_rev[i] = 0.0
        #############################################################################################
        self.ODE_params = [1e-12, 1e-64, 1.0E3] # Default ODE parameters (reltol, abstol and t_final)
        self.v_f = stoic_forward(self.v_matrix)
        self.v_b = stoic_backward(self.v_matrix)
        self.r = []
        for i in range(self.NR):
            self.r.append('R{}'.format(i+1))
        self.df_system = pd.DataFrame(self.v_matrix, index=species_sur_label+species_gas_label,
                                      columns=[self.r, reaction_type_list])
        self.df_system.index.name = 'species'
        self.df_gibbs = pd.DataFrame(np.array([self.dg_reaction,
                                               self.dg_barrier,
                                               self.dg_barrier_rev]).T,
                                     index=[self.r, reaction_type_list],
                                     columns=['DGR / eV',
                                              'DG barrier / eV',
                                              'DG reverse barrier / eV'])
        self.df_gibbs.index.name = 'reaction'
#-------------------------------------------------------------------------------------------------------------#
    def __str__(self):
        print("System: {}".format(self.name))
        print("")
        for i in self.gr_string:
            print(i)
        print("")
        print("Number of global reactions: {}".format(self.NGR))
        print("Number of elementary reactions: {}".format(self.NR))
        print("Number of surface species: {}".format(self.NC_sur))
        print("Number of gas species: {}".format(self.NC_gas))
        return ""

    def info(self):
        print(self)

    def set_ODE_params(self, t_final=1000.0, reltol=1e-12, abstol=1e-64):
        """
        Set paramters for numerical integration of ODE solvers.
        Args:
            t_final(float): total integration time [s]
            reltol(float): relative tolerance 
            abstol(float): absolute tolerance
        """
        self.ODE_params[0] = reltol
        self.ODE_params[1] = abstol
        self.ODE_params[2] = t_final
        print("Final integration time = {}s".format(t_final))
        print("Relative tolerance = {}".format(reltol))
        print("Absolute tolerance = {}".format(abstol))
        return "Changed ODE solver parameters."

    def kinetic_coeff(self, overpotential, temperature, area_active_site=1e-19):
        """
        Returns the kinetic coefficient for the direct and reverse reactions, according to 
        the reaction type (adsorption, desorption or surface reaction) and TST.
        Revisited from pymkm for electrocatalysis.                
        Args: 
            overpotential(float): applied overpotential [V].
            temperature(float): absolute temperature [K].
            A_site_0(float): Area of the catalytic ensemble [m2]. Default: 1e-19[m2].
        Returns:
            (list): list with 2 ndarrays for direct and reverse kinetic coefficients.
        """
        Keq = np.zeros(self.NR)  # Equilibrium constant
        kd = np.zeros(self.NR)   # Direct constant
        kr = np.zeros(self.NR)   # Reverse constant
        for reaction in range(self.NR):
            Keq[reaction] = np.exp(-self.dg_reaction[reaction] / (temperature * K_B))
            if self.reaction_type[reaction] == 'ads':
                kd[reaction] = (K_B * temperature / H) * np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                kr[reaction] = kd[reaction] / Keq[reaction]
            elif self.reaction_type[reaction] == 'des':
                kd[reaction] = (K_B * temperature / H ) * \
                    np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                kr[reaction] = kd[reaction] / Keq[reaction]
            elif self.reaction_type[reaction] == 'sur':  
                kd[reaction] = (K_B * temperature / H) * np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                kr[reaction] = kd[reaction] / Keq[reaction]
            else: # Charge transfer reaction
                f = F / (R * temperature)  # C/J
                index = self.species_tot.index('H(e)')
                if self.v_matrix[index, reaction] < 0: # Reduction (e- in the lhs of the reaction)
                    kd[reaction] = (K_B * temperature / H) * np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                    kd[reaction] *= np.exp(- self.alfa[reaction] * f * overpotential)
                    Keq[reaction] *= np.exp(-f * overpotential)
                    kr[reaction] = kd[reaction] / Keq[reaction]
                else: # Oxidation (e- in the rhs of the reaction)
                    kd[reaction] = (K_B * temperature / H) * np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                    kd[reaction] *= np.exp((1 - self.alfa[reaction]) * f * overpotential)
                    Keq[reaction] *= np.exp(f * overpotential)
                    kr[reaction] = kd[reaction] / Keq[reaction]
        return kd, kr

    def net_rate(self, y, kd, ki):
        """
        Returns the net reaction rate for each elementary reaction.
        Args:
            y(ndarray): surface coverage + partial pressures array [-/Pa].
            kd, ki(ndarray): kinetic constants of the direct/reverse elementary reactions.
        Returns:
            ndarray containing the net reaction rate for all the steps [1/s].
        """
        net_rate = np.zeros(len(kd))
        v_ff = self.v_f
        v_bb = self.v_b
        net_rate = kd * np.prod(y ** v_ff.T, axis=1) - ki * np.prod(y ** v_bb.T, axis=1)
        return net_rate

    def differential_pfr(self, time, y, kd, ki):
        """
        Returns the rhs of the ODE system for solve_ivp.
        Reactor model: differential PFR (zero conversion)
        """
        # Surface species
        dy = self.v_matrix @ self.net_rate(y, kd, ki)
        # Gas species
        dy[self.NC_sur:] = 0.0
        # H+ activity is constant, defined by pH
        index = self.species_tot.index('H(e)')
        dy[index] = 0.0 
        return dy

    def jac_diff(self, time, y, kd, ki):
        """
        Returns the analytical Jacobian matrix of the system for
        the differential reactor model.
        """
        J = np.zeros((len(y), len(y)))
        Jg = np.zeros((len(kd), len(y)))
        Jh = np.zeros((len(kd), len(y)))
        v_f = stoic_forward(self.v_matrix)
        v_b = stoic_backward(self.v_matrix)
        for r in range(len(kd)):
            for s in range(len(y)):
                if v_f[s, r] == 1:
                    v_f[s, r] -= 1
                    Jg[r, s] = kd[r] * np.prod(y ** v_f[:, r])
                    v_f[s, r] += 1
                elif v_f[s, r] == 2:
                    v_f[s, r] -= 1
                    Jg[r, s] = 2 * kd[r] * np.prod(y ** v_f[:, r])
                    v_f[s, r] += 1
                if v_b[s, r] == 1:
                    v_b[s, r] -= 1
                    Jh[r, s] = ki[r] * np.prod(y ** v_b[:, r])
                    v_b[s, r] += 1
                elif v_b[s, r] == 2:
                    v_b[s, r] -= 1
                    Jh[r, s] = 2 * ki[r] * np.prod(y ** v_b[:, r])
                    v_b[s, r] += 1
        J = self.v_matrix @ (Jg - Jh)
        J[self.NC_sur:, :] = 0
        index = self.species_tot.index('H(e)')
        J[index,:] = 0
        return J

    def __ode_solver_solve_ivp(self,
                               y_0,
                               dy,
                               temperature, 
                               overpotential, 
                               reltol,
                               abstol,
                               t_final,
                               end_events=None,
                               jacobian_matrix=None):
        """
        Helper function for solve_ivp integrator.
        """
        kd, ki = self.kinetic_coeff(overpotential, 
                                    temperature)
        args_list = [kd, ki]
        r = solve_ivp(dy,
                      (0.0, t_final),
                      y_0,
                      method='BDF', 
                      events=end_events,
                      jac=jacobian_matrix,
                      args=args_list,
                      atol=abstol,
                      rtol=reltol,
                      max_step=t_final)
        return r

    def kinetic_run(self,
                    overpotential,
                    pH,
                    initial_conditions=None,
                    temperature=298.0,
                    pressure=1e5,
                    gas_composition=None,
                    verbose=0,
                    jac=False):
        """
        Simulates a steady-state electrocatalytic run at the defined operating conditions.        
        Args:
            overpotential(float): applied overpotential [V vs SHE].
            pH(float): pH of the electrolyte solution [-].
            temperature(float): Temperature of the system [K].
            pressure(float): Absolute pressure of the system [Pa].
            initial_conditions(nparray): Initial surface coverage [-].
            verbose(int): 0=print all output; 1=print nothing.        
        Returns:
            (dict): Report of the electrocatalytic simulation.        
        """
        if verbose == 0:
            print('{}: Microkinetic run'.format(self.name))
            print('Overpotential = {}V vs SHE    pH = {}'.format(overpotential, pH))
            print('Temperature = {}K    Pressure = {:.1f}bar'.format(temperature, pressure/1e5))
        y_0 = np.zeros(self.NC_tot)
        if initial_conditions is None:  # Convention: first surface species is the active site
            y_0[0] = 1.0
            indexH = self.species_tot.index("H(e)") 
            y_0[indexH] = 10 ** (-pH)
        else:
            y_0[:self.NC_sur] = initial_conditions[:self.NC_sur]
        if gas_composition is None:
            y_0[self.NC_sur:] = 0.0
        else:
            y_0[self.NC_sur:] = pressure * gas_composition / 1e5
        #print(y_0)
        #-----------------------------------------------------------------------------------------------
        if temperature < 0.0:
            raise ValueError('Wrong temperature (T > 0 K)')
        if pressure < 0.0:
            raise ValueError('Wrong pressure (P > 0 Pa)')
        if pH < 0.0 or pH > 14:
            raise ValueError('Wrong pH definition (0 < pH < 14)')
        #-----------------------------------------------------------------------------------------------
        results_sr = []                      # solver output
        final_sr = []                        # final Value of derivatives
        yfin_sr = np.zeros((self.NC_tot))    # steady-state output [-]
        r_sr = np.zeros((self.NR))           # reaction rate [1/s]
        s_target_sr = np.zeros(1)            # selectivity
        t0 = time.time()
        keys = ['T',
                'P',
                'theta',
                'ddt',
                'r',
                *['r_{}'.format(i) for i in list(self.grl.keys())],
                *['j_{}'.format(i) for i in list(self.grl.keys())],
                'S_{}'.format(self.target_label),
                'MASI',
                'solver']
        r = ['R{}'.format(i+1) for i in range(self.NR)]
        values = [temperature, pressure / 1e5]
        _ = None
        if jac: 
            _ = self.jac_diff
        results_sr = self.__ode_solver_solve_ivp(y_0,
                                                 self.differential_pfr,
                                                 temperature,
                                                 overpotential,
                                                 *self.ODE_params,
                                                 end_events=None,
                                                 jacobian_matrix=_)
        final_sr = self.differential_pfr(results_sr.t[-1],
                                         results_sr.y[:, -1],
                                         *self.kinetic_coeff(overpotential,
                                                             temperature))
        yfin_sr = results_sr.y[:self.NC_sur, -1]
        r_sr = self.net_rate(results_sr.y[:, -1],
                            *self.kinetic_coeff(overpotential,
                                                temperature))
        j_sr = -r_sr * F / (N_AV * 1.0E-19)
        bp = list(set(self.by_products))
        s_target_sr = r_sr[self.target] / (r_sr[self.target] + r_sr[bp].sum())
        value_masi = max(yfin_sr[:self.NC_sur-1])
        key_masi = self.species_sur[np.argmax(yfin_sr[:self.NC_sur-1])]
        masi_sr = {key_masi: value_masi}
        coverage_dict = dict(zip(self.species_sur, yfin_sr))
        ddt_dict = dict(zip(self.species_tot, final_sr))
        r_dict = dict(zip(r, r_sr))
        values += [coverage_dict,
                   ddt_dict,
                   r_dict,
                   *[r_sr[i] for i in list(self.grl.values())],
                   *[j_sr[i] for i in list(self.grl.values())],
                   s_target_sr,
                   masi_sr,
                   results_sr]
        output_dict = dict(zip(keys, values))
        if verbose == 0:
            print('')
            print('{} Current density: {:0.2e} mA cm-2'.format(self.target_label,
                                                               j_sr[self.target]/10))
            print('{} Selectivity: {:.2f}%'.format(self.target_label,
                                                   s_target_sr*100.0))
            print('Most Abundant Surface Intermediate: {} Coverage: {:.2f}% '.format(
                key_masi, value_masi*100.0))
            print('CPU time: {:.2f} s'.format(time.time() - t0))
        return output_dict

    def tafel_plot(self,
                   reaction_label,
                   overpotential_vector,
                   pH,
                   initial_conditions=None,
                   temperature=298.0,
                   pressure=1e5,
                   gas_composition=None,
                   verbose=0,
                   jac=False):
        """
        Returns the Tafel plot for the defined potential range.
        Args:
            reaction_label(str): Label of the reaction of interest.
            overpotential_vector(ndarray): applied overpotential vector [V].
            pH(float): pH of the electrolyte solution [-].
            initial_conditions(ndarray): initial surface coverage and gas composition [-]
            temperature(float): Temperature of the system [K].
            pressure(float): Absolute pressure of the system [Pa].
            verbose(bool): 0=; 1=.
            jac(bool): Inclusion of the analytical Jacobian for ODE numerical solution.
        """
        exp = []
        j_vector = np.zeros(len(overpotential_vector))
        if reaction_label not in self.grl.keys():
            raise ValueError("Unexisting reaction label")
        print("{}: Tafel slope experiment for {}".format(self.name, reaction_label))
        print("Temperature: {} K    Pressure: {} bar    pH: {}".format(temperature, int(pressure/1e5), pH))
        print("")
        time0 = time.time()
        for i in range(len(overpotential_vector)):
            exp.append(self.kinetic_run(overpotential_vector[i],
                                           pH,
                                           initial_conditions=initial_conditions,
                                           temperature=temperature,
                                           pressure=pressure,
                                           gas_composition=gas_composition,
                                           verbose=1,
                                           jac=jac))
            j_vector[i] = exp[i]['j_{}'.format(reaction_label)]
            if overpotential_vector[i] < 0:
                print("Overpotential = {} V    {} Current Density = {:.2e} mA cm-2".format(overpotential_vector[i],
                                                                                           reaction_label,
                                                                                           j_vector[i]/10))
            else:
                print("Overpotential = +{} V    {} Current Density = {:.2e} mA cm-2".format(overpotential_vector[i],
                                                                                           reaction_label,
                                                                                           j_vector[i]/10))
        print("------------------------------------------------------------------")
        tafel_slope = calc_tafel_slope(overpotential_vector, j_vector)[0]
        f = F / R / temperature
        alfa = 1 + (tafel_slope / f) # Global charge transfer coefficient
        print("Tafel slope = {:.2f} mV    alfa = {:.2f}".format(tafel_slope*1000, alfa))
        print("CPU time: {:.2f} s".format(time.time() - time0)) 
        fig, ax = plt.subplots(2, figsize=(7,5), dpi=400)
        ax[0].plot(overpotential_vector, j_vector/10, 'ok', linewidth=4)
        ax[0].set(xlabel="Overpotential / V vs SHE", ylabel="j / mA cm-2", title="j vs U")
        ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[0].grid()
        ax[1].plot(overpotential_vector, np.log10(abs(j_vector)), 'ok')
        ax[1].set(title="{}: Tafel plot".format(self.name), xlabel="Overpotential / V vs SHE", ylabel="log10(|j|)")
        ax[1].grid()
        plt.tight_layout()
        plt.show()
        plt.savefig("{}_tafel.svg".format(self.name))            
        return tafel_slope   
    
    def degree_of_rate_control(self,
                               global_reaction_label,
                               ts_int_label,
                               overpotential,
                               pH,
                               initial_conditions=None,
                               temperature=298.0,
                               pressure=1e5,
                               gas_composition=None,
                               dg=1.0E-6,
                               jac=False,
                               verbose=0):
        """
        Calculates the degree of rate control(DRC) and selectivity control(DSC)
        for the selected transition state or intermediate species.
        Args:
            global_reaction_label(str): Global reaction for which DRC/DSC are computed
            ts_int_label(str): Transition state/surface intermediate for which DRC/DSC are computed
            overpotential_vector(ndarray): applied overpotential vector [V].
            pH(float): pH of the electrolyte solution [-].
            temperature(float): Temperature of the experiment [K]
            pressure(float): Partial pressure of the gaseous species [Pa]
            gas_composition(list): Molar fraction of the gaseous species [-]
            initial_conditions(nparray): Initial surface coverage [-]
            dg(float): Deviation applied to selected Gibbs energy to calculate the DRC/DSC values.
                       Default=1E-6 eV
            jac(bool): Inclusion of the analytical Jacobian for ODE numerical solution.
            verbose(int): 1= Print essential info
                          0= Print additional info
        Returns:
            List with DRC and DSC of TS/intermediate for the selected reaction [-]
        """
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception(
                'Reaction label must be related to a global reaction!')
        switch_ts_int = 0  # 0=TS 1=intermediate species
        if 'R' not in ts_int_label:
            switch_ts_int = 1
        index = 0
        if switch_ts_int == 0:
            index = int(ts_int_label[1:]) - 1
        else:
            index = self.species_sur.index(ts_int_label)

        if verbose == 0:
            if switch_ts_int == 0:
                print('{}: DRC analysis for elementary reaction R{} wrt {} reaction'.format(self.name,
                                                                                            index+1,
                                                                                            global_reaction_label))
            else:
                print('{}: DRC and DSC for intermediate {} wrt {} reaction'.format(self.name,
                                                                                   self.species_sur[index],
                                                                                   global_reaction_label))
            print('Temperature = {}K    Pressure = {:.1f}bar'.format(
                temperature, pressure/1e5))
            sgas = []
            for i in self.species_gas:
                sgas.append(i.strip('(g)'))
            str_list = ['{}={:.1f}%  '.format(i, j) for i, j in zip(sgas,
                                                                    list(np.array(gas_composition)*100.0))]
            gas_string = 'Gas composition: '
            for i in str_list:
                gas_string += i
            print(gas_string)
            print('')
        r = np.zeros(2)
        s = np.zeros(2)
        if switch_ts_int == 0:    # Transition state
            if self.g_ts[index] != 0.0:  # Originally activated reaction
                for i in range(2):
                    mk_object = electroMKM('i',
                                    self.input_rm,
                                    self.input_g,
                                    t_ref=self.t_ref,
                                    reactor=self.reactor_model,
                                    inerts=self.inerts)
                    mk_object.dg_barrier[index] += dg*(-1)**(i)
                    mk_object.dg_barrier_rev[index] += dg*(-1)**(i)
                    run = mk_object.kinetic_run(overpotential,
                                                pH,
                                                initial_conditions=initial_conditions,
                                                temperature=temperature,
                                                pressure=pressure,
                                                gas_composition=gas_composition,
                                                verbose=1,
                                                jac=jac)
                    if mk_object.reactor_model == 'differential':
                        r[i] = list(run['r'].values())[
                            self.grl[global_reaction_label]]
                        r_tot = list(run['r'].values())
                        r_tot = [r_tot[i] for i in range(
                            self.NR) if i in list(self.grl.values())]
                        s[i] = r[i] / sum(r_tot)
                    else:  # dynamic CSTR
                        r[i] = run['R_' + global_reaction_label]
                drc = (-K_B*temperature) * (np.log(r[0])-np.log(r[1])) / (2*dg)
                dsc = (-K_B*temperature) * (np.log(s[0])-np.log(s[1])) / (2*dg)
            else:  # Originally unactivated reaction
                for i in range(2):
                    mk_object = electroMKM('i',
                                    self.input_rm,
                                    self.input_g,
                                    t_ref=self.t_ref,
                                    reactor=self.reactor_model,
                                    inerts=self.inerts)
                    if mk_object.dg_reaction[index] < 0.0:
                        mk_object.dg_barrier[index] = dg * i
                        mk_object.dg_barrier_rev[index] += dg * i
                    else:
                        mk_object.dg_barrier[index] = mk_object.dg_reaction[index] + dg * i
                    run = mk_object.kinetic_run(overpotential,
                                                pH,
                                                initial_conditions=initial_conditions,
                                                temperature=temperature,
                                                pressure=pressure,
                                                gas_composition=gas_composition,
                                                verbose=1,
                                                jac=jac)
                    if mk_object.reactor_model == 'differential':
                        r[i] = list(run['r'].values())[
                            self.grl[global_reaction_label]]
                        r_tot = list(run['r'].values())
                        r_tot = [r_tot[i] for i in range(
                            self.NR) if i in list(self.grl.values())]
                        s[i] = r[i] / sum(r_tot)
                    else:  # dynamic CSTR
                        r[i] = run['R_'+global_reaction_label]
                drc = (-K_B*temperature) * (np.log(r[1])-np.log(r[0])) / dg
                dsc = (-K_B*temperature) * (np.log(s[1])-np.log(s[0])) / dg
        else:  # Surface intermediate
            for i in range(2):
                mk_object = electroMKM('i',
                                       self.input_rm,
                                       self.input_g,
                                       t_ref=self.t_ref,
                                       reactor=self.reactor_model,
                                       inerts=self.inerts)
                mk_object.g_species[index] += dg * (-1) ** (i)
                for j in range(mk_object.NR):
                    mk_object.dg_reaction[j] = np.sum(
                        mk_object.v_matrix[:, j]*np.array(mk_object.g_species))
                    condition1 = mk_object.g_ts[j] != 0.0
                    ind = list(np.where(mk_object.v_matrix[:, j] == -1)[0]) + list(
                        np.where(mk_object.v_matrix[:, j] == -2)[0])
                    gis = sum([mk_object.g_species[k] *
                              mk_object.v_matrix[k, j]*(-1) for k in ind])
                    condition2 = mk_object.g_ts[j] > max(
                        gis, gis+mk_object.dg_reaction[j])

                    if condition1 and condition2:  # Activated elementary reaction
                        mk_object.dg_barrier[j] = mk_object.g_ts[j] - gis
                        mk_object.dg_barrier_rev[j] = mk_object.dg_barrier[j] - \
                            mk_object.dg_reaction[j]
                    else:  # Unactivated elementary reaction
                        if mk_object.dg_reaction[j] < 0.0:
                            mk_object.dg_barrier[j] = 0.0
                            mk_object.dg_barrier_rev[j] = - \
                                mk_object.dg_reaction[j]
                        else:
                            mk_object.dg_barrier[j] = mk_object.dg_reaction[j]
                            mk_object.dg_barrier_rev[j] = 0.0
                run = mk_object.kinetic_run(overpotential,
                                            pH,
                                            initial_conditions=initial_conditions,
                                            temperature=temperature,
                                            pressure=pressure,
                                            gas_composition=gas_composition,
                                            verbose=1,
                                            jac=jac)
                if mk_object.reactor_model == 'differential':
                    r[i] = list(run['r'].values())[
                        self.grl[global_reaction_label]]
                    r_tot = list(run['r'].values())
                    r_tot = [r_tot[i] for i in range(
                        self.NR) if i in list(self.grl.values())]
                    s[i] = r[i] / sum(r_tot)
                else:  # dynamic CSTR
                    r[i] = run['R_'+global_reaction_label]
            drc = (-K_B*temperature) * (np.log(r[0])-np.log(r[1])) / (2*dg)
            dsc = (-K_B*temperature) * (np.log(s[0])-np.log(s[1])) / (2*dg)
        print('DRC = {:0.2f}'.format(drc))
        return drc, dsc

    def drc_full(self,
                 global_reaction_label,
                 overpotential,
                 pH,
                 initial_conditions=None,
                 temperature=298.0,
                 pressure=1e5,
                 gas_composition=None,
                 dg=1.0E-6,
                 jac=False):
        """
        Wrapper function that calculates the degree of rate control of all
        intermediates and transition states at the desired conditions.        
        Args:
            global_reaction_label(str): Global reaction for which DRC/DSC are computed
            overpotential_vector(ndarray): applied overpotential vector [V].
            pH(float): pH of the electrolyte solution [-].
            temperature(float): Temperature of the experiment [K]
            pressure(float): Partial pressure of the gaseous species [Pa]
            gas_composition(list): Molar fraction of the gaseous species [-]
            initial_conditions(nparray): Initial surface coverage [-]
            dg(float): Deviation applied to selected Gibbs energy to calculate the DRC/DSC values.
                       Default=1E-6 eV
            jac(bool): Inclusion of the analytical Jacobian for ODE numerical solution.
            verbose(int): 1= Print essential info
                          0= Print additional info

        Returns:
            Two Pandas DataFrames with final all results.        
        """
        print('{}: Full DRC and DSC analysis wrt {} global reaction'.format(self.name,
                                                                            global_reaction_label))
        print('Temperature = {}K    Pressure = {:.1f}bar'.format(
            temperature, pressure/1e5))
        sgas = []
        for i in self.species_gas:
            sgas.append(i.strip('(g)'))
        str_list = ['{}={:.1f}%  '.format(i, j) for i, j in zip(sgas,
                                                                list(np.array(gas_composition)*100.0))]
        gas_string = 'Gas composition: '
        for i in str_list:
            gas_string = gas_string + i
        print(gas_string)
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception('Unexisting global reaction string.')
        if dg > 0.1:
            raise Exception(
                'Too high perturbation (recommended lower than 1e-6 eV)')
        drc_ts = np.zeros(self.NR)
        dsc_ts = np.zeros(self.NR)
        drc_int = np.zeros(self.NC_sur)
        dsc_int = np.zeros(self.NC_sur)
        for reaction in range(self.NR):
            print('')
            print('R{}'.format(reaction+1))
            drc_ts[reaction], dsc_ts[reaction] = self.degree_of_rate_control(global_reaction_label,
                                                                            'R{}'.format(reaction+1),
                                                                            overpotential,
                                                                            pH,
                                                                            initial_conditions=initial_conditions,
                                                                            temperature=temperature,
                                                                            pressure=pressure,
                                                                            gas_composition=gas_composition,
                                                                            dg=dg,
                                                                            jac=jac,
                                                                            verbose=1)
        for species in range(self.NC_sur):
            print('')
            print('{}'.format(self.species_sur[species]))
            drc_ts[reaction], dsc_ts[reaction] = self.degree_of_rate_control(global_reaction_label,
                                                                             self.species_sur[species],
                                                                             overpotential,
                                                                             pH,
                                                                             initial_conditions=initial_conditions,
                                                                             temperature=temperature,
                                                                             pressure=pressure,
                                                                             gas_composition=gas_composition,
                                                                             dg=dg,
                                                                             jac=jac,
                                                                             verbose=1)
        r = []
        for i in range(self.NR):
            r.append('R{}'.format(i+1))
        drsc_ts = np.concatenate((np.array([drc_ts]).T,
                                  np.array([dsc_ts]).T),
                                 axis=1)
        df_drsc_ts = pd.DataFrame(np.round(drsc_ts, decimals=2),
                                  index=r,
                                  columns=['DRC', 'DSC'])
        df_drsc_ts.to_csv("X_{}_{}_{}_ts.csv".format(global_reaction_label,
                                                     int(temperature),
                                                     int(pressure/1e5)))
