"""
MKM class for heterogeneous catalysis.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from natsort import natsorted
from thermo import k_eq_H, k_eq_S, reaction_enthalpy, reaction_entropy
from constants import *
from functions import *
from reactor import *
import graphviz
from math import pi

class MKM:
    """
    A class to represent microkinetic models for heterogeneous catalysis. It provides 
    functionalities to obtain information as reaction rates, steady-state surface coverage, apparent
    activation energy and reaction orders. Moreover, it provides tools for identifying the descriptors
    of the global process, like reversibility and degree of rate control analysis.
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
                 t_ref=273.15,
                 reactor='differential',
                 inerts=[]):

        self.name = name
        self.input_rm = rm_input_file
        self.input_g = g_input_file
        self.t_ref = t_ref
        self.reactor_model = reactor
        self.inerts = inerts
        ############################################################################
        # rm.mkm parsing -> Global reactions, Stoichiometric matrix, Species 
        ############################################################################
        rm = open(rm_input_file, "r")
        lines = rm.readlines()
        self.NGR = lines.index('\n')
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
        self.NR = len(lines) - self.NGR - 3  # Number of elementary steps
        reaction_type_list = []
        species_label = []
        species_sur_label = []
        species_gas_label = []

        for reaction in range(self.NR):
            line_list = lines[reaction + 3 + self.NGR].split()
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
        # ---------------------------------------------------------------------------------
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
        # g.mkm parsing -> System energetics (H, S and G)
        ###########################################################################
        e = open('./{}'.format(g_input_file), 'r')
        lines = e.readlines()
        for i in range(len(lines)): 
            lines[i] = lines[i].strip("\n")
        E_ts = lines[:self.NR]
        E_species = [i for i in lines[self.NR+3:] if i != ""]
        H_ts = np.zeros(self.NR)
        S_ts = np.zeros(self.NR) 
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
            S_ts[i] = float(E_ts[index].split()[-1]) / t_ref
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
        self.v_global = np.zeros((self.NC_tot, self.NGR))
        for i in range(self.NC_tot):
            for j in range(self.NGR):
                reaction_list = self.gr_string[j].split()
                arrow_index = reaction_list.index('->')
                if self.species_tot[i].strip('(g)') in reaction_list:
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
        # stoichiometric vector for global_reactions
        self.stoich_numbers = np.zeros((self.NR, self.NGR))
        for i in range(self.NGR):
            sol = np.linalg.lstsq(
                self.v_matrix, self.v_global[:, i], rcond=None)
            self.stoich_numbers[:, i] = np.round(sol[0], decimals=2)
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

        self.ODE_params = [1e-12, 1e-70, 1e3]
        self.v_f = stoic_forward(self.v_matrix)
        self.v_b = stoic_backward(self.v_matrix)
        r = []
        for i in range(self.NR):
            r.append('R{}'.format(i+1))
        self.df_system = pd.DataFrame(self.v_matrix, index=species_sur_label+species_gas_label,
                                      columns=[r, reaction_type_list])
        self.df_system.index.name = 'species'
        self.df_gibbs = pd.DataFrame(np.array([self.dg_reaction,
                                               self.dg_barrier,
                                               self.dg_barrier_rev]).T,
                                     index=[r, reaction_type_list],
                                     columns=['DGR / eV',
                                              'DG barrier / eV',
                                              'DG reverse barrier / eV'])
        self.df_gibbs.index.name = 'reaction'
#-------------------------------------------------------------------------------------------------------------#
    def set_reactor(self, reactor):
        """
        Define the reactor model 
        Two options:
            "differential" = differential PFR, zero conversion model
            "dynamic" = dynamic CSTR, integral model (finite conversion)
        """
        if (reactor not in ("differential", "dynamic")):
            raise "Wrong reactor model definition. Please choose between 'differential' or 'dynamic'."
        self.reactor_model = reactor
        if reactor == "differential":
            self.reactor = DifferentialPFR()
        elif reactor == "dynamic":
            self.reactor = DynamicCSTR()
        return "Reactor model: {}".format(reactor)

    def set_CSTR_params(self,
                        radius,
                        length,
                        Q,
                        m_cat,
                        S_BET,
                        A_site=1.0E-19,
                        verbose=0):
        """ 
        Method for defining the parameters of the 
        Dynamic CSTR reactor.
        Args:
            radius(float): Reactor inner radius in [m]
            length(float): reactor length in [m]
            Q(float): inlet volumetric flowrate in [m3/s]
            m_cat(float): catalyst mass in [kg]
            S_BET(float): BET surface in [m2/kg_cat]
            A_site(float): Area of the active site in [m2]. Default to 1.0E-19
        """
        self.CSTR_V = (pi * radius ** 2) * length
        self.CSTR_Q = Q
        self.CSTR_tau = self.CSTR_V / self.CSTR_Q  # Residence time [s]
        self.CSTR_mcat = m_cat
        self.CSTR_sbet = S_BET
        self.CSTR_asite = A_site
        if verbose == 0:
            print("Reactor volume: {:0.2e} [m3]".format(self.CSTR_V))
            print(
                "Inlet volumetric flowrate : {:0.2e} [m3/s]".format(self.CSTR_Q))
            print("Residence time: {:0.2e} [s]".format(self.CSTR_tau))
            print("Catalyst mass: {:0.2e} [kg]".format(self.CSTR_mcat))
            print(
                "Catalyst surface: {:0.2e} [m2/kg_cat]".format(self.CSTR_sbet))
            print("Active site surface: {:0.2e} [m2]".format(self.CSTR_asite))
            return None
        else:
            return None

    def set_ODE_params(self, t_final=1e3, reltol=1e-12, abstol=1e-64):
        """
        Set parameters for ODE numerical solution with scipy solve_ivp solver.
        Args:
            t_final(float): final integration time [s]. Default to 1000 s.
            reltol(float): relative tolerance. Default to 1e-12.
            abstol(float): absolute tolerance. Default to 1e-64.
        """
        self.ODE_params[0] = reltol
        self.ODE_params[1] = abstol
        self.ODE_params[2] = t_final
        print("Integration time = {}s".format(t_final))
        print("Relative tolerance = {}".format(reltol))
        print("Absolute tolerance = {}".format(abstol))
        return "Changed ODE parameters."

    def get_ODE_params(self):
        """Print ODE parameters used in scipy solver solve_ivp."""
        print("Integration time = {}s".format(self.ODE_params[2]))
        print("Relative tolerance = {}".format(self.ODE_params[0]))
        print("Absolute tolerance = {}".format(self.ODE_params[1]))
        return None

    def conversion(self, reactant, P_in, P_out):
        """
        Returns the conversion of reactant i
        Internal function used for the dynamic CSTR model
        """
        index = self.species_gas.index(reactant)
        X = 1 - (P_out[index] / P_in[index])
        return X

    def selectivity(self, reactant, product, P_in, P_out):
        """
        Returns the selectivity of reactant i to product j
        Internal function used for the dynamic CSTR model
        """
        reactant_index = self.species_gas.index(reactant)
        product_index = self.species_gas.index(product)
        r = reactant_index
        p = product_index
        S = (P_out[p] - P_in[p]) / (P_in[r] - P_out[r])
        return S

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

    @staticmethod
    def methods():
        """Prints all current available MKM methods"""
        functions = ['thermodynamic_consistency_analysis',
                     'kinetic_run',
                     'map_reaction_rate',
                     'apparent_activation_energy',
                     'apparent_activation_energy_local',
                     'apparent_reaction_order',
                     'degree_of_rate_control',
                     'drc_t',
                     'drc_full',
                     'reversibility']
        for method in functions:
            print(method)

    def rxn_network(self):
        """Render reaction network with GraphViz."""
        rn = graphviz.Digraph(name=self.name,
                              comment='Reaction mechanism',
                              format='png',
                              engine='dot')
        rn.attr('graph', nodesep='1.0')
        rn.attr('graph', ratio='1.2')
        rn.attr('graph', dpi='300')
        for species in self.species_sur:
            rn.node(species, color='green', style='bold', fill='red')
        for species in self.species_gas:
            rn.node(species, color='black', style='bold', fill='red')
        for j in range(self.NR):
            a = np.where(self.v_matrix[:, j] < 0)[0]
            b = np.where(self.v_matrix[:, j] > 0)[0]
            if (self.reaction_type[j] == 'ads') or (self.reaction_type[j] == 'des'):
                for i in range(len(a)):
                    for k in range(len(b)):
                        rn.edge(self.species_tot[a[i]],
                                self.species_tot[b[k]])
            else:
                for i in range(len(a)):
                    for k in range(len(b)):
                        rn.edge(self.species_tot[a[i]], self.species_tot[b[k]])
        rn.render(view=True)

    def rxn_profile():
        """
        Returns a plot with the reaction energy profile of the process.
        WORK IN PROGRESS
        """
        pass

    def apply_bep(self,
                  reaction,
                  q,
                  m):
        """
        Function that applies a BEP relation to the selected elementary reaction.

            dh_barrier = q + m * dh_reaction [eV]

        Args:
            reaction(str): reaction string. Ex: "R12"
            q(float): intercept of the line
            m(float): slope of the line
        """
        i = int(reaction[1:]) - 1
        self.dh_barrier[i] = q + m * self.dh_reaction[i]

    def apply_ts_scaling(self,
                         reaction,
                         q,
                         m,
                         initial_state=True):
        """
        Function that applies a TS scaling relation to the selected elementary reaction.
        NB Apply only if in g.mkm the intermediate states species enthalpies are from LSR

            dh_barrier = q + m * dh_ads(initial/final state)        
        Args:
            reaction(str): reaction string. Ex: "R12"
            q(float): intercept of the line
            m(float): slope of the line
            initial_state(bool): if scaling is wrt initial(True) or final(state)
        """
        i = int(reaction[1:]) - 1
        if initial_state:
            ind = list(np.where(
                self.v_matrix[:, i] == -1)[0]) + list(np.where(self.v_matrix[:, i] == -2)[0])
            his = sum([self.h_species[j]*self.v_matrix[j, i]*(-1)
                      for j in ind])
            self.dh_barrier[i] = q + m * his
        else:
            ind = list(np.where(self.v_matrix[:, i] == 1)[
                       0]) + list(np.where(self.v_matrix[:, i] == 2)[0])
            hfs = sum([self.h_species[j]*self.v_matrix[j, i]*(-1)
                      for j in ind])
            self.dh_barrier[i] = q + m * hfs

    def apply_lsr_1d(self,
                     descriptor_name,
                     descriptor_value,
                     scaling_matrix_h,
                     scaling_matrix_ts,
                     bep,
                     initial_state=True):
        """
        Function that builds the whole enthalpy reaction profile
        based on linear scaling relations (LSR).
        Args:
            descriptor_name(str): name of the descriptor (ads. energy of specific species)
            descriptor_value(float): value of the descriptor in eV.
            scaling_matrix_h(ndarray): array with dimension (NC_sur-1)*2.
            scaling_matrix_ts(ndarray): array with dimension NR*2.
        """
        q = scaling_matrix_h[:, 0]
        m = scaling_matrix_h[:, 1]
        for i in range(1, self.NC_sur-1):
            self.h_species[i] = q[i] + m[i] * descriptor_value
        self.h_species[0] = 0.0  # surface
        self.h_species[self.NC_sur:] = 0.0  # gas species
        for i in range(self.NR):
            self.dh_reaction[i] = np.sum(
                self.v_matrix[:, i]*np.array(self.h_species))
        if bep:
            q = scaling_matrix_ts[:, 0]
            m = scaling_matrix_ts[:, 1]
            for j in range(self.NR):
                self.apply_bep("R{}".format(j+1), q[j], m[j])
            self.bep = True
        else:
            q = scaling_matrix_ts[:, 0]
            m = scaling_matrix_ts[:, 1]
            for j in range(self.NR):
                self.apply_ts_scaling("R{}".format(
                    j+1), q[j], m[j], initial_state=initial_state)
            self.bep = False
        self.lsr_mode = True  # possible to compute scaled-DRC
        self.scaling_matrix_h = scaling_matrix_h
        self.scaling_matrix_ts = scaling_matrix_ts

    def thermodynamic_consistency(self, temperature):
        """
        This function evaluates the thermodynamic consistency of the microkinetic 
        model based on the provided energetics and reaction mechanism.
        It compares the equilibrium constants of the global reactions derived from
        a gas-phase thermochemistry database with the equilibrium constant
        extracted by the DFT-derived microkinetic model.        
        Args:
            temperature(float): temperature in Kelvin [K].
        Returns:
            None
        """
        k_h = np.exp(-self.dh_reaction/(K_B*temperature))
        k_s = np.exp(self.ds_reaction/K_B)
        DHR_model = np.zeros(self.NGR)
        DSR_model = np.zeros(self.NGR)
        DGR_model = np.zeros(self.NGR)
        DHR_database = np.zeros(self.NGR)
        DSR_database = np.zeros(self.NGR)
        DGR_database = np.zeros(self.NGR)
        keq_H_model = np.zeros(self.NGR)
        keq_S_model = np.zeros(self.NGR)
        keq_model = np.zeros(self.NGR)
        keq_H_database = np.zeros(self.NGR)
        keq_S_database = np.zeros(self.NGR)
        keq_database = np.zeros(self.NGR) 
        for i in range(self.NGR):
            DHR_model[i] = np.sum(self.dh_reaction * self.stoich_numbers[:, i]) * cf
            DSR_model[i] = np.sum(self.ds_reaction * self.stoich_numbers[:, i]) * cf
            DGR_model[i] = DHR_model[i] - temperature * DSR_model[i] * cf
            DHR_database[i] = reaction_enthalpy(self.gr_string[i], temperature)
            DSR_database[i] = reaction_entropy(self.gr_string[i], temperature)
            DGR_database[i] = DHR_database[i] - temperature * DSR_database[i]
            keq_H_model[i] = np.prod(k_h ** self.stoich_numbers[:, i])
            keq_H_database[i] = k_eq_H(self.gr_string[i], temperature)
            keq_S_model[i] = np.prod(k_s ** self.stoich_numbers[:, i])
            keq_S_database[i] = k_eq_S(self.gr_string[i], temperature)
            keq_model[i] = keq_H_model[i] * keq_S_model[i]
            keq_database[i] = keq_H_database[i] * keq_S_database[i]
        print(" {}: Thermodynamic consistency analysis".format(self.name))
        print(" Temperature = {}K".format(temperature))
        print("")
        print("----------------------------------------------------------------------------------")
        for global_reaction in range(self.NGR):
            print(self.gr_string[global_reaction])
            print("")
            print("Model:    DHR={:0.2e} kJ/mol    DSR={:0.2e} kJ/mol/K     DGR={:0.2e} kJ/mol".format(
                DHR_model[global_reaction], DSR_model[global_reaction], DGR_model[global_reaction]))
            print("Database: DHR={:0.2e} kJ/mol    DSR={:0.2e} kJ/mol/K     DGR={:0.2e} kJ/mol".format(
                DHR_database[global_reaction], DSR_database[global_reaction], DGR_database[global_reaction]))
            print("")
            print("Model:    keqH={:0.2e}    keqS={:0.2e}    Keq={:0.2e}".format(
                keq_H_model[global_reaction], keq_S_model[global_reaction], keq_model[global_reaction]))
            print("Database: keqH={:0.2e}    keqS={:0.2e}    Keq={:0.2e}".format(
                keq_H_database[global_reaction], keq_S_database[global_reaction], keq_database[global_reaction]))
            print(
                "----------------------------------------------------------------------------------")
            print("")
        return None

    def kinetic_coeff(self, temperature, area_active_site=1e-19):
        """
        Returns the kinetic coefficient for the direct and reverse reactions, according to 
        the reaction type (adsorption, desorption or surface reaction) and TST.                
        Args: 
            temperature(float): Temperature in [K].
            A_site_0(float): Area of the catalytic ensemble in [m2]. Default: 1e-19[m2]
        """
        Keq = np.zeros(self.NR)  # Equilibrium constant
        kd = np.zeros(self.NR)   # Direct constant
        kr = np.zeros(self.NR)   # Reverse constant
        for reaction in range(self.NR):
            Keq[reaction] = np.exp(-self.dg_reaction[reaction] /
                                   temperature / K_B)
            if self.reaction_type[reaction] == 'ads':
                A = area_active_site / \
                    (2 * np.pi * self.m[reaction] * K_BU * temperature) ** 0.5
                kd[reaction] = A * \
                    np.exp(-self.dg_barrier[reaction] / K_B / temperature)
                kr[reaction] = kd[reaction] / Keq[reaction]
            elif self.reaction_type[reaction] == 'des':
                kd[reaction] = (K_B * temperature / H) * \
                    np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                kr[reaction] = kd[reaction] / Keq[reaction]
            else:  # Surface reaction
                kd[reaction] = (K_B * temperature / H) * \
                    np.exp(-self.dg_barrier[reaction] / temperature / K_B)
                kr[reaction] = kd[reaction] / Keq[reaction]
        return kd, kr

    def net_rate(self, y, kd, ki):
        """
        Returns the net reaction rate for each elementary reaction.
        Args:
            y(ndarray): surface coverage + partial pressures array [-/Pa].
            kd, ki(ndarray): kinetic constants of the direct/reverse steps.
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
        Returns the rhs of the ODE system.
        Reactor model: differential PFR (zero conversion)
        """
        # Surface species
        dy = self.v_matrix @ self.net_rate(y, kd, ki)
        # Gas species
        dy[self.NC_sur:] = 0.0
        return dy

    def jac_diff(self, time, y, kd, ki):
        """
        Returns Jacobian matrix of the system for
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
        J[self.NC_sur:, :] = 0.0
        return J

    def dynamic_cstr(self, time, y, kd, ki, P_in, temperature):
        """
        Returns the rhs of the ODE system.
        Reactor model: dynamic CSTR
        Args:
            P_in(ndarray): inlet partial pressures of gas species in Pascal [Pa]
            temperature(float): temperature in Kelvin [K]
        """
        # Surface species
        dy = self.v_matrix @ self.net_rate(y, kd, ki)
        # Gas species
        dy[self.NC_sur:] *= (R * temperature/(N_AV * self.CSTR_V))
        dy[self.NC_sur:] *= (self.CSTR_sbet * self.CSTR_mcat/self.CSTR_asite)
        dy[self.NC_sur:] += (P_in - y[self.NC_sur:]) / self.CSTR_tau
        return dy

    def jac_dyn(self, time, y, kd, ki, P_in, temperature):
        """
        Returns the Jacobian matrix of the ODE for the
        dynamic CSTR model.
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
        J[self.NC_sur:, :] *= (R * temperature * self.CSTR_sbet *
                               self.CSTR_mcat) / (N_AV * self.CSTR_V * self.CSTR_asite)
        for i in range(self.NC_gas):
            J[self.NC_sur+i, self.NC_sur+i] -= 1/self.CSTR_tau
        return J

    def __ode_solver_solve_ivp(self, y_0, dy, temperature,
                               reltol, abstol, t_final,
                               end_events=None, jacobian_matrix=None,
                               P_in=None):
        """
        Helper function
        """
        kd, ki = self.kinetic_coeff(temperature)
        if self.reactor_model == 'dynamic':
            args_list = [kd, ki, P_in, temperature]
        else:
            args_list = [kd, ki]
        r = solve_ivp(dy,
                      (0.0, t_final),
                      y_0,
                      method='BDF', 
                      events=end_events,
                      jac=jacobian_matrix,
                      args=args_list,
                      atol=abstol,
                      rtol=reltol)
        return r

    def kinetic_run(self,
                   temperature,
                   pressure,
                   gas_composition,
                   initial_conditions=None,
                   verbose=0,
                   jac=True):
        """
        Simulates a single catalytic run at the desired reaction conditions.        
        Args:
            temperature(float): Temperature of the experiment in Kelvin.
            pressure(float): Total abs. pressure of gaseous species in Pascal.
            gas_composition(list): molar fraction of gas species [-].
            initial_conditions(nparray): initial coverage of the catalyst surface [-].
            verbose(int): 0=print all output; 1=print nothing.        
        Returns:
            Dictionary containing a full report of the virtual catalytic test.        
        """
        if verbose == 0:
            print('{}: Microkinetic run'.format(self.name))
            if self.reactor_model == 'dynamic':
                print('Reactor model: Dynamic CSTR')
            else:
                print('Reactor model: Differential PFR')
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
        y_0 = np.zeros(self.NC_tot)
        if initial_conditions is None:  # Convention: first surface species is the active site
            y_0[0] = 1.0
        else:
            y_0[:self.NC_sur] = initial_conditions[:self.NC_sur]
        y_0[self.NC_sur:] = pressure * np.array(gas_composition)
        if np.sum(y_0[:self.NC_sur]) != 1.0:
            raise ValueError(
                'Wrong initial surface coverage: the sum of the provided coverage is not equal 1.')
        if sum(gas_composition) != 1.0:
            raise ValueError(
                'Wrong gas composition: the sum of the provided molar fractions is not equal to 1.')
        for molar_fraction in gas_composition:
            if (molar_fraction < 0.0) or (molar_fraction > 1.0):
                raise ValueError(
                    'Wrong gas composition: molar fractions must be between 0 and 1.')
        if temperature < 0.0:
            raise ValueError('Wrong temperature (T > 0 K)')
        if pressure < 0.0:
            raise ValueError('Wrong pressure (P > 0 Pa)')
        results_sr = []                      # solve_ivp solver output
        final_sr = []                        # final Value of derivatives
        yfin_sr = np.zeros((self.NC_tot))    # steady-state output [-]
        r_sr = np.zeros((self.NR))           # reaction rate [1/s]
        s_target_sr = np.zeros(1)            # selectivity
        t0 = time.time()
        keys = ['T',
                'P',
                'y_in',
                'theta',
                'ddt',
                'r',
                *['r_{}'.format(i) for i in list(self.grl.keys())],
                'S_{}'.format(self.target_label),
                'MASI',
                'solver']
        r = ['R{}'.format(i+1) for i in range(self.NR)]
        gas_comp_dict = dict(zip(self.species_gas, gas_composition))
        values = [temperature, pressure / 1e5, gas_comp_dict]
        if self.reactor_model == 'dynamic':
            _ = None
            if jac:
                _ = self.jac_dyn
            results_sr = self.__ode_solver_solve_ivp(y_0,
                                                     self.dynamic_cstr,
                                                     temperature,
                                                     *self.ODE_params,
                                                     end_events=None,
                                                     P_in=y_0[self.NC_sur:])  # scipy output
            final_sr = self.dynamic_cstr(results_sr.t[-1],
                                         results_sr.y[:, -1],
                                         *self.kinetic_coeff(temperature),
                                         y_0[self.NC_sur:],
                                         temperature)  # dydt
            yfin_sr = results_sr.y[:self.NC_tot, -1]  # y
            P_in = y_0[self.NC_sur:]
            P_out = yfin_sr[self.NC_sur:]
            reactants = []
            products = []
            conv = []
            for species in self.species_gas:
                index = self.species_gas.index(species)
                if P_in[index] == 0.0:
                    products.append(species)
                else:
                    if species.strip('(g)') in self.inerts:
                        pass
                    else:
                        reactants.append(species)
                        conv.append(self.conversion(species, P_in, P_out))
            Y = np.zeros(self.NGR)
            RR = []
            for reaction in range(self.NGR):
                try:
                    x = self.species_gas.index(
                        self.gr_string[reaction].split()[-3]+'(g)')
                except:
                    x = 0

                RR.append(self.CSTR_Q *
                          (P_out[x] - P_in[x]) / (R * temperature))  # [mol/s]
            s_target_sr = RR[0] / np.sum(RR)
            r_sr = self.net_rate(
                results_sr.y[:, -1], *self.kinetic_coeff(temperature))
            value_masi = max(yfin_sr[:self.NC_sur-1])
            key_masi = self.species_sur[np.argmax(yfin_sr[:self.NC_sur-1])]
            masi_sr = {key_masi: value_masi*100.0}
            keys += ['y_out', 'conversion']
            coverage_dict = dict(zip(self.species_sur, yfin_sr[:self.NC_sur]))
            r_dict = dict(zip(r, r_sr))
            y_gas_out = P_out / np.sum(P_out)
            ddt_dict = dict(zip(self.species_tot, final_sr))
            gas_out = dict(zip(self.species_gas, y_gas_out))
            conv_dict = dict(zip(reactants, conv))
            values += [coverage_dict,
                       ddt_dict,
                       r_dict,
                       *RR,
                       s_target_sr,
                       masi_sr,
                       results_sr,
                       gas_out,
                       conv_dict]
            output_dict = dict(zip(keys, values))
        else:  # differential PFR
            _ = None
            if jac:
                _ = self.jac_diff
            results_sr = self.__ode_solver_solve_ivp(y_0,
                                                     self.differential_pfr,
                                                     temperature,
                                                     *self.ODE_params,
                                                     end_events=None,
                                                     jacobian_matrix=_)
            final_sr = self.differential_pfr(results_sr.t[-1],
                                             results_sr.y[:, -1],
                                             *self.kinetic_coeff(temperature))
            yfin_sr = results_sr.y[:self.NC_sur, -1]
            r_sr = self.net_rate(
                results_sr.y[:, -1], *self.kinetic_coeff(temperature))
            bp = list(set(self.by_products))
            s_target_sr = r_sr[self.target] / \
                (r_sr[self.target] + r_sr[bp].sum())
            value_masi = max(yfin_sr[:self.NC_sur-1])
            key_masi = self.species_sur[np.argmax(yfin_sr[:self.NC_sur-1])]
            masi_sr = {key_masi: value_masi*100.0}
            coverage_dict = dict(zip(self.species_sur, yfin_sr))
            ddt_dict = dict(zip(self.species_tot, final_sr))
            r_dict = dict(zip(r, r_sr))
            values += [coverage_dict,
                       ddt_dict,
                       r_dict,
                       *[r_sr[i] for i in list(self.grl.values())],
                       s_target_sr,
                       masi_sr,
                       results_sr]
            output_dict = dict(zip(keys, values))
        if verbose == 0:
            print('')
            print(
                '{} Reaction Rate: {:0.2e} 1/s'.format(self.target_label, r_sr[self.target]))
            print('{} Selectivity: {:.2f}%'.format(
                self.target_label, s_target_sr*100.0))
            print('Most Abundant Surface Intermediate: {} Coverage: {:.2f}% '.format(
                key_masi, value_masi*100.0))
            print('CPU time: {:.2f} s'.format(time.time() - t0))
        return output_dict

    def map_reaction_rate(self,
                          temp_vector,
                          p_vector,
                          composition,
                          global_reaction_label,
                          initial_conditions=None):
        """
        Wrapper function of single_run to map the reaction rate 
        over the desired temperature/pressure domain.
        Args:
           temp_vector(nparray or list): temperature range of interest [K]
           p_vector(nparray or list): pressure range of interest [Pa]
           composition(list): composition of the gas phase in molar fraction [-]
           global_reaction_rate(str): string of the selected global reaction rate.
                                      Available labels are listed in self.grl
        Returns:
           nparray of size len(temp_vector)*len(p_vector).
           It contains the reaction rate of the called global reaction over
           the selected temperature-pressure domain.
        """
        r_matrix = np.zeros((len(temp_vector), len(p_vector)))
        for i in range(len(temp_vector)):
            for j in range(len(p_vector)):
                run = self.kinetic_run(temp_vector[i],
                                      p_vector[j],
                                      composition,
                                      initial_conditions=initial_conditions,
                                      verbose=0)
                r_matrix[i, j] = list(run['r'].values())[
                    self.grl[global_reaction_label]]
        return r_matrix

    def apparent_activation_energy(self,
                                   temp_range,
                                   pressure,
                                   gas_composition,
                                   global_reaction_label,
                                   initial_conditions=None,
                                   switch=0):
        """
        Function that evaluates the apparent activation energy of the selected global reaction.
        It solves an ODE stiff system for each temperature studied until the steady state convergence.
        From the steady state output, the global reaction rates are evaluated.        
        Args:
            temp_range(list): List containing 3 items: 
                T_range[0]: Lower temperarure range bound [K]
                T_range[1]: Upper temperature range bound [K]
                T_range[2]: Delta of temperature between each point [K]
            pressure(float): Total abs. pressure of the experiment [Pa]
            gas_composition(list): Molar fraction of gas species [-]
            global_reaction_label(str): Label of the global reaction
            initial_conditions(nparray): Array containing initial surface coverage [-]
            switch(int): 0=Print and plot all outputs
                         1=Only print text, no plot generation        
        Returns:
            Apparent activation energies for the selected reaction in kJ/mol.      
        """
        print('{}: Apparent activation energy for {} reaction'.format(
            self.name, global_reaction_label))
        print('')
        print('Temperature range: {}-{}K    Pressure = {:.1f}bar'.format(temp_range[0],
                                                                         temp_range[1] -
                                                                         temp_range[2],
                                                                         pressure/1e5))
        sgas = []
        for i in self.species_gas:
            sgas.append(i.strip('(g)'))
        str_list = ['{}={:.1f}%  '.format(i, j) for i, j in zip(sgas,
                                                                list(np.array(gas_composition)*100.0))]
        gas_string = 'Gas composition: '
        for i in str_list:
            gas_string = gas_string + i
        print(gas_string)
        print('')
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception('Unexisting global process string!')
        temperature_vector = list(
            range(temp_range[0], temp_range[1], temp_range[2]))
        r_ea = np.zeros((len(temperature_vector), 1)
                        )     # production rate [1/s]
        for i in range(len(temperature_vector)):
            t0 = time.time()
            run = self.kinetic_run(temperature_vector[i],
                                  pressure,
                                  gas_composition,
                                  initial_conditions=initial_conditions,
                                  verbose=1)
            r_ea[i] = list(run['r'].values())[self.grl[global_reaction_label]]
            print('Temperature = {}K    CPU Time: {:.2f}s'.format(temperature_vector[i],
                                                                  time.time() - t0))
        eapp, r_squared = calc_eapp(np.asarray(temperature_vector), r_ea)
        keys = ['Tmin',
                'Tmax',
                'N',
                'P',
                'y_gas',
                'Eapp_{}'.format(global_reaction_label),
                'R2']
        gas_comp_dict = dict(zip(self.species_gas, gas_composition))
        values = [temperature_vector[0],
                  temperature_vector[-1],
                  len(temperature_vector),
                  pressure/1e5,
                  gas_comp_dict,
                  eapp,
                  r_squared]
        output_dict = dict(zip(keys, values))
        fig_ea = plt.figure(1, dpi=500)
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
        plt.scatter(1000/np.array(temperature_vector), np.log(r_ea), lw=2)
        m, b = np.polyfit(1000/np.array(temperature_vector), np.log(r_ea), 1)
        plt.plot(1000/np.array(temperature_vector),
                 m*1000/np.array(temperature_vector) + b,
                 lw=1, ls='--')
        plt.grid()
        plt.xlabel('1000/T / $K^{-1}$')
        plt.ylabel('$ln(r_{{{}}})$ / $s^{{-1}}$'.format(global_reaction_label))
        plt.title('{}: {} Apparent Activation Energy'.format(
            self.name, global_reaction_label))
        plt.text(0.65,
                 0.75,
                 '$E_{{app}}$={:.0f} kJ/mol\n$R^2$={:.2f}'.format(
                     eapp, r_squared),
                 transform=fig_ea.transFigure,
                 bbox=dict(facecolor='white', alpha=1.0))
        plt.savefig('{}_Eapp_{}_{}{}K_{}bar.png'.format(self.name,
                                                        global_reaction_label,
                                                        temp_range[0],
                                                        temp_range[1],
                                                        int(pressure/1e5)))
        plt.show()
        return output_dict

    def apparent_activation_energy_local(self,
                                         temperature,
                                         pressure,
                                         gas_composition,
                                         global_reaction_label,
                                         delta_temperature=0.1,
                                         initial_conditions=None,
                                         switch=0):
        """
        Function that evaluates the apparent activation energy of the selected reaction.
        It solves an ODE stiff system for each temperature studied until the steady state convergence.
        From the steady state output, the global reaction rates are evaluated.        
        Args:
            temp_range(list): List containing 3 items: 
                T_range[0]: Lower temperarure range bound [K]
                T_range[1]: Upper temperature range bound [K]
                T_range[2]: Delta of temperature between each point [K]
            pressure(float): Total abs. pressure of the experiment [Pa]
            gas_composition(list): Molar fraction of gas species [-]
            global_reaction_label(str): Label of the global reaction
            initial_conditions(nparray): Array containing initial surface coverage [-]
            switch(int): 0=Print and plot all outputs
                         1=Only print text, no plot generation        
        Returns:
            Apparent activation energies for the selected reaction in kJ/mol.      
        """
        print('{}: Apparent activation energy for {} reaction'.format(self.name,
                                                                      global_reaction_label))
        print('')
        print('Temperature = {}K    Pressure = {:.1f}bar'.format(temperature,
                                                                 pressure/1e5))
        sgas = []
        for i in self.species_gas:
            sgas.append(i.strip('(g)'))
        str_list = ['{}={:.1f}%  '.format(i, j) for i, j in zip(sgas,
                                                                list(np.array(gas_composition)*100.0))]
        gas_string = 'Gas composition: '
        for i in str_list:
            gas_string = gas_string + i
        print(gas_string)
        print('')
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception('Unexisting global process string!')

        temperature_vector = [
            temperature - delta_temperature, temperature + delta_temperature]
        r_ea = np.zeros((len(temperature_vector), 1))
        for i in range(len(temperature_vector)):
            t0 = time.time()
            run = self.kinetic_run(temperature_vector[i],
                                  pressure,
                                  gas_composition,
                                  initial_conditions=initial_conditions,
                                  verbose=1)
            r_ea[i] = list(run['r'].values())[self.grl[global_reaction_label]]
            print('Temperature = {}K    CPU Time: {:.2f}s'.format(temperature_vector[i],
                                                                  time.time() - t0))
        eapp = (R / 1000.0) * temperature**2 * \
            (np.log(r_ea[1]) - np.log(r_ea[0])) / (2*delta_temperature)
        keys = []
        values = []
        output_dict = dict(zip(keys, values))
        return eapp[0]

    def apparent_reaction_order(self,
                                temperature,
                                pressure,
                                composition_matrix,
                                species_label,
                                global_reaction_label,
                                initial_conditions=None,
                                switch=0):
        """
        Args:
            temperature(float): Temperature of the experiment [K]
            pressure(float): Total pressure of the experiment [Pa]
            composition_matrix(nparray): Matrix containing gas composition at each run.
                                         Dimension in Nruns*NC_gas, where Nruns is the number
                                         of experiments with different composition
            species_label(str): Gas species for which the apparent reaction order is computed
            global_reaction_label(str): Selected global reaction
            initial_conditions(nparray): Array containing initial surface coverage [-]
            switch(int): 0=Print and plot all outputs
                         1=Only print text, no plot generation                            
        Returns:
            Apparent reaction order of the selected species for the selected global reaction.        
        """
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception('unexisting global process string!')

        if species_label+'(g)' not in self.species_gas:
            raise Exception('Undefined gas species!')

        index = self.species_gas.index(species_label+'(g)')
        for i in composition_matrix[:, index]:
            if i == 0.0:
                raise ValueError(
                    'Provide non-zero molar fraction for the gaseous species for which the apparent reaction order is computed.')

        n_runs = composition_matrix.shape[0]
        r_napp = np.zeros((n_runs, 1))    # Reaction rate [1/s]

        print('{}: {} Apparent reaction order for {} reaction'.format(self.name,
                                                                      species_label,
                                                                      global_reaction_label))
        print('')
        print('Temperature = {}K    Pressure = {:.1f}bar'.format(
            temperature, pressure/1e5))
        print('')
        for i in range(n_runs):
            t0 = time.time()
            run = self.kinetic_run(temperature,
                                  pressure,
                                  composition_matrix[i, :],
                                  initial_conditions=initial_conditions,
                                  verbose=1)
            r_napp[i] = list(run['r'].values())[
                self.grl[global_reaction_label]]
            print('y_{} = {:.2f}    CPU Time: {:.2f}s'.format(species_label,
                                                              composition_matrix[i,
                                                                                 index],
                                                              time.time() - t0))

        napp, r_squared = calc_reac_order(
            pressure*composition_matrix[:, index], r_napp)
        keys = ['T',
                'P',
                'N',
                'y_{}'.format(species_label),
                'r_{}'.format(global_reaction_label),
                'napp_{}'.format(species_label),
                'R2']
        values = [temperature,
                  pressure/1e5,
                  n_runs,
                  composition_matrix[:, index],
                  r_napp,
                  napp,
                  r_squared]
        output_dict = dict(zip(keys, values))
        fig_na = plt.figure(2, dpi=500)
        plt.scatter(
            np.log(pressure*composition_matrix[:, index]), np.log(r_napp), lw=2)
        m, b = np.polyfit(
            np.log(pressure*composition_matrix[:, index]), np.log(r_napp), 1)
        plt.plot(np.log(pressure*composition_matrix[:, index]),
                 m*np.log(pressure*composition_matrix[:, index]) + b,
                 lw=1, ls='--')
        plt.grid()
        plt.xlabel('ln($P_{{{}}}$ / Pa)'.format(species_label))
        plt.ylabel('ln($r_{{{}}}$) / $s^{{-1}}$'.format(global_reaction_label))
        plt.title('{}: {} apparent reaction order for {}'.format(self.name,
                                                                 species_label,
                                                                 global_reaction_label))
        plt.text(0.75, 0.75,
                 '$n_{{app}}$={:.2f}\n$R^2={:.2f}$'.format(napp, r_squared),
                 transform=fig_na.transFigure,
                 bbox=dict(facecolor='white', alpha=1.0))
        plt.savefig('{}_napp_{}_{}_{}K_{}bar.png'.format(self.name,
                                                         global_reaction_label,
                                                         species_label,
                                                         temperature,
                                                         int(pressure/1e5)))
        plt.show()
        return output_dict

    def degree_of_rate_control(self,
                               temperature,
                               pressure,
                               gas_composition,
                               global_reaction_label,
                               ts_int_label,
                               initial_conditions=None,
                               dg=1.0E-6,
                               verbose=0):
        """
        Calculates the degree of rate control(DRC) and selectivity control(DSC)
        for the selected transition state or intermediate species.        
        Args:
            temperature(float): Temperature of the experiment [K]
            pressure(float): Partial pressure of the gaseous species [Pa]
            gas_composition(list): Molar fraction of the gaseous species [-]
            global_reaction_label(str): Global reaction for which DRC/DSC are computed
            ts_int_label(str): Transition state/surface intermediate for which DRC/DSC are computed
            initial_conditions(nparray): Initial surface coverage [-]
            dg(float): Deviation applied to selected Gibbs energy to calculate the DRC/DSC values.
                       Default=1E-6 eV
            verbose(int): 1= Print essential info
                          0= Print additional info                       
        Returns:
            List with DRC and DSC of TS/intermediate for the selected reaction [-] 
        """
        if (global_reaction_label != self.target_label) and (global_reaction_label not in self.by_products_label):
            raise Exception(
                'Reaction label must be related to a global reaction!')
        if dg > 0.1:
            raise Exception(
                'Define a dg < 0.1 eV: the lower the value, the more accurate the DRC/DSC values.')
        switch_ts_int = 0  # Switch elementary step/intermediate
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
                    mk_object = MKM('i', 
                                    self.input_rm,
                                    self.input_g,
                                    t_ref=self.t_ref,
                                    reactor=self.reactor_model,
                                    inerts=self.inerts)
                    if mk_object.reactor_model == 'dynamic':
                        mk_object.set_CSTR_params(volume=self.CSTR_V,
                                              Q=self.CSTR_Q,
                                              m_cat=self.CSTR_mcat,
                                              S_BET=self.CSTR_sbet,
                                              verbose=1)
                    mk_object.dg_barrier[index] += dg*(-1)**(i)
                    mk_object.dg_barrier_rev[index] += dg*(-1)**(i)
                    run = mk_object.kinetic_run(temperature,
                                               pressure,
                                               gas_composition,
                                               initial_conditions=initial_conditions,
                                               verbose=1)
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
                    mk_object = MKM('i',
                                    self.input_rm,
                                    self.input_g,
                                    t_ref=self.t_ref,
                                    reactor=self.reactor_model,
                                    inerts=self.inerts)
                    if mk_object.reactor_model == 'dynamic':
                        mk_object.set_CSTR_params(volume=self.CSTR_V,
                                                  Q=self.CSTR_Q,
                                                  m_cat=self.CSTR_mcat,
                                                  S_BET=self.CSTR_sbet,
                                                  verbose=1)
                    if mk_object.dg_reaction[index] < 0.0:
                        mk_object.dg_barrier[index] = dg * i
                        mk_object.dg_barrier_rev[index] += dg * i
                    else:
                        mk_object.dg_barrier[index] = mk_object.dg_reaction[index] + dg * i
                    run = mk_object.kinetic_run(temperature,
                                               pressure,
                                               gas_composition,
                                               initial_conditions=initial_conditions,
                                               verbose=1)
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
                mk_object = MKM('i',
                                self.input_rm,
                                self.input_g,
                                t_ref=self.t_ref,
                                reactor=self.reactor_model,
                                inerts=self.inerts)
                if mk_object.reactor_model == 'dynamic':
                    mk_object.set_CSTR_params(volume=self.CSTR_V,
                                          Q=self.CSTR_Q,
                                          m_cat=self.CSTR_mcat,
                                          S_BET=self.CSTR_sbet,
                                          verbose=1)
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
                run = mk_object.kinetic_run(temperature,
                                           pressure,
                                           gas_composition,
                                           initial_conditions=initial_conditions,
                                           verbose=1)
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

    def drc_t(self,
              temp_vector,
              pressure,
              gas_composition,
              global_reaction_label,
              ts_int_label,
              initial_conditions=None,
              dg=1.0E-6,
              verbose=1):
        """
        Calculates the degree of rate control(DRC) and selectivity control(DSC)
        for the selected transition states or intermediate species as function of temperature.        
        Args:
            temp_vector(nparray): Temperature vector [K]
            pressure(float): Partial pressure of the gaseous species [Pa]
            gas_composition(list): Molar fraction of the gaseous species [-]
            global_reaction_label(str): Global reaction for which DRC/DSC are computed
            ts_int_label(str or list of str): Transition state/surface intermediate for which DRC/DSC are computed
            initial_conditions(nparray): Initial surface coverage [-]
            dg(float): Deviation applied to selected Gibbs energy to calculate the DRC/DSC values.
                       Default=1E-6 eV
            verbose(int): 1= Print essential info
                          0= Print additional info                       
        Returns:
            List with DRC and DSC of TS/intermediate for the selected reaction [-] 
        """
        if (type(ts_int_label) == list):  # multiple species/ts
            drc_array = np.zeros((len(temp_vector), len(ts_int_label)))
            dsc_array = np.zeros((len(temp_vector), len(ts_int_label)))
            for i in range(len(temp_vector)):
                for j in range(len(ts_int_label)):
                    drc_array[i, j], dsc_array[i, j] = self.degree_of_rate_control(temp_vector[i],
                                                                                   pressure,
                                                                                   gas_composition,
                                                                                   global_reaction_label,
                                                                                   ts_int_label[j],
                                                                                   initial_conditions=initial_conditions,
                                                                                   verbose=verbose,
                                                                                   dg=dg)
            drsc_temp = np.concatenate((np.array([temp_vector]).T,
                                        drc_array, dsc_array), axis=1)
            col = ['T[K]']
            for i in range(len(ts_int_label)):
                col.append('DRC_{}'.format(ts_int_label[i]))
            for i in range(len(ts_int_label)):
                col.append('DSC_{}'.format(ts_int_label[i]))

            df_drsc_temp = pd.DataFrame(np.round(drsc_temp, decimals=2),
                                        columns=col)
            fig = plt.figure(dpi=500)
            for i in range(len(ts_int_label)):
                plt.plot(temp_vector, drc_array[:, i], label=ts_int_label[i])
            plt.grid()
            plt.legend()
            plt.xlabel('Temperature / K')
            plt.ylabel('DRC')
            plt.ylim([0.0, 1.0])
            title = "-".join(ts_int_label)
            plt.title(title)
            plt.savefig('{}_drc_{}_{}_{}{}K_{}bar.png'.format(self.name,
                                                              global_reaction_label,
                                                              title,
                                                              temp_vector[0],
                                                              temp_vector[-1],
                                                              int(pressure/1e5)))
            plt.show()
            return df_drsc_temp
        else:  # single species/ts
            drc_array = np.zeros(len(temp_vector))
            dsc_array = np.zeros(len(temp_vector))
            for i in range(len(temp_vector)):
                drc_array[i], dsc_array[i] = self.degree_of_rate_control(temp_vector[i],
                                                                         pressure,
                                                                         gas_composition,
                                                                         global_reaction_label,
                                                                         ts_int_label,
                                                                         verbose=1)
            drsc_temp = np.concatenate((np.array([temp_vector]).T,
                                        np.array([drc_array]).T,
                                        np.array([dsc_array]).T),
                                       axis=1)
            df_drsc_temp = pd.DataFrame(np.round(drsc_temp, decimals=2),
                                        columns=['T[K]', 'DRC', 'DSC'])
            fig = plt.figure(dpi=400)
            plt.plot(temp_vector, drc_array)
            plt.grid()
            plt.xlabel('Temperature / K')
            plt.ylabel('DRC')
            plt.ylim([0, 1])
            plt.title('{}'.format(ts_int_label))
            plt.savefig('{}_drc_{}_{}_{}{}K_{}bar.png'.format(self.name,
                                                              global_reaction_label,
                                                              ts_int_label,
                                                              temp_vector[0],
                                                              temp_vector[-1],
                                                              int(pressure/1e5)))
            plt.show()
            return df_drsc_temp

    def drc_full(self,
                 temperature,
                 pressure,
                 gas_composition,
                 global_reaction_label,
                 initial_conditions=None,
                 dg=1.0E-6):
        """
        Wrapper function that calculates the degree of rate control of all
        intermediates and transition states at the desired conditions.        
        Args:
            temperature(float): Temperature of the experiment [K]
            pressure(float): Partial pressure of the gaseous species [Pa]
            gas_composition(nparray): Molar fraction of the gaseous species [-]
            global_reaction_label(str): Global reaction for which all DRC/DSC are computed
            initial_conditions(nparray): Initial surface coverage [-]
            dg(float): Applied perturbation to the Gibbs energy of the TS/intermediates.
                       Default=1E-6 eV                       
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
            drc_ts[reaction], dsc_ts[reaction] = self.degree_of_rate_control(temperature,
                                                                             pressure,
                                                                             gas_composition,
                                                                             global_reaction_label,
                                                                             'R{}'.format(reaction+1),
                                                                             verbose=1,
                                                                             initial_conditions=initial_conditions)
        for species in range(self.NC_sur):
            print('')
            print('{}'.format(self.species_sur[species]))
            drc_int[species], dsc_int[species] = self.degree_of_rate_control(temperature,
                                                                             pressure,
                                                                             gas_composition,
                                                                             global_reaction_label,
                                                                             self.species_sur[species],
                                                                             verbose=1,
                                                                             initial_conditions=initial_conditions)
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

        def style_significant(v, props=''):
            return props if abs(v) >= 0.01 else None

        df_drsc_ts = df_drsc_ts.style.applymap(style_significant, props='color:red;').applymap(
            lambda v: 'opacity: 50%;' if abs(v) < 0.01 else None)
        df_drsc_ts.format({'DRC': '{:,.2f}'.format, 'DSC': '{:,.2f}'.format})
        drsc_int = np.concatenate((np.array([drc_int]).T,
                                   np.array([dsc_int]).T),
                                  axis=1)
        df_drsc_int = pd.DataFrame(np.round(drsc_int, decimals=2),
                                   index=self.species_sur,
                                   columns=['DRC', 'DSC'])
        df_drsc_int.to_csv("X_{}_{}_{}_int.csv".format(global_reaction_label,
                                                       int(temperature),
                                                       int(pressure/1e5)))
        df_drsc_int = df_drsc_int.style.applymap(style_significant, props='color:red;').applymap(
            lambda v: 'opacity: 50%;' if abs(v) < 0.01 else None)
        df_drsc_int.format({'DRC': '{:,.2f}'.format, 'DSC': '{:,.2f}'.format})
        return df_drsc_ts, df_drsc_int

    def reversibility(self,
                      temperature,
                      pressure,
                      gas_composition,
                      initial_conditions=None):
        """
        Function that provides the reversibility of all elementary reaction at the desired
        reaction conditions.

        Args:
            temperature(float): Temperature in [K]
            pressure(float): Pressure in [Pa]
            gas_composition(list): Gas species molar fractions [-]
            initial_conditions(nparray): Initial surface coverage [-]        
        Returns:
            List containing reversibility of all elementary reactions [-]        
        """
        print("{}: Reversibility analysis".format(self.name))
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
        run = self.kinetic_run(temperature,
                              pressure,
                              gas_composition,
                              initial_conditions=initial_conditions,
                              verbose=1)
        k = self.kinetic_coeff(temperature)
        composition_ss = list(run['theta'].values())
        for i in range(self.NC_gas):
            composition_ss.append(pressure*gas_composition[i])
        reversibility = z_calc(composition_ss, *k, self.v_f, self.v_b)
        r = []
        for i in range(self.NR):
            r.append('R{}'.format(i+1))
        df_reversibility = pd.DataFrame(np.round(np.array([reversibility]).T, decimals=2),
                                        index=r,
                                        columns=['Reversibility [-]'])

        def style_significant(v, props=''):
            return props if ((v >= 0.01) and (v <= 0.99)) else None
        df_reversibility.to_csv("Z_{}_{}_{}.csv".format(
            self.name, int(temperature), int(pressure/1e5)))
        df_reversibility = df_reversibility.style.applymap(
            style_significant, props='color:red;')
        df_reversibility.format({'Reversibility [-]': '{:,.2f}'.format})
        return df_reversibility