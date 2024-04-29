#!/usr/bin/env python
# coding: utf-8

# # ElectroMKM class import and object instantiation

# In[1]:


import sys
sys.path.insert(0, "../") 
from electromkm import electroMKM

f=open('input','r')
lines=f.readlines()
f.close()

potential=float(lines[0].split()[1])
pH=float(lines[1].split()[1])
conc=float(lines[2].split()[1])
cation_conc=float(lines[3].split()[1])

# The demo system deal with the Hydrogen Evolution Reaction (HER) with random values.

# In[2]:


model = electroMKM('CO2R',
                   'rm.mkm', 
                   'g.mkm', 
                    t_ref=298)


# # Model exploration
# 

# To investigate the characteristics of the system under study, several attributes can be easily inspected to check general information like number of elementary reactions, energetics, reaction network, etc.

# In[3]:


print(model.species_gas)


model.set_ODE_params(t_final=1000)



# In[11]:


import numpy as np
array=[]
for i in model.species_gas:
    if i=='C1O2(g)':
        array.append(conc)
    elif i=='H2O(g)':
        array.append(1)
    else:
        array.append(0)
array=np.array(array)
print(array)
exp = model.kinetic_run(potential, pH, cation_conc, gas_composition = array , jac = True)


# In[12]:


exp['ddt']


# Once steady state conditions have been checked, the solution can be easily analyzed. the main output consists of steady state surface coverage and reaction rate in term of current density.

# In[13]:


exp['theta']


# In[14]:


exp['r']


# Negative current density means reduction is occurring, while positive values means that reaction is evolving in the opposite direction. Values of current density are stored in mA cm-2.

# In[15]:


exp['j_C2C']

target_label=model.target_label

with open('output', 'w') as outfile:
    outfile.write('Current Density (mA/cm2) of {} {:.4e}\n'.format(target_label,exp[str('j_'+target_label)]/10))
    for label in model.by_products_label:
        outfile.write('Current Density (mA/cm2) of {} {:.4e}'.format(label,exp[str('j_'+label)]/10))
