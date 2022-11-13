# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 10:58:20 2022

@author: 20215
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

c1 = pd.read_csv('serie_11.csv') # <=====
c_1 = c1.dropna()
c2 = pd.read_csv('serie_11_2.csv') # <=====
c_2 = c2.dropna()

time_1 = c_1['Tiempo (s)'].tolist()
time_2 = c_2['Tiempo (s)'].tolist()
position_1 = c_1['Posición (m)'].tolist()
velocity_1 = c_1['Vector velocidad (m/s)'].tolist()
aceleration_1 =c_1['Aceleración (m/s²)'].tolist()

position_2 = c_2['Posición (m)'].tolist()
velocity_2 = c_2['Vector velocidad (m/s)'].tolist()
aceleration_2 = c_2['Aceleración (m/s²)'].tolist()

t1, t2 = np.array(time_1, dtype=np.float64), np.array(time_2, dtype=np.float64)
p1, p2 = np.array(position_1, dtype=np.float64), np.array(position_2, dtype=np.float64)
v1, v2 = np.array(velocity_1, dtype=np.float64), np.array(velocity_2, dtype=np.float64)
a1, a2 = np.array(aceleration_1, dtype=np.float64), np.array(aceleration_2, dtype=np.float64)

# Plot
plt.style.use(['science', 'notebook', 'grid']) # 'dark_background'

fig, axes = plt.subplots(1, 3, figsize=(21, 5))
ax = axes[0] # Position
ax.plot(t1, p1, color='blue', marker='o', linestyle='--', alpha=1, label='Carro 1')
ax.plot(t2, p2, color='green', marker='^', linestyle='--', alpha=1, label='Carro 2')
ax.set_title('Posición')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Posición [m]')
ax.legend(fancybox=False, edgecolor='black')
ax = axes[1] # Velocity
ax.plot(t1, v1, color='orange', marker='o', linestyle='--', alpha=1, label='Carro 1')
ax.plot(t2, v2, color='red', marker='^', linestyle='--', alpha=1, label='Carro 2')
ax.set_title('Velocidad')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Velocidad [m/s]')
ax.legend(fancybox=False, edgecolor='black')
ax = axes[2] # Aceleration
ax.plot(t1, a1, color='c', marker='o', linestyle='--', alpha=1, label='Carro 1')
ax.plot(t2, a2, color='purple', marker='^', linestyle='--', alpha=1, label='Carro 2')
ax.set_title('Aceleración')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Aceleracion [m/s$^2$]')
ax.legend(fancybox=False, edgecolor='black')
#plt.savefig('serie_11_plot.png', dpi=200)
plt.show()

########## Before Colision ##########
##### Position Analysis #####
# Measure of position
# Carro 1
pb1b = p1[0:21] # <=====
pmax1b, pmin1b = np.max(pb1b), np.min(pb1b)
d_1b = pmax1b - pmin1b
# Carro 2
pb2b = p2[2:22] # <=====
pmax2b, pmin2b = np.max(pb2b), np.min(pb2b)
d_1b= pmax1b - pmin1b
d_2b = pmax2b - pmin2b

# Measure of time
# Carro 1
tb1b = t1[0:21] # <=====
t1maxb, t1minb = np.max(tb1b), np.min(tb1b)
t_1b = t1maxb - t1minb
pmedb = (t1maxb + t1minb)/2
# Carro 2
tb2b = t2[2:22] # <=====
t2maxb, t2minb = np.max(tb2b), np.min(tb2b)
t_2b = t2maxb - t2minb
pmedb = (t2maxb + t2minb)/2

textstr_1 = '\n'.join((
    r'$\Delta x_1=%.3f$' % (d_1b),
    r'$\Delta t_1=%.3f$' % (t_1b),
    r'$\Delta x_2=%.3f$' % (d_2b),
    r'$\Delta t_2=%.3f$' % (t_2b)
    ))

##### Velocity analysis #####
x1ib, x1fb = 4, 18 # <=====
x2ib, x2fb = 4, 19 # <=====
v_b1b = v1[x1ib:x1fb]
v_b2b = v2[x2ib:x2fb]
t_b1b = t1[4:18] # <=====
t_b2b = t2[4:19] # <=====


# measures of central tendency 1 
mean_1b = np.mean(v_b1b)
median_1b = np.median(v_b1b)

range1 = np.ptp(v_b1b)
variance1 = np.var(v_b1b)
sd1 = np.std(v_b1b)

# measures of central tendency 2
mean_2b = np.mean(v_b2b)
median_2b = np.median(v_b2b)

range2 = np.ptp(v_b2b)
variance2 = np.var(v_b2b)
sd2 = np.std(v_b2b)

textstr_2 = '\n'.join((
    r'$\mu v_1=%.3f$' % (mean_1b),
    r'$range_1 =%.3f$' % (range1),
    r'$variance_1 =%.3f$' % (variance1),
    r'$\sigma_1 =%.3f$' % (sd1),
    r'$\mu v_2=%.3f$' % (mean_2b),
    r'$range_2 =%.3f$' % (range2),
    r'$variance_2 =%.3f$' % (variance2),
    r'$\sigma_2 =%.3f$' % (sd2)
    ))

def const(size, value):
    requiredlist = [value]*size
    return requiredlist
size = len(v_b1b)
value = mean_1b
v1_new = const(size, value)

def const(size, value):
    requiredlist = [value]*size
    return requiredlist
size = len(v_b2b)
value = mean_2b
v2_new = const(size, value)


##### Plot Before Colision #####
fig, axes = plt.subplots(1, 3, figsize=(21, 5))
ax = axes[0] # Position
ax.plot(t1, p1, color='blue', marker='o', linestyle='--', alpha=1, label='Carro 1')
ax.plot(t2, p2, color='green', marker='^', linestyle='--', alpha=1, label='Carro 2')
ax.axvspan(t1minb, t1maxb, color='lightgrey', alpha=0.7) 
ax.text(1.5, 0.3, textstr_1,  ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black')) # <=====
ax.text(pmedb, 0.44, 'Antes de colisión', ha='center', va='center', fontsize=12, color='grey') # <=====
ax.set_title('Posición')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Posición [m]')
ax.legend(fancybox=False, edgecolor='black')
ax = axes[1] # Velocity
ax.plot(t1, v1, color='orange', marker='o', linestyle='--', alpha=1, label='Carro 1')
ax.plot(t_b1b, v1_new, color='lime')
ax.plot(t2, v2, color='red', marker='^', linestyle='--', alpha=1, label='Carro 2')
ax.plot(t_b2b, v2_new, color='lime')
ax.axvspan(t1minb, t1maxb, color='lightgrey', alpha=0.7) 
ax.text(0.7, 0.1, textstr_2,  ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black')) # <=====
ax.text(pmedb, 0.24, 'Antes de colisión', ha='center', va='center', fontsize=12, color='grey') # <=====
ax.set_title('Velocidad')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Velocidad [m/s]')
ax.legend(fancybox=False, edgecolor='black')
ax = axes[2] # Aceleration
ax.plot(t1, a1, color='c', marker='o', linestyle='--', alpha=1, label='Carro 1')
ax.plot(t2, a2, color='purple', marker='^', linestyle='--', alpha=1, label='Carro 2')
ax.axvspan(t1minb, t1maxb, color='lightgrey', alpha=0.7)
ax.text(pmedb, -2, 'Antes de colisión', ha='center', va='center', fontsize=12, color='grey') # <=====
ax.set_title('Aceleración')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Aceleracion [m/s$^2$]')
ax.legend(fancybox=False, edgecolor='black')
#plt.savefig('serie_11_before_plot.png', dpi=200)
plt.show()


######### After Colision #########
##### Position Analysis #####
# Measure of position
# Carro 1
pb1a = p1[20:36] # <=====
pmax1a, pmin1a = np.max(pb1a), np.min(pb1a)
d_1a = pmin1a - pmax1a
# Carro 2
pb2a = p2[21:36] # <=====
pmax2a, pmin2a = np.max(pb2a), np.min(pb2a)
d_2a = pmin2a - pmax2a

# Measure of time
# Carro 1
tb1a = t1[20:36] # <=====
t1maxa, t1mina = np.max(tb1a), np.min(tb1a)
t_1a = t1maxa - t1mina
pmeda = (t1maxa + t1mina)/2
# Carro 2
tb2a = t2[21:36] # <=====
t2maxa, t2mina = np.max(tb2a), np.min(tb2a)
t_2a = t2maxa - t2mina
pmeda = (t2maxa + t2mina)/2

textstr_3 = '\n'.join((
    r'$\Delta x_1=%.3f$' % (d_1a),
    r'$\Delta t_1=%.3f$' % (t_1a),
    r'$\Delta x_2=%.3f$' % (d_2a),
    r'$\Delta t_2=%.3f$' % (t_2a)
    ))

##### Velocity analysis #####
x1ia, x1fa = 21, 36 # <=====
x2ia, x2fa = 21, 36 # <=====
v_b1a = v1[x1ia:x1fa]
v_b2a = v2[x2ia:x2fa]
t_b1a = t1[x1ia:x2fa]
t_b2a = t2[x1ia:x2fa]

tb1av = t1[21:36]
tb2av = t1[21:36]

# measures of central tendency 1 
mean_1a = np.mean(v_b1a)
median_1a = np.median(v_b1a)

range1 = np.ptp(v_b1a)
variance1 = np.var(v_b1a)
sd1 = np.std(v_b1a)

# measures of central tendency 2
mean_2a = np.mean(v_b2a)
median_2a = np.median(v_b2a)

range2 = np.ptp(v_b2a)
variance2 = np.var(v_b2a)
sd2 = np.std(v_b2a)

textstr_4 = '\n'.join((
    r'$\mu v_1=%.3f$' % (mean_1a),
    r'$range_1 =%.3f$' % (range1),
    r'$variance_1 =%.3f$' % (variance1),
    r'$\sigma_1 =%.3f$' % (sd1),
    r'$\mu v_2=%.3f$' % (mean_2a),
    r'$range_2 =%.3f$' % (range2),
    r'$variance_2 =%.3f$' % (variance2),
    r'$\sigma_2 =%.3f$' % (sd2)
    ))

def const(size, value):
    requiredlist = [value]*size
    return requiredlist
size = len(v_b1a)
value = mean_1a
v1_new = const(size, value)

def const(size, value):
    requiredlist = [value]*size
    return requiredlist
size = len(v_b2a)
value = mean_2a
v2_new = const(size, value)

##### Plot After Colision #####
plt.style.use(['science', 'notebook', 'grid']) # 'dark_background'

fig, axes = plt.subplots(1, 3, figsize=(21, 5))
ax = axes[0] # Position
ax.plot(t1, p1, color='blue', marker='o', linestyle='--', alpha=1, label='Carro 1')
ax.plot(t2, p2, color='green', marker='^', linestyle='--', alpha=1, label='Carro 2')
ax.axvspan(t1mina, t1maxa, color='lightgrey', alpha=0.7) 
ax.text(1.5, 0.25, textstr_3,  ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black')) # <=====
ax.text(pmeda, 0.35, 'Después de colisión', ha='center', va='center', fontsize=12, color='grey') # <=====
ax.set_title('Posición')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Posición [m]')
ax.legend(fancybox=False, edgecolor='black')
ax = axes[1] # Velocity
ax.plot(t1, v1, color='orange', marker='o', linestyle='--', alpha=1, label='Carro 1')
ax.plot(tb1av, v1_new, color='lime')
ax.plot(t2, v2, color='red', marker='^', linestyle='--', alpha=1, label='Carro 2')
ax.plot(tb2av, v2_new, color='lime')
ax.axvspan(t1mina, t1maxa, color='lightgrey', alpha=0.7) 
ax.text(0.7, 0.1, textstr_4,  ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black')) # <=====
ax.text(pmeda, 0.2, 'Después de colisión', ha='center', va='center', fontsize=12, color='grey') # <=====
ax.set_title('Velocidad')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Velocidad [m/s]')
ax.legend(fancybox=False, edgecolor='black')
ax = axes[2] # Aceleration
ax.plot(t1, a1, color='c', marker='o', linestyle='--', alpha=1, label='Carro 1')
ax.plot(t2, a2, color='purple', marker='^', linestyle='--', alpha=1, label='Carro 2')
ax.axvspan(t1mina, t1maxa, color='lightgrey', alpha=0.7)
ax.text(pmeda, 2, 'Después de colisión', ha='center', va='center', fontsize=12, color='grey') # <=====
ax.set_title('Aceleración')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Aceleracion [m/s$^2$]')
ax.legend(fancybox=False, edgecolor='black')
#plt.savefig('serie_11_after_plot.png', dpi=200)
plt.show()