#Full simulation of a basic excitatory-inhibitory feedback loop with cyclic spiking through constant voltage injection.
#Reset of membrane potential occurs from the inhibitory and excitatory neuron interation.
import numpy as np
import matplotlib.pyplot as plt

#Constants 
Erest = -65e-3
Eexc = -50e-3
Einh = -70e-3
Vth_base = -35e-3 #-25e-3 - Adjust to change the frequency
taue = 0.003 #0.003
taui = 0.001 #0.001
tauge = 0.002 #0.009
taugi = 0.001 #0.005
Vinj = 0.04 #0.04
wExc = 8 #5
wInh = 5 #3
dt = 0.001
num_steps = 500

#Array setup
Vexc = np.zeros(num_steps)
Vinh = np.zeros(num_steps)
ge = np.zeros(num_steps)
gi = np.zeros(num_steps)
SpikeExc = np.zeros(num_steps)
SpikeInh = np.zeros(num_steps)
Vth_exc = np.full(num_steps, Vth_base)

Vexc[0] = Erest
Vinh[0] = Erest


#Simulation from 0 to num_steps
for i in range(0, num_steps):
    #Synaptic conductance updates
    ge[i] = ge[i-1] + dt * ((-ge[i-1] + wExc * SpikeExc[i-1]) / tauge)
    gi[i] = gi[i-1] + dt * ((-gi[i-1] + wInh * SpikeInh[i-1]) / taugi)

    #Membrane potential updates
    Vexc[i] = Vexc[i-1] + dt * ((Erest - Vexc[i-1]) + gi[i] * (Einh - Vexc[i-1]) + Vinj) / taue
    Vinh[i] = Vinh[i-1] + dt * ((Erest - Vinh[i-1]) + ge[i] * (Eexc - Vinh[i-1])) / taui

    #Spike detection
    if Vexc[i] >= Vth_exc[i-1]:
        SpikeExc[i] = 1
    if Vinh[i] >= Vth_exc[i-1]:
        SpikeInh[i] = 1

#CScale values for ease of demonstration 
Vexc_mV = Vexc * 1000 
Vinh_mV = Vinh * 1000 
ge_scaled = ge * 1000 
gi_scaled = gi * 1000 

#Plot results
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axs[0].plot(Vexc_mV, label='Vexc (mV)', color='black')
axs[0].legend()
axs[0].set_ylabel("Vexc (mV)")

axs[1].plot(Vinh_mV, label='Vinh (mV)', color='red')
axs[1].legend()
axs[1].set_ylabel("Vinh (mV)")

axs[2].plot(ge_scaled, label='ge (scaled)', color='blue')
axs[2].legend()
axs[2].set_ylabel("ge (x1000)")

axs[3].plot(gi_scaled, label='gi (scaled)', color='purple')
axs[3].legend()
axs[3].set_ylabel("gi (x1000)")
axs[3].set_xlabel("Timestep")

plt.tight_layout()
plt.show()


#Create raster plots of excitatory and inhibitory spikes
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

#Excitatory neuron raster plot
spike_times_exc = np.where(SpikeExc == 1)[0]
axes[0].scatter(spike_times_exc, np.full_like(spike_times_exc, 0), s=10, color='black')
axes[0].set_title("Excitatory Neuron - Spike Raster")
axes[0].set_ylabel("Neuron Index")

#Inhibitory neuron raster plot
spike_times_inh = np.where(SpikeInh == 1)[0]
axes[1].scatter(spike_times_inh, np.full_like(spike_times_inh, 0), s=10, color='red')
axes[1].set_title("Inhibitory Neuron - Spike Raster")
axes[1].set_ylabel("Neuron Index")
axes[1].set_xlabel("Timestep")

plt.tight_layout()
plt.show()




