# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:27:36 2020

@author: mattb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def simple_pulse(time, onset, amplitude, risetime, decaytime):
    ''' pulse model function to work with numpy.'''
    pulse = np.exp(-(time - onset) / risetime) - np.exp(-(time - onset) / decaytime)
    pulse[np.where(time < onset)] = 0.0
    return -amplitude * pulse

def oscillation(length, amplitude, freq, decaytime):
    ''' oscillation model function for numpy '''
    t = np.linspace(0, length-1, length)
    return amplitude * np.sin(2 * np.pi * freq * t) * np.exp(-t / decaytime)

def create_pulses(numpulse, pulsesize):
    ''' creates the noisy pulses '''
    pulses = np.zeros([numpulse, pulsesize])
    amps = np.array([11, 16, 31, 82])
    counter = 0
    u1, u2, u3 = int(numpulse * 0.45), int(numpulse * 0.45), int(numpulse * 0.1)
    risetime = 6
    decaytime = 200
    for i in range(u1):
        times = np.linspace(0, pulsesize-1, pulsesize)
        amp = np.random.uniform(1, 100)
        onset = 250
        pulse = simple_pulse(times, onset, amp, risetime, decaytime)
        pulse_n = np.random.normal(pulse, scale=np.sqrt(amp))
        pulses[i] = pulse_n
        counter += 1
    for i in range(counter, counter+u2):
        times = np.linspace(0, pulsesize-1, pulsesize)
        amp = np.random.choice(amps)
        pulse = simple_pulse(times, 250, np.random.choice(amps), risetime, decaytime)
        pulse_n = np.random.normal(pulse, scale=np.sqrt(amp))
        pulses[i] = pulse_n
        counter += 1
    for i in range(counter, counter+u3):
        times = np.linspace(0, pulsesize-1, pulsesize)
        A = np.random.uniform(1, 20)
        f = 1/80
        tau = 500
        pulse = oscillation(pulsesize, A, f, tau)
        pulse_n = np.random.normal(pulse, scale=np.sqrt(A))
        pulses[i] = pulse_n
        counter += 1
    return pulses

def fit_pulses(pulsearray):
    ''' fits the pulses to the simple_pulse model '''
    fits = []
    errors = []
    for idx in range(len(pulsearray)):
        pulse_i = pulsearray[idx]
        t = np.linspace(0, len(pulse_i)-1, len(pulse_i))
        filtered = np.convolve(pulse_i, np.ones(20)/20, mode='valid')
        ampguess = max(filtered)
        rtguess = 6
        dtguess = 200
        startguess = 250
        initial = [startguess, ampguess, rtguess, dtguess]
        #print(init)
        try:
            fit, error = curve_fit(simple_pulse, t, pulse_i,
                                   p0=initial,
                                   bounds=((50, 1, 1, 150), (450, 200, 50, 250)))
            fits.append(fit)
            errors.append(np.diag(error))
        except(RuntimeError, ValueError):
            continue
    return fits, errors

p = create_pulses(500, 1000)

a, e = fit_pulses(p)
print(len(a))
e1 = np.array(e)
a1 = np.array(a)
oerror = np.sqrt(e1[:, 0])
oparam = a1[:, 0]

ampfits = a1[:, 1]
orelerr = oerror/oparam

plt.plot(orelerr, ampfits, 'x')
plt.title('Figure plotting Fitted Amplitude against Relative Onset Error')
plt.ylabel('Amplitude')
plt.xlabel('Relative Onset Error')
plt.show()

badamps = np.where(ampfits < 5)
selected_amps = np.delete(ampfits, badamps)
amp_hist = plt.hist(selected_amps, bins=100, color="skyblue", lw=0)


def gaussian(data, total, position, width):
    ''' Gaussian function '''
    term = -0.5 * ((data - position)**2 / width**2)
    return total / (np.sqrt(2 * np.pi) * width) * np.exp(term)

def multi_gauss(tvals, t1, p1, w1, t2, p2, w2, t3, p3, w3, t4, p4, w4):
    ''' numpy function to model four gaussian peaks '''
    g1 = gaussian(tvals, t1, p1, w1)
    g2 = gaussian(tvals, t2, p2, w2)
    g3 = gaussian(tvals, t3, p3, w3)
    g4 = gaussian(tvals, t4, p4, w4)
    return g1 + g2 + g3 + g4

values = amp_hist[0]
bins = amp_hist[1]
step = bins[1] - bins[0]
centres_arr = np.array([(bins[0] + step/2.0 + i * step) for i in range(len(bins)-1)])

init = [300, 11, 1, 30, 16, 1, 300, 30, 1, 300, 80, 1]

gaussfit, _ = curve_fit(multi_gauss, centres_arr, values, p0=init)

plt.plot(centres_arr, multi_gauss(centres_arr, *gaussfit), 'k')
plt.title('Figure showing Fitting of Histogram to Gaussian Function')
plt.xlabel('Amplitude')
plt.ylabel('Frequency Density')
plt.legend(['Fitted Gaussian Function'])

plt.show()

p_arr = np.zeros(4)
w_arr = np.zeros(4)

p = 0
for m in range(1, len(gaussfit), 3):
    p_arr[p] = gaussfit[m]
    w_arr[p] = gaussfit[m+1]
    p += 1

plt.plot(p_arr, w_arr, 'x')
plt.title('Fitted Peak Width plotted against Fitted Peak Position')
plt.xlabel('Peak Position')
plt.ylabel('Peak Width')
