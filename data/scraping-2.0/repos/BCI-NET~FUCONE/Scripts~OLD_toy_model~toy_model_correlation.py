import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pyriemann.estimation import Coherences, Covariances


sfreq = 250
duration = 5


def sin_source(amp=1.0, freq=1.0, phase=0.0):
    t = np.arange(0, duration, 1 / sfreq)
    return amp * np.sin(2 * np.pi * freq * t + phase)


A1, A2 = 2, 0.5
f1, f2 = 8.0, 12.0
ph1, ph2 = 0, np.pi / 2
noise = 0.2
n_sources = 2
n_elec = 4


s1 = sin_source(A1, f1, ph1)
s2 = sin_source(A2, f2, ph2)
S = np.stack([s1, s2])
mix = np.random.randn(n_sources, n_elec)

eeg = S.T @ mix + noise * np.random.randn(duration * sfreq, 4)
eeg = eeg.T

plt.plot(eeg.T)
plt.show()

Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

cov = Cov.fit_transform(eeg[np.newaxis, ...])
coh = Coh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)
for i in range(n_elec):
    coh[0, i, i] = 0.0
imcoh = ImCoh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)
for i in range(n_elec):
    imcoh[0, i, i] = 0.0

fig, axes = plt.subplots(1, 3)
axes[0].imshow(
    cov[0], cmap="Purples",
)
axes[1].imshow(coh[0], cmap="Oranges")
axes[2].imshow(imcoh[0], cmap="Greens")
plt.show()

up_idx = np.triu_indices(n_elec, k=1)
fig, axes = plt.subplots(1, 3)
axes[0].imshow(np.atleast_2d(cov[0][up_idx]), cmap="Purples")
axes[1].imshow(np.atleast_2d(coh[0][up_idx]), cmap="Oranges", vmin=0.0)
axes[2].imshow(np.atleast_2d(imcoh[0][up_idx]), cmap="Greens", vmin=0.0)
plt.show()

off_diag_coef = np.zeros((3, n_elec * (n_elec - 1) // 2))
for i, est in enumerate([cov, coh, imcoh]):
    off_diag_coef[i, :] = est[0][up_idx]
R = np.corrcoef(off_diag_coef)
fig, ax = plt.subplots(1, 1)
ax.imshow(R, cmap="YlOrRd")
plt.show()


###############################################################################
# Phase

A = 1
f = 10.0
ph = 0.0
noise = 0.0
n_sources = 2
n_elec = 4

mix = np.random.randn(n_elec, n_elec)
p, l, u = sp.linalg.lu(mix)
mix = (l @ u)[:n_sources, :]
# for i in range(n_sources):
#     mix[i, :] /= mix[i, :].sum()
for i in range(n_elec):
    mix[:, i] /= mix[:, i].sum()
up_idx = np.triu_indices(n_elec, k=1)
Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

corr_cov_coh = []
corr_cov_imcoh = []
corr_coh_imcoh = []
p_space = np.linspace(0, 6 * np.pi, 60)
for dec_phase in p_space:
    s1 = sin_source(A, f, ph)
    s2 = sin_source(A, f, dec_phase)
    S = np.stack([s1, s2])
    eeg = S.T @ mix + noise * np.random.randn(duration * sfreq, 4)
    eeg = eeg.T

    cov = Cov.fit_transform(eeg[np.newaxis, ...])
    coh = Coh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)
    imcoh = ImCoh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)

    off_diag_coef = np.zeros((3, n_elec * (n_elec - 1) // 2))
    for i, est in enumerate([cov, coh, imcoh]):
        off_diag_coef[i, :] = est[0][up_idx]
    R = np.corrcoef(off_diag_coef)

    corr_cov_coh.append(R[0, 1])
    corr_cov_imcoh.append(R[0, 2])
    corr_coh_imcoh.append(R[1, 2])

corr_cov_coh = np.array(corr_cov_coh)
corr_cov_imcoh = np.array(corr_cov_imcoh)
corr_coh_imcoh = np.array(corr_coh_imcoh)

fig, ax = plt.subplots(1, 1)
ax.plot(p_space, corr_cov_coh, label=r"$R(\mathrm{cov}, \mathrm{coh})$")
ax.plot(p_space, corr_cov_imcoh, label=r"$R(\mathrm{cov}, \mathrm{imcoh})$")
ax.plot(p_space, corr_coh_imcoh, label=r"$R(\mathrm{coh}, \mathrm{imcoh})$")
ax.set_xlim(2 * np.pi, 4 * np.pi)
ax.set_xlabel("source phase")
ax.xaxis.set_ticks([2 * np.pi, 3 * np.pi, 4 * np.pi])
ax.xaxis.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])
ax.set_title("Variation of the source phase")
ax.legend(loc="upper right")
plt.show()

###############################################################################
# Amplitude

A = 1
f1, f2 = 10.0, 14.0
ph1, ph2 = 0.0, 0.75 * np.pi
noise = 0.0
n_sources = 2
n_elec = 4

mix = np.random.randn(n_elec, n_elec)
p, l, u = sp.linalg.lu(mix)
mix = (l @ u)[:n_sources, :]
# for i in range(n_sources):
#     mix[i, :] /= mix[i, :].sum()
for i in range(n_elec):
    mix[:, i] /= mix[:, i].sum()
up_idx = np.triu_indices(n_elec, k=1)
Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

corr_cov_coh = []
corr_cov_imcoh = []
corr_coh_imcoh = []
a_space = np.linspace(0.1, 4.0, 40)
for amp in a_space:
    s1 = sin_source(A, f1, ph1)
    s2 = sin_source(amp, f2, ph2)
    S = np.stack([s1, s2])
    eeg = S.T @ mix + noise * np.random.randn(duration * sfreq, 4)
    eeg = eeg.T

    cov = Cov.fit_transform(eeg[np.newaxis, ...])
    coh = Coh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)
    imcoh = ImCoh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)

    off_diag_coef = np.zeros((3, n_elec * (n_elec - 1) // 2))
    for i, est in enumerate([cov, coh, imcoh]):
        off_diag_coef[i, :] = est[0][up_idx]
    R = np.corrcoef(off_diag_coef)

    corr_cov_coh.append(R[0, 1])
    corr_cov_imcoh.append(R[0, 2])
    corr_coh_imcoh.append(R[1, 2])

corr_cov_coh = np.array(corr_cov_coh)
corr_cov_imcoh = np.array(corr_cov_imcoh)
corr_coh_imcoh = np.array(corr_coh_imcoh)

fig, ax = plt.subplots(1, 1)
ax.plot(a_space, corr_cov_coh, label=r"$R(\mathrm{cov}, \mathrm{coh})$")
ax.plot(a_space, corr_cov_imcoh, label=r"$R(\mathrm{cov}, \mathrm{imcoh})$")
ax.plot(a_space, corr_coh_imcoh, label=r"$R(\mathrm{coh}, \mathrm{imcoh})$")
ax.set_xlabel("amplitude")
ax.set_title("Variation of the amplitude")
ax.legend(loc="upper right")
plt.show()


###############################################################################
# freq

A = 1
f = 10.0
ph = 0.0
noise = 0.0
n_sources = 2
n_elec = 4

mix = np.random.randn(n_elec, n_elec)
p, l, u = sp.linalg.lu(mix)
mix = (l @ u)[:n_sources, :]
# for i in range(n_sources):
#     mix[i, :] /= mix[i, :].sum()
for i in range(n_elec):
    mix[:, i] /= mix[:, i].sum()
up_idx = np.triu_indices(n_elec, k=1)
Cov = Covariances()
Coh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="ordinary")
ImCoh = Coherences(fmin=1.0, fmax=40.0, fs=sfreq, coh="imaginary")

corr_cov_coh = []
corr_cov_imcoh = []
corr_coh_imcoh = []
f_space = np.linspace(4.0, 50.0, 47)
for freq in f_space:
    s1 = sin_source(A, f, ph)
    s2 = sin_source(A, freq, ph)
    S = np.stack([s1, s2])
    eeg = S.T @ mix + noise * np.random.randn(duration * sfreq, 4)
    eeg = eeg.T

    cov = Cov.fit_transform(eeg[np.newaxis, ...])
    coh = Coh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)
    imcoh = ImCoh.fit_transform(eeg[np.newaxis, ...]).mean(axis=-1)

    off_diag_coef = np.zeros((3, n_elec * (n_elec - 1) // 2))
    for i, est in enumerate([cov, coh, imcoh]):
        off_diag_coef[i, :] = est[0][up_idx]
    R = np.corrcoef(off_diag_coef)

    corr_cov_coh.append(R[0, 1])
    corr_cov_imcoh.append(R[0, 2])
    corr_coh_imcoh.append(R[1, 2])

corr_cov_coh = np.array(corr_cov_coh)
corr_cov_imcoh = np.array(corr_cov_imcoh)
corr_coh_imcoh = np.array(corr_coh_imcoh)

fig, ax = plt.subplots(1, 1)
ax.plot(f_space, corr_cov_coh, label=r"$R(\mathrm{cov}, \mathrm{coh})$")
ax.plot(f_space, corr_cov_imcoh, label=r"$R(\mathrm{cov}, \mathrm{imcoh})$")
ax.plot(f_space, corr_coh_imcoh, label=r"$R(\mathrm{coh}, \mathrm{imcoh})$")
ax.vlines(10.0, ymin=-1.1, ymax=1.0, linestyles="dashed", color="k")
# ax.set_ylim(-1.08, 0.9)
ax.xaxis.set_ticks([10, 20, 30, 40, 50])
ax.xaxis.set_ticklabels([r"$f$", r"$2f$", r"$3f$", r"$4f$", r"$5f$"])
ax.set_xlabel("frequency")
ax.set_title("Variation of the frequency")
ax.legend(loc="center right")
plt.show()
