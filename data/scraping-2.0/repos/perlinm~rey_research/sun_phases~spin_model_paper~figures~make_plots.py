#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import proj3d

preamble = r"""
\usepackage{braket}
"""

fontsize = 10
params = { "font.family" : "serif",
           "font.serif" : "Computer Modern",
           "font.size" : fontsize,
           "axes.titlesize" : fontsize,
           "axes.labelsize" : fontsize,
           "text.usetex" : True,
           "text.latex.preamble" : preamble }
plt.rcParams.update(params)

### arrow styles and line colors

arrow_style = "head_width=1.5,head_length=4"
single_kw = dict(arrowstyle = "-|>,"+arrow_style, color="k")
double_kw = dict(arrowstyle = "<|-|>,"+arrow_style, color="k")

def marker_kw(color = None):
    return { "marker" : "o",
             "color" : color,
             "markeredgecolor" : "k",
             "markeredgewidth" : 1 }

color_cycle = [ _["color"] for _ in plt.rcParams["axes.prop_cycle"] ]

dn_color = color_cycle[0]
up_color = color_cycle[1]

def add_arrow(start, end, axis = None, *args, **kw_args):
    if axis is None: axis = plt.gca()
    arrow = patches.FancyArrowPatch(start, end, *args, **kw_args)
    axis.add_patch(arrow)

# ##########################################################################################
# # lattice diagram
# ##########################################################################################

# figsize = (2,0.75)
# plt.figure(figsize = figsize)
# x_lim, y_lim = 1.6, 0.5
# x_vals = np.linspace(-x_lim, x_lim, 300)
# plt.plot(x_vals, -y_lim*np.cos(2*np.pi*x_vals), "k")

# btm_y, top_y = -y_lim*0.4, y_lim*0.7
# plt.plot([-1], [btm_y], **marker_kw(dn_color))
# plt.plot([+0], [btm_y], **marker_kw(dn_color))
# add_arrow((-1,btm_y+0.08), (+0,btm_y+0.08),
#       connectionstyle="arc3,rad=-0.4", **single_kw)
# plt.annotate(r"$J$", (-0.5, btm_y), ha = "center", va = "center")

# y_vals = np.linspace(top_y, btm_y, 30)
# x_vals = 1+0.06*np.sin(5*np.pi*(y_vals-top_y)/(top_y-btm_y))
# plt.plot(x_vals, y_vals, "r", linewidth = 1)
# plt.plot([1], [btm_y], **marker_kw(dn_color))
# plt.plot([1], [top_y], **marker_kw(up_color))
# plt.annotate(r"$U$", (0.8, y_lim*0.9), ha = "center", va = "center")

# plt.axis("off")
# plt.subplots_adjust(0,0.05,1,1.2,0,0)
# plt.savefig("lattice.pdf")
# plt.close("all")

# ##########################################################################################
# # frozen-mode approximation
# ##########################################################################################

# figsize = (1.75,1.12)
# momenta = np.linspace(-np.pi,np.pi,100)

# figure = plt.figure(figsize = figsize)
# axis = plt.gca()
# plt.plot(momenta, -np.cos(momenta), "k")
# plt.xlabel("momentum")
# plt.ylabel("kinetic energy")
# plt.xlim(-np.pi,np.pi)
# plt.xticks([])
# plt.yticks([])

# ylim = axis.get_ylim()
# yrange = ylim[1] - ylim[0]
# btm = (-1-ylim[0]) / yrange
# top = (+1-ylim[0]) / yrange

# axis.annotate("", xytext = (1.1,btm), xy = (1.1,top),
#               xycoords = "axes fraction",
#               arrowprops = dict( arrowstyle = "<->" ))
# for hh in [ btm, top ]:
#     axis.annotate("", xytext = (1.05,hh), xy = (1.15,hh),
#                   xycoords = "axes fraction",
#                   arrowprops = dict( arrowstyle = "-" ))
# axis.annotate(r"$4J$", xy = (1.15,.5),
#               xycoords = "axes fraction",
#               va = "center", ha = "left")

# h1, h2 = 0.56, 0.44
# axis.annotate("", xytext = (.5,h1+.25), xy = (.5,h1),
#               xycoords = "axes fraction",
#               arrowprops = dict( arrowstyle = "->" ))
# axis.annotate("", xytext = (.5,h2-.25), xy = (.5,h2),
#               xycoords = "axes fraction",
#               arrowprops = dict( arrowstyle = "->" ))
# for hh in [ h1, h2 ]:
#     axis.annotate("", xytext = (.45,hh), xy = (.55,hh),
#                   xycoords = "axes fraction",
#                   arrowprops = dict( arrowstyle = "-" ))
# axis.annotate(r"$U$", xy = (.62,(h1+h2)/2),
#               xycoords = "axes fraction",
#               va = "center", ha = "center")

# plt.tight_layout(pad = 0.2)
# plt.savefig("frozen_modes.pdf")
# plt.close("all")

# ##########################################################################################
# # interacting spins
# ##########################################################################################

# radius = 0.4
# height = 0.6

# def place_spins(centers, angles, colors, labels = None, height = height):
#     assert(len(centers) == len(angles) == len(colors))

#     x_vals, y_vals = list(zip(*centers))
#     axis_pad = 2.6 * radius
#     def axis_width(vals):
#         return max(vals) - min(vals) + 2*axis_pad

#     width = axis_width(x_vals) / axis_width(y_vals) * height
#     figsize = (width,height)
#     figure, axis = plt.subplots(figsize = figsize)

#     if labels is None: labels = [ " " ] * len(centers)
#     for center, angle, color, label in zip(centers, angles, colors, labels):
#         center = np.array(center)

#         circle = patches.Circle(center, radius = radius, facecolor = color)
#         axis.add_patch(circle)

#         arrow_tip = 4 * radius * np.array([ np.sin(angle), np.cos(angle) ])
#         arrow_base = np.array(center) - arrow_tip*43/100
#         arrow = patches.Arrow(*arrow_base, *arrow_tip, width = 0.3, color = color)
#         axis.add_patch(arrow)

#         axis.text(*center, f"${label}$", ha = "center", va = "center")

#         if len(colors) == 2:
#             state_text = r"\psi" if color == color_cycle[0] else r"\phi"
#             state_text = r"\large$\ket{" + state_text + r"}_{" + label + "}$"
#             axis.text(*center+np.array([0.2,-1.2]), state_text, ha = "center", va = "center")

#     axis.set_xlim(min(x_vals) - axis_pad, max(x_vals) + axis_pad)
#     axis.set_ylim(min(y_vals) - axis_pad, max(y_vals) + axis_pad)
#     axis.set_aspect("equal")
#     axis.set_axis_off()

#     plt.tight_layout(pad = 0)
#     return figure, axis

# def make_wiggles(centers, points = 100, waves = 4, axis = None):
#     if axis is None: axis = plt.gca()
#     pairs = [ ( centers[jj], centers[kk] )
#               for jj in range(len(centers))
#               for kk in range(jj) ]
#     for center_lft, center_rht in pairs:
#         x_lft, x_rht = center_lft[0], center_rht[0]
#         y_lft, y_rht = center_lft[1], center_rht[1]
#         angle = np.arctan2(y_rht - y_lft, x_rht - x_lft)

#         t_vals = np.linspace(0, 1, points)
#         h_vals = radius/5 * np.sin(t_vals * 2*np.pi * waves)

#         x_vals = np.linspace(x_lft, x_rht, points) - np.sin(angle) * h_vals
#         y_vals = np.linspace(y_lft, y_rht, points) + np.cos(angle) * h_vals
#         axis.plot(x_vals, y_vals, "r", zorder = -1)

# ### spin pair

# centers = [ (-1,0), (+1,0) ]
# angles = [ -np.pi/3, np.pi/6 ]
# colors = color_cycle[:2]
# labels = [ "p", "q" ]

# figure, axis = place_spins(centers, angles, colors, labels)
# make_wiggles(centers)
# bbox = figure.bbox_inches.from_bounds(0.15, -0.01, 0.9, 0.61)
# plt.savefig("spins_int.pdf", bbox_inches = bbox)

# figure, axis = place_spins(centers, angles[::-1], colors[::-1], labels)
# make_wiggles(centers)
# bbox = figure.bbox_inches.from_bounds(0.2, -0.01, 0.85, 0.61)
# plt.savefig("spins_int_swap.pdf", bbox_inches = bbox)

# ### spin triplet

# centers = [ (0,0), (+1,0), (+2,0) ]
# angles = [ np.pi/6 ] * 3
# colors = [ color_cycle[0] ] * 3
# figure, axis = place_spins(centers, angles, colors = colors)
# bbox = figure.bbox_inches.from_bounds(0.15, 0.05, 0.9, 0.53)
# plt.savefig("many_same.pdf", bbox_inches = bbox)

# centers = [ (0,0), (+1,0), (+2,0) ]
# angles = [ np.pi/6, 0, np.pi/3 ]
# colors = color_cycle[:3]
# figure, axis = place_spins(centers, angles, colors = colors)
# bbox = figure.bbox_inches.from_bounds(0.15, 0.05, 1, 0.55)
# plt.savefig("many_diff.pdf", bbox_inches = bbox)

# plt.close("all")

# ##########################################################################################
# # spin representations
# ##########################################################################################

# import qutip as qt

# def plot_sphere(vectors, points = None, color = "#d62728", width = 2):
#     sphere = qt.Bloch()
#     sphere.add_vectors(vectors)
#     if points is not None:
#         zipped_points = [ np.array(pts) for pts in zip(*points) ]
#         sphere.add_points(zipped_points)

#     sphere.frame_alpha = 0
#     sphere.xlabel, sphere.ylabel, sphere.zlabel = [["",""]]*3
#     sphere.figsize = (width,width)

#     sphere.vector_color = [ color ]
#     sphere.point_color = [ color ]

#     sphere.render()
#     plt.gca().set_zlim(-.55,.55)
#     return sphere

# sphere = plot_sphere([0,1,0])
# plt.savefig("bloch_x.pdf")

# def vec_xy(angle = 0):
#     xhat = np.array([1,0,0])
#     yhat = np.array([0,1,0])
#     return np.cos(angle) * yhat - np.sin(angle) * xhat

# angle = 2*np.pi * 2/3
# points = 10
# points = [ vec_xy(part) for part in np.linspace(0,angle,points) ]
# sphere = plot_sphere(vec_xy(angle), points)
# plt.savefig("bloch_xy.pdf")

# import scipy.stats
# from dicke_methods import coherent_spin_state, spin_op_z_dicke, plot_dicke_state

# kwargs = dict( single_sphere = True, shade = False, grid_size = 501 )

# qubit_state = coherent_spin_state("+X", 1)
# plot_dicke_state(qubit_state, view_angles = (30,-50), **kwargs)
# plt.savefig("qubit_dist.pdf")

# dim = 10
# sz = spin_op_z_dicke(dim-1).todense()
# rot = scipy.linalg.expm(-1j*sz * np.pi/2)

# np.random.seed(0)
# random_U = scipy.stats.unitary_group.rvs(dim)
# qudit_state = rot @ random_U @ coherent_spin_state("+X", dim-1)
# figure, axes = plot_dicke_state(qudit_state, view_angles = (0,0), **kwargs)
# plt.savefig("qudit_dist.pdf")

# plt.close("all")

##########################################################################################
# spin-orbit coupling dispersion relations
##########################################################################################

figsize = (3.25,2.5)
figure, axes = plt.subplots(2, 2, figsize = figsize, sharex=True, sharey=True)
momenta = np.linspace(-np.pi,np.pi,200)

##################################################
# SU(2), lab frame

phi = np.pi/2
q_i = np.pi/6

axis = axes[0,0]
axis.plot(momenta, -np.cos(momenta), "k-")

# make arrow for coupling
q_f = q_i + phi
E_i, E_f = -np.cos(q_i), -np.cos(q_f)
add_arrow((q_i+0.1,E_i), (q_f,E_f-0.08), axis = axis,
          connectionstyle="arc3,rad=0.4", **single_kw)
axis.annotate(r"$\Omega$", (q_i + (q_f-q_i), E_i + (E_f-E_i)*1/5))
axis.plot([q_i], [E_i], **marker_kw(dn_color))
axis.plot([q_f], [E_f], **marker_kw(up_color))

# make arrow showing SOC angle
start, end = -1.05, -1.25
mid = ( start + end ) / 2
add_arrow((q_i,mid), (q_f,mid), axis = axis, zorder = 10, **double_kw)
axis.annotate(r"$\phi$", ((q_i+q_f)/2-0.2, start))
axis.plot([q_i,q_i], [start,end], "k", lw = 1)
axis.plot([q_f,q_f], [start,end], "k", lw = 1)

# make arrows annotating "atoms"
axis.annotate(r"$\ket\downarrow$", (q_i-0.1,E_i+0.1), va = "bottom", ha = "right")
axis.annotate(r"$\ket\uparrow$", (q_f-0.1,E_f+0.1), va = "bottom", ha = "right")

##################################################
# SU(2), gauge frame

axis = axes[1,0]
axis.plot(momenta, -np.cos(momenta-phi/2), color = dn_color)
axis.plot(momenta, -np.cos(momenta+phi/2), color = up_color)

# make arrow showing SOC angle
start, end = -1.15, -1.35
mid = ( start + end ) / 2
add_arrow((-phi/2,mid), (+phi/2,mid), axis = axis, zorder = 10, **double_kw)
axis.annotate(r"$\phi$", (-0.2, start))
axis.plot([-phi/2,-phi/2], [start,end], "k", linewidth = 1)
axis.plot([+phi/2,+phi/2], [start,end], "k", linewidth = 1)

# make arrow for coupling
q_i = q_i + phi/2
q_f = q_i
E_i, E_f = -np.cos(q_i-phi/2), -np.cos(q_f+phi/2)
add_arrow((q_i,E_i+0.03), (q_f,E_f-0.08), axis = axis, **single_kw)
axis.annotate(r"$\Omega$", (q_i+np.pi/15, -0.1))
axis.plot([q_i], [E_i], **marker_kw(dn_color))
axis.plot([q_f], [E_f], **marker_kw(up_color))

# make arrows annotating "atoms"
axis.annotate(r"$\ket\downarrow$", (q_i+0.3,E_i-0.15))
axis.annotate(r"$\ket\uparrow$", (q_f-0.9,E_f+0.15))

##################################################
# SU(d), lab frame

axis = axes[0,1]
axis.plot(momenta, -np.cos(momenta), "k")

# make arrow for coupling
q_1 = -np.pi/3
q_2 = q_1 + phi
q_3 = q_2 + phi
E_1, E_2, E_3 = -np.cos(q_1), -np.cos(q_2), -np.cos(q_3)
add_arrow((q_1+0.1,E_1), (q_2,E_2+0.08), axis = axis, connectionstyle="arc3,rad=-0.4", **single_kw)
add_arrow((q_2+0.1,E_2), (q_3,E_3-0.08), axis = axis, connectionstyle="arc3,rad=+0.4", **single_kw)
axis.annotate(r"$\Omega$", [ (q_1+q_2)/2, E_1+0.1 ])
axis.plot([q_1], [E_1], **marker_kw("tab:blue"))
axis.plot([q_2], [E_2], **marker_kw("tab:orange"))
axis.plot([q_3], [E_3], **marker_kw("tab:green"))

# make arrow showing SOC angle
start, end = 0, 0.2
mid = ( start + end ) / 2
add_arrow((q_1,mid), (q_2,mid), axis = axis, zorder = 10, **double_kw)
axis.annotate(r"$\phi$", ((q_1+q_2)/2-0.2, end))
axis.plot([q_1,q_1], [start,end], "k", lw = 1)
axis.plot([q_2,q_2], [start,end], "k", lw = 1)

# make arrows annotating "atoms"
axis.annotate(r"$\ket\mu$", (q_1-0.3,E_1), ha = "right", va = "center")
axis.annotate(r"$\ket{\mu+1}$", (q_2+0.6,E_2), ha = "left", va = "center")
axis.annotate(r"$\ket{\mu+2}$", (q_3-0.1,E_3), ha = "right", va = "bottom")

##################################################
# SU(d), gauge frame

phi = np.pi/2 * 0.8
qq = np.pi*5/9
levels = 4

axis = axes[1,1]

# draw energy levels
for level in range(levels):
    mu = level - (levels-1)/2
    line = axis.plot(momenta, -np.cos(momenta+mu*phi))
axis.set_prop_cycle(None) # reset color cycle

# add three "atoms"
heights = {}
for level in range(3):
    mu = level - (levels-1)/2
    heights[level] = -np.cos(qq+mu*phi)
    axis.plot([qq], [heights[level]], **marker_kw())

# make arrows for coupling
add_arrow((qq,heights[0]+0.02), (qq,heights[1]-0.05), axis = axis, **single_kw, zorder = 5)
add_arrow((qq,heights[1]+0.02), (qq,heights[2]-0.05), axis = axis, **single_kw, zorder = 5)
axis.annotate(r"$\Omega$", (qq-np.pi/10, -0.1), ha = "center")

# make arrow showing SOC angle
start, end = -1.15, -1.35
mid = ( start + end ) / 2
add_arrow((-phi/2-0.1,mid), (+phi/2+0.1,mid), axis = axis, zorder = 10, **double_kw)
axis.annotate(r"$\phi$", (-0.2, start))
axis.plot([-phi/2,-phi/2], [start,end], "k", linewidth = 1)
axis.plot([+phi/2,+phi/2], [start,end], "k", linewidth = 1)

##################################################
# cleanup, axis labels, etc.

for axis in axes.ravel():
    axis.set_xlim(momenta[0], momenta[-1])
    axis.set_xticks([])
    axis.set_yticks([])

axes[0,0].set_ylabel("kinetic energy")
axes[1,0].set_ylabel("kinetic energy")
axes[1,0].set_xlabel("momentum")
axes[1,1].set_xlabel("momentum")

axes[0,0].set_title("$n=2$")
axes[0,1].set_title(f"$n={levels}$")

for loc, tag in [ [ (0,1), "lab" ],
                  [ (1,1), "gauge" ] ]:
    axis = axes[loc].twinx()
    axis.yaxis.tick_right()
    axis.set_yticks([])
    axis.set_ylabel(f"{tag} frame")

bbox = dict( boxstyle = "round", facecolor = "lightgray", alpha = 1)
kwargs = dict( bbox = bbox, fontweight = "bold" )
axes[0,0].text(0.1, 0.8, r"{\bf (a)}", transform = axes[0,0].transAxes, **kwargs, va = "center")
axes[0,1].text(0.1, 0.8, r"{\bf (b)}", transform = axes[0,1].transAxes, **kwargs, va = "center")
axes[1,0].text(0.1, 0.8, r"{\bf (c)}", transform = axes[1,0].transAxes, **kwargs, va = "center")
axes[1,1].text(0.1, 0.8, r"{\bf (d)}", transform = axes[1,1].transAxes, **kwargs, va = "center")

plt.tight_layout(pad = 0.2)
plt.savefig("soc_panels.pdf")

##########################################################################################
# three-laser drive
##########################################################################################

# class Arrow3D(patches.FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         arrow_kwargs = dict( lw = 1, mutation_scale = 15,
#                              color = "k", arrowstyle = "wedge" )
#         arrow_kwargs.update(kwargs)
#         patches.FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **arrow_kwargs)
#         self._verts3d = xs, ys, zs

#     def draw(self, renderer):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
#         patches.FancyArrowPatch.draw(self, renderer)

# def wave(start, end, ampl = 0.05, num = 20, points = 1000, spiral = False):
#     line = np.linspace(start, end, points)
#     xs, ys = line, -ampl*np.cos(line*np.pi*num)
#     if not spiral: return xs, ys
#     else:
#         zs = ampl*np.sin(line*np.pi*num)
#         return xs, ys, zs

# def rot(xs, ys, angle):
#     xs, ys = np.array(xs), np.array(ys)
#     return [ np.cos(angle) * xs + np.sin(angle) * ys,
#             -np.sin(angle) * xs + np.cos(angle) * ys ]

# def draw_lasers(colors, name):
#     figure = plt.figure()
#     axis = figure.add_subplot(111, projection = "3d")
#     axis.set_axis_off()

#     # grey reference lines
#     axis.plot([-1,+1],[0,0],[0,0], color = "gray")
#     axis.plot([0,0],[-1,+1],[0,0], color = "gray")
#     axis.plot([0,0],[0,0],[-1,+1], color = "gray")
#     axis.plot([-1,+1,+1,-1,-1], [0]*5, [-1,-1,+1,+1,-1], color = "gray")

#     # lattice
#     height = 0.1
#     peaks = 10
#     angle = np.pi/5
#     xlim  = 1 / max( np.cos(angle), np.sin(angle) )
#     xs, ys = rot(*wave(-2, 2, height, peaks), angle)
#     in_bounds = (abs(xs) < 1) & (abs(ys) < 1)
#     xs, ys = xs[in_bounds], ys[in_bounds]
#     axis.plot(xs, 0*xs, ys, color = "k")
#     xxs = 2/peaks * (np.arange(2*peaks)-peaks) - 0.005 # fudge factor
#     for xx in xxs:
#         xx, yy = rot(xx, 0, angle)
#         if abs(xx) > 1 or abs(yy) > 1: continue
#         axis.plot(xx, 0, yy, "o", color = color_cycle[0], markeredgecolor = "k")

#     # x-laser
#     xs, ys = wave(-1,-0.32)
#     axis.plot(ys, xs, color = colors[0])
#     axis.add_artist(Arrow3D([0,0], [-0.34,-0.19], [0,0], color = colors[0], zorder = 2))

#     # up/dn-lasers
#     xs, ys, zs = wave(+1, +0.35, 0.04, spiral = True)
#     arrow_xs = 0.38 + np.array([0,-0.14])
#     arrow_ys = np.array([0,0])

#     axis.plot(+xs, +ys, +zs, color = colors[1])
#     axis.plot(-xs, -ys, -zs, color = colors[2])
#     axis.add_artist(Arrow3D(+arrow_xs, 0*arrow_xs, +arrow_ys, color = colors[1], zorder = 2))
#     axis.add_artist(Arrow3D(-arrow_xs, 0*arrow_xs, -arrow_ys, color = colors[2], zorder = 2))

#     # arrows for angles between up/dn-lasers and lattice
#     anchor, offset = ( 0.84, 0 ), angle/15
#     start = rot(*anchor, offset)
#     end = rot(*anchor, angle-offset)
#     xs, zs = map(np.array,zip(start, end))
#     kwargs = dict( arrowstyle = "<|-|>,head_width=0.1,head_length=.2",
#                    connectionstyle = "arc3,rad=-.4" )
#     axis.add_artist(Arrow3D(+xs, (0,0), +zs, zorder = 2, **kwargs))
#     axis.add_artist(Arrow3D(-xs, (0,0), -zs, zorder = 2, **kwargs))

#     # text for angles between up/dn-lasers and lattice
#     base_loc = ( 0.93, 0 )
#     xx, zz = rot(*base_loc, angle/2)
#     axis.text(+xx, 0, +zz, r"\Large$\theta$", va = "center", ha = "center")
#     axis.text(-xx, 0, -zz, r"\Large$\theta$", va = "center", ha = "center")

#     # text for laser drive amplitudes
#     xx, zz = 0.8, 0.2
#     axis.text(+xx, 0, +zz, r"\Large$\Omega_-$", va = "center", ha = "center")
#     axis.text(-xx, 0, -zz, r"\Large$\Omega_+$", va = "center", ha = "center")
#     axis.text(-0.2, -0.8, 0, r"\Large$\Omega_0$", va = "center", ha = "center")

#     # arrows for axes
#     base = np.array([0.6,0,1.2])
#     kwargs = dict( arrowstyle = "-|>,head_width=0.1,head_length=.2" )
#     for dd, ll in [ [ (0.75,0,0), "z" ], [ (0,1,0), "x" ], [ (0,0,0.9), "y" ] ]:
#         start = base - 0.03*np.array(dd)
#         end = base + 0.3*np.array(dd)
#         xs, ys, zs = map(np.array, zip(start,end))
#         axis.add_artist(Arrow3D(xs, ys, zs, zorder = 2, **kwargs))
#         xx, yy, zz = end + 0.05*np.array(dd)
#         axis.text(xx, yy, zz, ll, va = "center", ha = "center")

#     # save figure
#     plt.tight_layout(pad = 0)
#     bbox = figure.bbox_inches.from_bounds(2, 1.1, 2.5, 2.8)
#     plt.savefig(name, bbox_inches = bbox)

# draw_lasers("rrr", "3LD_geometry.pdf")
# plt.close("all")
