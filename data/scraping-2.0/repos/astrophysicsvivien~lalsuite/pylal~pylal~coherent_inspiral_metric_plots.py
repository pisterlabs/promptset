# Copyright (C) 2012  Drew Keppel
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# ============================================================================
#
#                                   Preamble
#
# ============================================================================
#


"""
This module provides tools for plotting different results associated with the
coheerent inspiral metric module.
"""

import matplotlib
from pylal import coherent_inspiral_metric as metric
import numpy
import scipy
from scipy import pi,log,exp,sin,cos,log10
import pylab
import math
import sys
from mpl_toolkits.basemap import Basemap
#from pylal import inject
#from pylal.xlal.tools import cached_detector
pylab.rc('text', usetex=True)

__author__ = "Drew Keppel <drew.keppel@ligo.org>"

def plotErrorEllipse(xy,g,d,ax,edge='b',face='none', style='solid', **kwargs):
	"""
	Computes error ellipses for points in a two dimensional space given a
	position and the 2D metric.
	"""
	a,b,b,c = g.flatten()
	semimajor = (2./((a+c)-((a-c)**2+4*b**2)**.5))**.5
	semiminor = (2./((a+c)+((a-c)**2+4*b**2)**.5))**.5

	U, s, Vh = scipy.linalg.svd(g)
	orient = math.atan2(U[0,0],U[1,0])*180/pi
	ellipsePlot = matplotlib.patches.Ellipse(xy=xy, width=2*d**.5*semimajor,
		height=2*d**.5*semiminor, angle=-orient, facecolor=face, edgecolor=edge, linestyle=style, **kwargs)
	ax.add_patch(ellipsePlot)

def plot_skymetric_from_file(filename,maxmismatch=0.1):
	"""
	Plot the position, error ellipses, and square root of the determinant
	of the metric for the template locations on the sky. Takes in the name
	of the ascii file containing the sky locations and metric as well as
	the size of the error ellipses (maxmismatch).
	"""
	RAs,decs,gRRs,gRds,gdds = numpy.loadtxt(filename, unpack=True, usecols=(0,1,2,3,4))
	gs = []
	rootdetgs = []
	for gRR,gRd,gdd in zip(gRRs,gRds,gdds):
		g = numpy.array([[gRR,gRd],[gRd,gdd]])
		rootdetgs.append((scipy.linalg.det(g))**.5)
		gs.append(g)

	fig = pylab.figure()
	ax = fig.add_axes((.1,.1,.8,.8))

	dRA = 2*pi/200
	ddec = pi/200
	RA,dec = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2+ddec:pi/2:ddec]
	z = pylab.griddata(RAs,decs,rootdetgs,RA,dec)
	levels = scipy.linspace(-1.,2.,101)
	CF = ax.contourf(RA, dec, log10(z), levels=levels)
	matplotlib.pyplot.colorbar(CF)

	for RA,dec,g in zip(RAs,decs,gs):
		plotErrorEllipse((RA,dec),g,maxmismatch,ax,edge='k')

	ax.set_yticks([-pi/2,-pi/4,0,pi/4,pi/2])
	ax.set_yticklabels(['$-\pi/2$','$-\pi/4$','$0$','$\pi/4$','$\pi/2$'])
	ax.set_xticks([-pi,-pi/2,0,pi/2,pi])
	ax.set_xticklabels(['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])
	ax.set_xlim(-pi,pi)
	ax.set_ylim(-pi/2,pi/2)
	ax.set_xlabel(r'$\alpha$')
	ax.set_ylabel(r'$\delta$')
	ax.invert_xaxis()
	ax.set_title('Sky Metric')

def plot_skymetric_averagesnr_from_file(filename, maxmismatch=0.1, detectors=None):
	"""
	Plot the position, error ellipses, and average SNR^2 for the template
	locations on the sky. Takes in the name of the ascii file containing
	the sky locations and metric as well as the size of the error ellipses
	(maxmismatch).
	"""
	RAs,decs,gRRs,gRds,gdds = numpy.loadtxt(filename, unpack=True, usecols=(0,1,2,3,4))

	avesnr = metric.average_snr(RAs, decs, detectors)

	gs = []
	for gRR,gRd,gdd in zip(gRRs,gRds,gdds):
		g = numpy.array([[gRR,gRd],[gRd,gdd]])
		gs.append(g)

	fig = pylab.figure()
	ax = fig.add_axes((.1,.1,.8,.8))

	dRA = 2*pi/200
	ddec = pi/200
	RA,dec = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2+ddec:pi/2:ddec]
	z = pylab.griddata(RAs,decs,avesnr,RA,dec)
	CF = ax.contourf(RA, dec, log(z)/log(10.), 100)
	matplotlib.pyplot.colorbar(CF)

	for RA,dec,g in zip(RAs,decs,gs):
		plotErrorEllipse((RA,dec),g,maxmismatch,ax,edge='k')

	ax.set_yticks([-pi/2,-pi/4,0,pi/4,pi/2])
	ax.set_yticklabels(['$-\pi/2$','$-\pi/4$','$0$','$\pi/4$','$\pi/2$'])
	ax.set_xticks([-pi,-pi/2,0,pi/2,pi])
	ax.set_xticklabels(['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])
	ax.set_xlim(-pi,pi)
	ax.set_ylim(-pi/2,pi/2)
	ax.set_xlabel(r'$\alpha$')
	ax.set_ylabel(r'$\delta$')
	ax.invert_xaxis()
	ax.set_title('Sky Metric and Average SNR')

def plot_massmetric_on_sky_from_file(filename):
	"""
	Plot the position, and square root of the determinant of the mass
	metric for the template locations on the sky. Takes in the name of the
	ascii file containing the locations and metric as well as the size of
	the error ellipses (maxmismatch).
	"""
	RAs,decs,gmms,gmes,gees = numpy.loadtxt(filename, unpack=True, usecols=(0,1,5,6,7))
	gs = []
	rootdetgs = []
	for gmm,gme,gee in zip(gmms,gmes,gees):
		g = numpy.array([[gmm,gme],[gme,gee]])
		rootdetgs.append((scipy.linalg.det(g))**.5)
		gs.append(g)

	fig = pylab.figure()
	ax = fig.add_axes((.1,.1,.8,.8))

	dRA = 2*pi/200
	ddec = pi/200
	RA,dec = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2+ddec:pi/2:ddec]
	z = pylab.griddata(RAs,decs,rootdetgs,RA,dec)
	CF = ax.contourf(RA, dec, log(z), 100)
	ax.plot(RAs, decs, 'kx')
	matplotlib.pyplot.colorbar(CF)

	ax.set_yticks([-pi/2,-pi/4,0,pi/4,pi/2])
	ax.set_yticklabels(['$-\pi/2$','$-\pi/4$','$0$','$\pi/4$','$\pi/2$'])
	ax.set_xticks([-pi,-pi/2,0,pi/2,pi])
	ax.set_xticklabels(['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])
	ax.set_xlim(-pi,pi)
	ax.set_ylim(-pi/2,pi/2)
	ax.set_xlabel(r'$\alpha$')
	ax.set_ylabel(r'$\delta$')
	ax.invert_xaxis()
	ax.set_title('Mass Metric')

def plot_massmetric_variations_from_file(filename, maxmismatch=0.1):
	"""
	Plot the variations of the mass metric as a function of sky position
	for a single point in the mass space. Read mass metric from ascii
	file and the size of the error ellipses are determined by the
	parameter maxmismatch.
	"""
	gmms,gmes,gees = numpy.loadtxt(filename, unpack=True, usecols=(5,6,7))
	gs = []
	rootdetgs = []
	for gmm,gme,gee in zip(gmms,gmes,gees):
		g = numpy.array([[gmm,gme],[gme,gee]])
		rootdetgs.append((scipy.linalg.det(g))**.5)
		gs.append(g)

	fig = pylab.figure()
	ax = fig.add_axes((.1,.1,.8,.8))

	eta = 0.25
	mchirp = 2. * metric.M_sun * eta**(3./5.)

	for g in gs:
		plotErrorEllipse((mchirp,eta),g,maxmismatch,ax,edge='r')

	g = numpy.array([[scipy.mean(gmms),scipy.mean(gmes)],[scipy.mean(gmes),scipy.mean(gees)]])
	plotErrorEllipse((mchirp,eta),g,maxmismatch,ax,edge='k')

	g = scipy.array([[4.22613677e+18, -2.17901658e+11],
		[ -2.17901658e+11,1.47648365e+04]])
	plotErrorEllipse((mchirp,eta),g,maxmismatch,ax,edge='b')

	ax.set_xlim(mchirp/1.0005,mchirp*1.0005)
	ax.set_ylim(0.24,0.26)

	ax.set_xlabel(r'$M_c$')
	ax.set_ylabel(r'$\eta$')
	ax.set_title('Mass Metric')

def plot_massmetric_from_file(filename,maxmismatch=0.06):
	"""
	Plot the position, error ellipses, and average SNR^2 for the template
	locations in the mass space. Takes in the name of the ascii file
	containing the locations and metric as well as the size of the error
	ellipses (maxmismatch).
	"""
	mchirps,etas,gmms,gmes,gees = numpy.loadtxt(filename, unpack=True, usecols=(0,1,2,3,4))
	gs = []
	rootdetgs = []
	for gmm,gme,gee in zip(gmms,gmes,gees):
		g = numpy.array([[gmm,gme],[gme,gee]])
		rootdetgs.append((scipy.linalg.det(g))**.5)
		gs.append(g)

	fig = pylab.figure()
	ax = fig.add_axes((.1,.1,.8,.8))

	for mchirp,eta,g in zip(mchirps,etas,gs):
		plotErrorEllipse((mchirp,eta),g,maxmismatch,ax,edge='r')

	ax.plot(mchirps,etas,'kx')

	mmin = 1.0
	mmax = 3.0

	m1s = scipy.linspace(mmin,mmax,20)*metric.M_sun
	m2s = scipy.linspace(mmin,mmin,20)*metric.M_sun
	Ms = m1s + m2s
	etas = m1s * m2s / Ms**2
	mchirps = Ms * etas**.6
	ax.plot(mchirps, etas, 'b')

	m1s = scipy.linspace(mmax,mmax,20)*metric.M_sun
	m2s = scipy.linspace(mmin,mmax,20)*metric.M_sun
	Ms = m1s + m2s
	etas = m1s * m2s / Ms**2
	mchirps = Ms * etas**.6
	ax.plot(mchirps, etas, 'b')

	m1s = scipy.linspace(mmin,mmax,20)*metric.M_sun
	m2s = scipy.linspace(mmin,mmax,20)*metric.M_sun
	Ms = m1s + m2s
	etas = m1s * m2s / Ms**2
	mchirps = Ms * etas**.6
	ax.plot(mchirps, etas, 'b')

	ax.set_xlabel(r'$M_c$')
	ax.set_ylabel(r'$\eta$')
	ax.set_title('Mass Metric')

def mollwiede_map(ax):
	"""
	Create a mollwiede projection map of the earth for plotting ontop of.
	"""
	m = Basemap(projection='moll',lon_0=0,resolution='c', ax=ax)

	# draw coastlines, country boundaries, fill continents.
	m.drawcoastlines()
	m.fillcontinents(color='none', lake_color='none')
	m.drawcountries()
	m.drawmapboundary(fill_color='w') 
	# draw the edge of the map projection region (the projection limb)
	m.drawmapboundary()
	# draw lat/lon grid lines every 30 degrees.
	m.drawmeridians(scipy.arange(-180, 180, 30))
	m.drawparallels(scipy.arange(-90, 90, 30))

	return m

def plot_arms(m, detectors):
	"""
	Plot the arm directions of an interferometric GW detector
	at the correct location and with the correct orientation.
	"""
	for detector in detectors:
		vertex = scipy.array(detector.vertex)
		vertex /= sum(vertex**2)**.5

		for arm in [detector.xarm, detector.yarm]:
			n_arm = scipy.array(arm)
			arm = []
			xs = scipy.linspace(0,.05,10)
			for x in xs:
				point = x*n_arm + vertex
				point /= sum(point**2)**.5
				dec = pi/2 - scipy.arccos(point[2])
				RA = scipy.arctan2(point[1],point[0])

				arm.append([RA,dec])
			arm = scipy.array(arm)

			x, y = m(arm[:,0]*180./pi, arm[:,1]*180./pi)
			m.plot(x, y, 'k', linewidth=3)

def boundingbox(nrows, ncols, rownum, colnum):
	"""
	Create a bounding box for use in plots with many panels.
	Generalized past single digits from matplotlib's subplot
	routine.
	"""
	vtextspace = .05
	xspace = (1. - (ncols+1)*vtextspace)/ncols

	htextspace = .05
	yspace = (1. - (nrows+1)*htextspace)/nrows

	rownum = nrows + 1 - rownum 

	return (vtextspace*(colnum) + xspace*(colnum-1), htextspace*rownum + yspace*(rownum-1), xspace, yspace)

def plot_detector(detector):
	"""
	Plot different detector antenna responses in a new figure.
	"""
	fig = pylab.figure()

	ax = fig.add_axes(boundingbox(3,1,1,1))
	m = mollwiede_map(ax)
	dRA = 2*pi/200
	ddec = pi/200
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]
	z = metric.Fp(RAs, decs, detector)**2
	z += metric.Fx(RAs, decs, detector)**2
	z = z**.5
	levels = scipy.linspace(min(0,min(z.flatten())),max(z.flatten()),20)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m,[detector])
	ax.set_title(detector.name)

	ax = fig.add_axes(boundingbox(3,3,2,1))
	m = mollwiede_map(ax)
	dRA = 2*pi/200
	ddec = pi/200
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]
	z = metric.Fp(RAs, decs, detector)
	levels = scipy.linspace(min(0,min(z.flatten())),max(z.flatten()),20)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m,[detector])
	ax.set_title(r'$F_+$')

	ax = fig.add_axes(boundingbox(3,3,2,2))
	m = mollwiede_map(ax)
	dRA = 2*pi/200
	ddec = pi/200
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]
	z = metric.dFp_dRA(RAs, decs, detector)
	levels = scipy.linspace(min(0,min(z.flatten())),max(z.flatten()),20)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m,[detector])
	ax.set_title(r'$\partial_\alpha F_+$')

	ax = fig.add_axes(boundingbox(3,3,2,3))
	m = mollwiede_map(ax)
	dRA = 2*pi/200
	ddec = pi/200
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]
	z = metric.dFp_ddec(RAs, decs, detector)
	levels = scipy.linspace(min(0,min(z.flatten())),max(z.flatten()),20)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m,[detector])
	ax.set_title(r'$\partial_\delta F_+$')

	ax = fig.add_axes(boundingbox(3,3,3,1))
	m = mollwiede_map(ax)
	dRA = 2*pi/200
	ddec = pi/200
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]
	z = metric.Fx(RAs, decs, detector)
	levels = scipy.linspace(min(0,min(z.flatten())),max(z.flatten()),20)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m,[detector])
	ax.set_title(r'$F_\times$')

	ax = fig.add_axes(boundingbox(3,3,3,2))
	m = mollwiede_map(ax)
	dRA = 2*pi/200
	ddec = pi/200
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]
	z = metric.dFx_dRA(RAs, decs, detector)
	levels = scipy.linspace(min(0,min(z.flatten())),max(z.flatten()),20)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m,[detector])
	ax.set_title(r'$\partial_\alpha F_\times$')

	ax = fig.add_axes(boundingbox(3,3,3,3))
	m = mollwiede_map(ax)
	dRA = 2*pi/200
	ddec = pi/200
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]
	z = metric.dFx_ddec(RAs, decs, detector)
	levels = scipy.linspace(min(0,min(z.flatten())),max(z.flatten()),20)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m,[detector])
	ax.set_title(r'$\partial_\delta F_\times$')

def check_RA_derivatives(detector):
	"""
	Create a plot to check the RA directional derivatives.
	"""
	pylab.figure()
	dRA = 2*pi/100
	ddec = pi/100
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2+ddec:pi/2:ddec]
	dec0 = pi/3

	ax = pylab.subplot(231)
	z = metric.Fp(RAs, decs, detector)
	levels = 100
	CF = ax.contourf(RAs, decs, z, levels)
	CS = ax.contour(RAs, decs, z, levels)
	RAs = scipy.linspace(-pi,pi,100)
	decs = dec0*scipy.ones(len(RAs))
	CF = ax.plot(RAs,decs,'k')
	ax.set_title(r'$F_+$')

	ax = pylab.subplot(232)
	z = metric.Fp(RAs, decs, detector)
	CF = ax.plot(RAs, z)
	ax.set_title(r'$F_+$')

	ax = pylab.subplot(233)
	z = metric.dx_n2(z,RAs[1]-RAs[0])
	CF = ax.plot(RAs, z, 'r')
	z = metric.dFp_dRA(RAs, decs, detector)
	CF = ax.plot(RAs, z, 'b')
	ax.set_title(r'$\partial_\alpha F_+$')

	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2+ddec:pi/2:ddec]

	ax = pylab.subplot(234)
	z = metric.Fx(RAs, decs, detector)
	levels = 100
	CF = ax.contourf(RAs, decs, z, levels)
	CS = ax.contour(RAs, decs, z, levels)
	RAs = scipy.linspace(-pi,pi,100)
	decs = dec0*scipy.ones(len(RAs))
	CF = ax.plot(RAs,decs,'k')
	ax.set_title(r'$F_\times$')

	ax = pylab.subplot(235)
	z = metric.Fx(RAs, decs, detector)
	CF = ax.plot(RAs, z)
	ax.set_title(r'$F_\times$')

	ax = pylab.subplot(236)
	z = metric.dx_n2(z,RAs[1]-RAs[0])
	CF = ax.plot(RAs, z, 'r')
	z = metric.dFx_dRA(RAs, decs, detector)
	CF = ax.plot(RAs, z, 'b')
	ax.set_title(r'$\partial_\alpha F_\times$')

def check_dec_derivatives(detector):
	"""
	Create a plot to check the dec directional derivatives.
	"""
	pylab.figure()
	dRA = 2*pi/100
	ddec = pi/100
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2+ddec:pi/2:ddec]
	RA0 = pi/3

	ax = pylab.subplot(231)
	z = metric.Fp(RAs, decs, detector)
	levels = 100
	CF = ax.contourf(RAs, decs, z, levels)
	CS = ax.contour(RAs, decs, z, levels)
	decs = scipy.linspace(-pi/2,pi/2,100)
	RAs = RA0*scipy.ones(len(decs))
	CF = ax.plot(RAs,decs,'k')
	ax.set_title(r'$F_+$')

	ax = pylab.subplot(232)
	z = metric.Fp(RAs, decs, detector)
	CF = ax.plot(decs, z)
	ax.set_title(r'$F_+$')

	ax = pylab.subplot(233)
	z = metric.dx_n2(z,decs[1]-decs[0])
	CF = ax.plot(decs, z, 'r')
	z = metric.dFp_ddec(RAs, decs, detector)
	CF = ax.plot(decs, z, 'b')
	ax.set_title(r'$\partial_\delta F_+$')

	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2+ddec:pi/2:ddec]

	ax = pylab.subplot(234)
	z = metric.Fx(RAs, decs, detector)
	levels = 100
	CF = ax.contourf(RAs, decs, z, levels)
	CS = ax.contour(RAs, decs, z, levels)
	decs = scipy.linspace(-pi/2,pi/2,100)
	RAs = RA0*scipy.ones(len(decs))
	CF = ax.plot(RAs,decs,'k')
	ax.set_title(r'$F_\times$')

	ax = pylab.subplot(235)
	z = metric.Fx(RAs, decs, detector)
	CF = ax.plot(decs, z)
	ax.set_title(r'$F_\times$')

	ax = pylab.subplot(236)
	z = metric.dx_n2(z,decs[1]-decs[0])
	CF = ax.plot(decs, z, 'r')
	z = metric.dFx_ddec(RAs, decs, detector)
	CF = ax.plot(decs, z, 'b')
	ax.set_title(r'$\partial_\delta F_\times$')

def plot_metric(detectors, Fderivs):
	"""
	Plot the different components of the full metric as a function of sky
	location in different panels of the same figure.
	"""
	fig = pylab.figure(figsize=(16,9))
	dRA = 2*pi/200
	ddec = pi/200
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]
	levels = 20

	ax = fig.add_axes(boundingbox(5,5,1,1))
	m = mollwiede_map(ax)
	z = metric.g_RA_RA(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\alpha \alpha}$')

	ax = fig.add_axes(boundingbox(5,5,1,2))
	m = mollwiede_map(ax)
	z = metric.g_RA_dec(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\alpha \delta}$')

	ax = fig.add_axes(boundingbox(5,5,2,1))
	m = mollwiede_map(ax)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\delta \alpha}$')

	ax = fig.add_axes(boundingbox(5,5,1,3))
	m = mollwiede_map(ax)
	z = metric.g_RA_t(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\alpha t}$')

	ax = fig.add_axes(boundingbox(5,5,3,1))
	m = mollwiede_map(ax)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{t \alpha}$')

	ax = fig.add_axes(boundingbox(5,5,1,4))
	m = mollwiede_map(ax)
	z = metric.g_RA_mchirp(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\alpha M_c}$')

	ax = fig.add_axes(boundingbox(5,5,4,1))
	m = mollwiede_map(ax)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{M_c \alpha}$')

	ax = fig.add_axes(boundingbox(5,5,1,5))
	m = mollwiede_map(ax)
	z = metric.g_RA_eta(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\alpha \eta}$')

	ax = fig.add_axes(boundingbox(5,5,5,1))
	m = mollwiede_map(ax)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\eta \alpha}$')

	ax = fig.add_axes(boundingbox(5,5,2,2))
	m = mollwiede_map(ax)
	z = metric.g_dec_dec(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\delta \delta}$')

	ax = fig.add_axes(boundingbox(5,5,2,3))
	m = mollwiede_map(ax)
	z = metric.g_dec_t(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\delta t}$')

	ax = fig.add_axes(boundingbox(5,5,3,2))
	m = mollwiede_map(ax)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{t \delta}$')

	ax = fig.add_axes(boundingbox(5,5,2,4))
	m = mollwiede_map(ax)
	z = metric.g_dec_mchirp(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\delta M_c}$')

	ax = fig.add_axes(boundingbox(5,5,4,2))
	m = mollwiede_map(ax)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{M_c \delta}$')

	ax = fig.add_axes(boundingbox(5,5,2,5))
	m = mollwiede_map(ax)
	z = metric.g_dec_eta(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\delta \eta}$')

	ax = fig.add_axes(boundingbox(5,5,5,2))
	m = mollwiede_map(ax)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\eta \delta}$')

	ax = fig.add_axes(boundingbox(5,5,3,3))
	m = mollwiede_map(ax)
	z = metric.g_t_t(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{t t}$')

	ax = fig.add_axes(boundingbox(5,5,3,4))
	m = mollwiede_map(ax)
	z = metric.g_t_mchirp(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{t M_c}$')

	ax = fig.add_axes(boundingbox(5,5,4,3))
	m = mollwiede_map(ax)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{M_c t}$')

	ax = fig.add_axes(boundingbox(5,5,3,5))
	m = mollwiede_map(ax)
	z = metric.g_t_eta(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{t \eta}$')

	ax = fig.add_axes(boundingbox(5,5,5,3))
	m = mollwiede_map(ax)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\eta t}$')

	ax = fig.add_axes(boundingbox(5,5,4,4))
	m = mollwiede_map(ax)
	z = metric.g_mchirp_mchirp(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{M_c M_c}$')

	ax = fig.add_axes(boundingbox(5,5,4,5))
	m = mollwiede_map(ax)
	z = metric.g_mchirp_eta(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{M_c \eta}$')

	ax = fig.add_axes(boundingbox(5,5,5,4))
	m = mollwiede_map(ax)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\eta M_c}$')

	ax = fig.add_axes(boundingbox(5,5,5,5))
	m = mollwiede_map(ax)
	z = metric.g_eta_eta(RAs, decs, detectors, Fderivs=Fderivs)
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, z, levels)
	plot_arms(m, detectors)
	ax.set_title(r'$g_{\eta \eta}$')

def check_toa(detectors):
	"""
	Create a figure to plot the times of arrival of a GW signal
	as a function of sky location.
	"""
	fig = pylab.figure(figsize=(6,6))

	dRA = 2*pi/200
	ddec = pi/200
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]

	for idx,detector in enumerate(detectors):
		ax = fig.add_axes(boundingbox(len(detectors),3,idx+1,1))
		m = mollwiede_map(ax)
		levels = 20
		z = metric.rn(RAs, decs, detector)
		x, y = m(RAs*180./pi, decs*180./pi)
		CS = m.contour(x, y, z, levels)
		plot_arms(m, [detector])
		ax.set_title(r'$\vec{r_{\rm %s}} \cdot \hat{n}$'%(detector.name))

		ax = fig.add_axes(boundingbox(len(detectors),3,idx+1,2))
		m = mollwiede_map(ax)
		levels = 20
		z = metric.drn_dRA(RAs, decs, detector)
		x, y = m(RAs*180./pi, decs*180./pi)
		CS = m.contour(x, y, z, levels)
		plot_arms(m, [detector])
		ax.set_title(r'$\partial_\alpha (\vec{r_{\rm %s}} \cdot \hat{n})$'%(detector.name))

		ax = fig.add_axes(boundingbox(len(detectors),3,idx+1,3))
		m = mollwiede_map(ax)
		levels = 20
		z = metric.drn_ddec(RAs, decs, detector)
		x, y = m(RAs*180./pi, decs*180./pi)
		CS = m.contour(x, y, z, levels)
		plot_arms(m, [detector])
		ax.set_title(r'$\partial_\delta (\vec{r_{\rm %s}} \cdot \hat{n})$'%(detector.name))

def plot_maximum_likelihood_matrix(detectors):
	"""
	Create a figure to plot the components of the maximum likelihood
	matrix as a function of sky location.
	"""
	fig = pylab.figure(figsize=(8,4.5))

	dRA = 2*pi/200
	ddec = pi/200
	RAs,decs = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]
	A,B,C,D = metric.maximum_likelihood_matrix(RAs, decs, detectors)

	ax = fig.add_axes(boundingbox(2,2,1,1))
	m = mollwiede_map(ax)
	levels = 20
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, A, levels)
	plot_arms(m, detectors)
	ax.set_title('$A \in [%.2e, %.2e]$'%(min(A.flatten()), max(A.flatten())))

	ax = fig.add_axes(boundingbox(2,2,1,2))
	m = mollwiede_map(ax)
	levels = 20
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, B, levels)
	plot_arms(m, detectors)
	ax.set_title('$B \in [%.2e, %.2e]$'%(min(B.flatten()), max(B.flatten())))

	ax = fig.add_axes(boundingbox(2,2,2,1))
	m = mollwiede_map(ax)
	levels = 20
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, C, levels)
	plot_arms(m, detectors)
	ax.set_title('$C \in [%.2e, %.2e]$'%(min(C.flatten()), max(C.flatten())))

	ax = fig.add_axes(boundingbox(2,2,2,2))
	m = mollwiede_map(ax)
	levels = 20
	x, y = m(RAs*180./pi, decs*180./pi)
	CS = m.contour(x, y, scipy.log(D), levels)
	plot_arms(m, detectors)
	ax.set_title('$\log(D) ,\, D \in [%.2e, %.2e]$'%(min(D.flatten()), max(D.flatten())))

def plot_metric_component_pieces(RA, dec, z0, z1, z2, z3):
	"""
	Create a figure to plot the different components of a 2D metric as a
	function of sky location.
	"""
	Z = scipy.zeros(scipy.shape(RA))
	fig = pylab.figure(figsize=(6,6))
	for idx in range(len(detectors)):
		ax = fig.add_axes(boundingbox(len(detectors),2,idx+1,1))
		m = mollwiede_map(ax)
		levels = 20
		x, y = m(RA*180./pi, dec*180./pi)
		CS = m.contour(x, y, z0[idx], levels)
		plot_arms(m, detectors)

		Z += z0[idx]

		ax = fig.add_axes(boundingbox(len(detectors),2,idx+1,2))
		m = mollwiede_map(ax)
		levels = 20
		x, y = m(RA*180./pi, dec*180./pi)
		CS = m.contour(x, y, z1[idx], levels)
		plot_arms(m, detectors)

		Z += z1[idx]

	fig = pylab.figure(figsize=(6,6))
	for idx in range(len(detectors)):
		for jdx in range(len(detectors)):
			ax = fig.add_axes(boundingbox(len(detectors),len(detectors),idx+1,jdx+1))
			m = mollwiede_map(ax)
			levels = 20
			x, y = m(RA*180./pi, dec*180./pi)
			CS = m.contour(x, y, z2[idx][jdx], levels)
			plot_arms(m, detectors)

			Z += z2[idx][jdx]

	fig = pylab.figure(figsize=(6,6))
	for idx in range(len(detectors)):
		for jdx in range(len(detectors)):
			ax = fig.add_axes(boundingbox(len(detectors),len(detectors),idx+1,jdx+1))
			m = mollwiede_map(ax)
			levels = 20
			x, y = m(RA*180./pi, dec*180./pi)
			CS = m.contour(x, y, z3[idx][jdx], levels)
			plot_arms(m, detectors)

			Z += z3[idx][jdx]

	fig = pylab.figure(figsize=(6,6))

	ax = fig.add_axes(boundingbox(3,2,1,1))
	m = mollwiede_map(ax)
	levels = 20
	x, y = m(RA*180./pi, dec*180./pi)
	CS = m.contour(x, y, sum(z0), levels)
	plot_arms(m, detectors)

	ax = fig.add_axes(boundingbox(3,2,1,2))
	m = mollwiede_map(ax)
	levels = 20
	x, y = m(RA*180./pi, dec*180./pi)
	CS = m.contour(x, y, sum(z1), levels)
	plot_arms(m, detectors)

	ax = fig.add_axes(boundingbox(3,1,2,1))
	m = mollwiede_map(ax)
	levels = 20
	x, y = m(RA*180./pi, dec*180./pi)
	CS = m.contour(x, y, Z, levels)
	plot_arms(m, detectors)

	ax = fig.add_axes(boundingbox(3,2,3,1))
	m = mollwiede_map(ax)
	levels = 20
	x, y = m(RA*180./pi, dec*180./pi)
	CS = m.contour(x, y, sum([sum(z) for z in z2]), levels)
	plot_arms(m, detectors)

	ax = fig.add_axes(boundingbox(3,2,3,2))
	m = mollwiede_map(ax)
	levels = 20
	x, y = m(RA*180./pi, dec*180./pi)
	CS = m.contour(x, y, sum([sum(z) for z in z3]), levels)
	plot_arms(m, detectors)

def check_g_RA_RA(detectors):
	"""
	Create a figure to plot the RA RA component of the amplitude maximized
	metric.
	"""
	dRA = 2*pi/200
	ddec = pi/200
	RA,dec = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]

	A,B,C,D = metric.maximum_likelihood_matrix(RA, dec, detectors)

	z0 = []
	z1 = []
	z2 = []
	z3 = []
	Fderivs = True
	for detector1 in detectors:
		I_n1 = detector1.I_n
		fp1 = metric.Fp(RA, dec, detector1)
		fx1 = metric.Fx(RA, dec, detector1)
		if Fderivs:
			dfp_dRA1 = metric.dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = metric.dFx_dRA(RA, dec, detector1)
		drn_dRA1 = metric.drn_dRA(RA, dec, detector1)
		z0.append((B*fp1*fp1+A*fx1*fx1-2*C*fp1*fx1)*(-2*pi*drn_dRA1)**2*I_n1['-1']/D)
		if Fderivs:
			z1.append((B*dfp_dRA1*dfp_dRA1+A*dfx_dRA1*dfx_dRA1-C*dfp_dRA1*dfx_dRA1-C*dfx_dRA1*dfp_dRA1)*I_n1['-7']/D)
		z2.append([])
		z3.append([])
		for detector2 in detectors:
			I_n2 = detector2.I_n
			fp2 = metric.Fp(RA, dec, detector2)
			fx2 = metric.Fx(RA, dec, detector2)
			if Fderivs:
				dfp_dRA2 = metric.dFp_dRA(RA, dec, detector2)
				dfx_dRA2 = metric.dFx_dRA(RA, dec, detector2)
			drn_dRA2 = metric.drn_dRA(RA, dec, detector2)
			pre = B*fp1*fp2+A*fx1*fx2-C*fp1*fx2-C*fx1*fp2
			z2[-1].append(pre*(B*fp1*fp2+A*fx1*fx2-C*fp1*fx2-C*fx1*fp2)/D**2 \
				*I_n1['-4']*I_n2['-4']*(-2*pi*drn_dRA1)*(-2*pi*drn_dRA2))
			if Fderivs:
				z3[-1].append(pre*(B*dfp_dRA1*dfp_dRA2+A*dfx_dRA1*dfx_dRA2-C*dfp_dRA1*dfx_dRA2-C*dfx_dRA1*dfp_dRA2)/D**2 \
					*I_n1['-7']*I_n2['-7'])

	plot_metric_component_pieces(RA, dec, z0, z1, z2, z3)

def check_g_RA_dec(detectors):
	"""
	Create a figure to plot the RA dec component of the
	amplitude-maximized amplitude-averaged metric.
	"""
	dRA = 2*pi/200
	ddec = pi/200
	RA,dec = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]

	A,B,C,D = metric.maximum_likelihood_matrix(RA, dec, detectors)

	z0 = []
	z1 = []
	z2 = []
	z3 = []
	Fderivs = True
	for detector1 in detectors:
		I_n1 = detector1.I_n
		fp1 = metric.Fp(RA, dec, detector1)
		fx1 = metric.Fx(RA, dec, detector1)
		if Fderivs:
			dfp_dRA1 = metric.dFp_dRA(RA, dec, detector1)
			dfx_dRA1 = metric.dFx_dRA(RA, dec, detector1)
			dfp_ddec1 = metric.dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = metric.dFx_ddec(RA, dec, detector1)
		drn_dRA1 = metric.drn_dRA(RA, dec, detector1)
		drn_ddec1 = metric.drn_ddec(RA, dec, detector1)
		z0.append((B*fp1*fp1+A*fx1*fx1-2*C*fp1*fx1)/D*I_n1['-1']*(-2*pi*drn_dRA1)*(-2*pi*drn_ddec1))
		if Fderivs:
			z1.append((B*dfp_dRA1*dfp_ddec1+A*dfx_dRA1*dfx_ddec1-C*dfp_dRA1*dfx_ddec1-C*dfx_dRA1*dfp_ddec1)/D \
				*I_n1['-7'])
		z2.append([])
		z3.append([])
		for detector2 in detectors:
			I_n2 = detector2.I_n
			fp2 = metric.Fp(RA, dec, detector2)
			fx2 = metric.Fx(RA, dec, detector2)
			if Fderivs:
				dfp_ddec2 = metric.dFp_ddec(RA, dec, detector2)
				dfx_ddec2 = metric.dFx_ddec(RA, dec, detector2)
			drn_ddec2 = metric.drn_ddec(RA, dec, detector2)
			pre = B*fp1*fp2+A*fx1*fx2-C*fp1*fx2-C*fx1*fp2
			z2[-1].append(pre*(B*fp1*fp2+A*fx1*fx2-C*fp1*fx2-C*fx1*fp2)/D**2 \
				*I_n1['-4']*I_n2['-4']*(-2*pi*drn_dRA1)*(-2*pi*drn_ddec2))
			if Fderivs:
				z3[-1].append(pre*(B*dfp_dRA1*dfp_ddec2+A*dfx_dRA1*dfx_ddec2-C*dfp_dRA1*dfx_ddec2-C*dfx_dRA1*dfp_ddec2)/D**2 \
					*I_n1['-7']*I_n2['-7'])

	plot_metric_component_pieces(RA, dec, z0, z1, z2, z3)

def check_g_dec_dec(detectors):
	"""
	Create a figure to plot the dec dec component of the
	amplitude-maximized amplitude-averaged metric.
	"""
	dRA = 2*pi/200
	ddec = pi/200
	RA,dec = scipy.mgrid[-pi+dRA:pi:dRA, -pi/2:pi/2+ddec:ddec]

	A,B,C,D = metric.maximum_likelihood_matrix(RA, dec, detectors)

	z0 = []
	z1 = []
	z2 = []
	z3 = []
	Fderivs = True
	for detector1 in detectors:
		I_n1 = detector1.I_n
		fp1 = metric.Fp(RA, dec, detector1)
		fx1 = metric.Fx(RA, dec, detector1)
		if Fderivs:
			dfp_ddec1 = metric.dFp_ddec(RA, dec, detector1)
			dfx_ddec1 = metric.dFx_ddec(RA, dec, detector1)
		drn_ddec1 = metric.drn_ddec(RA, dec, detector1)
		z0.append((B*fp1*fp1+A*fx1*fx1-2*C*fp1*fx1)/D*I_n1['-1']*(-2*pi*drn_ddec1)**2)
		if Fderivs:
			z1.append((B*dfp_ddec1*dfp_ddec1+A*dfx_ddec1*dfx_ddec1-C*dfp_ddec1*dfx_ddec1-C*dfx_ddec1*dfp_ddec1)/D \
				*I_n1['-7'])
		z2.append([])
		z3.append([])
		for detector2 in detectors:
			I_n2 = detector2.I_n
			fp2 = metric.Fp(RA, dec, detector2)
			fx2 = metric.Fx(RA, dec, detector2)
			if Fderivs:
				dfp_ddec2 = metric.dFp_ddec(RA, dec, detector2)
				dfx_ddec2 = metric.dFx_ddec(RA, dec, detector2)
			drn_ddec2 = metric.drn_ddec(RA, dec, detector2)
			pre = B*fp1*fp2+A*fx1*fx2-C*fp1*fx2-C*fx1*fp2
			z2[-1].append(pre*(B*fp1*fp2+A*fx1*fx2-C*fp1*fx2-C*fx1*fp2)/D**2 \
				*I_n1['-4']*I_n2['-4']*(-2*pi*drn_ddec1)*(-2*pi*drn_ddec2))
			if Fderivs:
				z3[-1].append(pre*(B*dfp_ddec1*dfp_ddec2+A*dfx_ddec1*dfx_ddec2-C*dfp_ddec1*dfx_ddec2-C*dfx_ddec1*dfp_ddec2)/D**2 \
					*I_n1['-7']*I_n2['-7'])

	plot_metric_component_pieces(RA, dec, z0, z1, z2, z3)



