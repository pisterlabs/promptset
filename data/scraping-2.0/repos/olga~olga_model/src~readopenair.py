#
# Copyright (c) 2013-2014 Bart van Stratum (bart@vanstratum.com)
# 
# This file is part of OLGA.
# 
# OLGA is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# OLGA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with OLGA.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
from pylab import *
from matplotlib.patches import Circle, Wedge, Polygon, Arc
from matplotlib.collections import PatchCollection
from math import radians, cos, sin, asin, sqrt
from mpl_toolkits.basemap import Basemap

from tools import *

close('all')

"""
Main function
Plot airspace from OpenAir format
"""
def plotairspace(m):
  plotairspace = ['CTR','A','B','C','D']

  #databases = ['EHEDv13_5.txt','EDv13_4.txt']
  databases = ['data/EHEDv13_5.txt','data/BELLUX_WEEK_130404.txt']
  #databases = ['BELLUX_WEEK_130404.txt']

  """
  Loop through databases
  """
  for data in databases:
    f = open(data,'r')  
 
    """
    Loop through lines in file
    """
    doread = False
    debug = False
    poly_lats = []
    poly_lons = []
    l=-1
    for line in f:
      l+=1
      # #$@#$% do bunch of replacements...
      data = line.replace('N',' N ')    
      data = data.replace('E',' E ')    
      data = data.replace(',',' , ')   
      data = data.replace('X= ' ,'X=')   
      data = data.split()

      if (len(data)>0):
    
        if(data[0]=='AC'):       # Read airspace type
          if(len(poly_lats)>0):  # Close previous airspace
            # 1. close polygon
            poly_lons.append(poly_lons[0])
            poly_lats.append(poly_lats[0])
            # 2. basemap transform
            poly_lons,poly_lats = m(poly_lons,poly_lats)
            # 3. plot
            if(atype == 'CTR'):
              fill(poly_lons,poly_lats,color='k',alpha=0.1)
              m.plot(poly_lons,poly_lats,'k',linewidth=0.5)
            elif(atype in ['A','B','C','D']):
              m.plot(poly_lons,poly_lats,color='k',linewidth=0.5,alpha=0.5)

            # clear list with coordinates
            poly_lats = []
            poly_lons = []

          """
          Check if airspace in list "plotairspace"
          if not, doread = False until next valid airspace is found
          """
          if(data[1] in plotairspace):
            doread = True
            atype = data[1]
          else:
            doread = False
 
        """
        Proceed if valid airspace, and not a comment ("#")
        """
        if(doread and data[0] != '#'): 
          if(debug):
            print l,'----',data,'----'
 
          """
          DP = new polygon point
          """ 
          if(data[0]=='DP'):
            poly_lats.append(ctrans(data[1]))
            poly_lons.append(ctrans(data[3]))
    
            if(debug):
              print 'new polygon point lat/lon = %.2f,%.2f'%(poly_lats[-1],poly_lons[-1])
 
          """
          V = either indicator of cw/ccw, of center of new feature (e.g. arc)
          """ 
          if(data[0]=='V'):
            if(data[1][0] == 'D'):                   # Set direction cw or ccw
              direction = +1 if(data[1][-1] == '+') else -1
            elif(data[1][0] == 'X'):                    # center of feature   
              # save center of feature for later use 
              clat = ctrans(data[1])
              clon = ctrans(data[3])

              if(debug): 
                print 'new center lat/lon = %.2f,%.2f'%(clat,clon)
    
          """
          DB: start and end coordinates arc
          """
          if(data[0]=='DB'):
            lat1 = ctrans(data[1])
            lon1 = ctrans(data[3])
            lat2 = ctrans(data[6])
            lon2 = ctrans(data[8])
              

            # add beginning of arc to coordinate list:
            poly_lats.append(lat1)
            poly_lons.append(lon1)

            """
            calculate individual points or arc
            """
            d1   = haversine(clon,clat,lon1,lat1)    # radius (km) of arc
            th1  = angle(clon,clat,lon1,lat1)        # start angle
            th2  = angle(clon,clat,lon2,lat2)        # end angle

            if(th1>th2):
              dth = (360-th1)+th2 if(direction>0) else th2-th1 
            elif(th1<th2):
              dth = th2-th1 if(direction>0) else ((360-th2)+th1)

            nsec = int(abs(dth)/5.)                   # number of sections in curve 
            for i in range(nsec-1):
              th1 = th1 + dth/float(nsec)
 
              # calculate coordinate point arc and add to list
              nlon,nlat = point(clon,clat,d1,th1)
              poly_lats.append(nlat)
              poly_lons.append(nlon) 
              
            # add end of arc to coordinate list:
            poly_lats.append(lat2)
            poly_lons.append(lon2)
               
            if(debug):
              print 'new coordinates arc = %.2f,%.2f / %.2f,%.2f'%(lat1,lon1,lat2,lon2)
 
"""
if main; test plot
""" 
if __name__ == "__main__":
  fig   = plt.figure(figsize=(6.5,6.0))
  m = Basemap(width=590000,height=590000,
              rsphere=(6378137.00,6356752.3142),\
              resolution='i',area_thresh=10.,projection='lcc',\
              lat_1=51.3,lat_2=51.3,lat_0=51.3,lon_0=6.7)
  axloc = [0.03,0.05,0.85,0.87]  # left,bottom,width,height
  ax    = fig.add_axes(axloc)
  
  m.drawrivers(linewidth=0.5,color='#0066FF')
  m.drawmeridians(arange(0, 360, 5))
  m.drawparallels(arange(30, 60, 5))
  m.drawcoastlines(linewidth=1.5,color='0.2')
  m.drawcountries(linewidth=1,color='0.2')
  m.drawmapboundary()

  plotairspace(m)
