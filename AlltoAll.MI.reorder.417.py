import sys
import numpy
from math import pi
from numpy import *
import sys
import os
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

#HB=numpy.loadtxt(sys.argv[1]).astype(int)
#Charge=numpy.loadtxt(sys.argv[2]).astype(int)
#Stack=numpy.loadtxt(sys.argv[3]).astype(int)
#Hphob=numpy.loadtxt(sys.argv[4]).astype(int)
#
#TotalInt=numpy.hstack((HB,Charge,Stack,Hphob))

  
#TotalInt=numpy.hstack((HB,Charge,Stack,Hphob))
#numpy.savetxt("TotaInt.reorder.dat",TotalInt,'%i')

TotalInt=numpy.loadtxt("TotaInt.reorder.417.dat")
frames=float(len(TotalInt[:,0]))
features=len(TotalInt[0])
probx1=TotalInt.sum(axis=0)/frames
probx0=1-probx1
probxy11=numpy.zeros((features,features))
probxy00=numpy.zeros((features,features))
probxy01=numpy.zeros((features,features))
probxy10=numpy.zeros((features,features))

for i in range(0,features):
  a=numpy.where(TotalInt[:,i]==(1))
  b=numpy.where(TotalInt[:,i]==(0))
  for j in range(0,features):
    c=numpy.where(TotalInt[:,j]==(1))
    d=numpy.where(TotalInt[:,j]==(0))
    probxy11[i][j]=len(numpy.intersect1d(a,c))/frames
    probxy00[i][j]=len(numpy.intersect1d(b,d))/frames
    probxy01[i][j]=len(numpy.intersect1d(b,c))/frames
    probxy10[i][j]=len(numpy.intersect1d(a,d))/frames

MI=numpy.zeros((features,features))
for i in range(0,features):
  px1=probx1[i]
  px0=probx0[i]
  for j in range(0,features):
    py1=probx1[j]
    py0=probx0[j]
    pxy11=probxy11[i][j]
    pxy00=probxy00[i][j]
    pxy01=probxy01[i][j]
    pxy10=probxy10[i][j]
    MI11=0
    MI10=0
    MI01=0
    MI00=0
    H11=0
    H10=0
    H01=0
    H00=0
    if (px1!=0 and py1!=0 and pxy11!= 0):
     MI11=pxy11*numpy.log2(pxy11/(px1*py1))
     H11=pxy11*numpy.log2(pxy11)
    if (px0!=0 and py0!=0 and pxy00!= 0):
     MI00=pxy00*numpy.log2(pxy00/(px0*py0))
     H00=pxy00*numpy.log2(pxy00)
    if (px0!=0 and py1!=0 and pxy01!= 0):
     MI01=pxy01*numpy.log2(pxy01/(px0*py1))
     H01=pxy01*numpy.log2(pxy01)
    if (px1!=0 and py0!=0 and pxy10!= 0):
     MI10=pxy10*numpy.log2(pxy10/(px1*py0))
     H10=pxy10*numpy.log2(pxy10)
    H=H11+H00+H01+H10
    if(H!=0):
     MI[i][j]=(MI11+MI00+MI01+MI10)/(-H)
    
    #print i,j, MI[i][j],pxy11,pxy00    
#MI[isnan(MI)]=0

numpy.savetxt('MI.Snorm.reorder.417.dat',MI,'%f')

plt.clf()
#plt.title('Pythonspot.com heatmap example')
#plt.xlabel(r'r_OF ($\AA$)')
#plt.ylabel(r'r_OH ($\AA$)')
extent = (0, 80, 0, 80)
plt.imshow(numpy.flipud(MI), extent=extent, interpolation='None', aspect=1.0,cmap='jet')
plt.yticks([0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76],('D121','N122','E123','A124','Y125','E126','M127','P128','S129','E130','E131','G132','Y133','Q134','D135','Y136','E137','P138','E139','A140'),va='bottom')
plt.xticks([0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76],('D121','N122','E123','A124','Y125','E126','M127','P128','S129','E130','E131','G132','Y133','Q134','D135','Y136','E137','P138','E139','A140'),rotation=90,ha='left')
plt.tick_params(axis='x', color='white',labelcolor='black',length=10,top='on')
plt.tick_params(axis='y', color='white',labelcolor='black',length=10,right='on')
plt.clim(vmin=0,vmax=0.10)
plt.colorbar()
plt.show()
plt.savefig('MI.Snorm.reorder.417.png', format='png', dip=1000)
#.figure()
