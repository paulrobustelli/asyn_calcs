#!/usr/bin/env python
# coding: utf-8

# In[43]:

from __future__ import print_function, division
import mdtraj as md
from mdtraj.utils import ensure_type
from mdtraj.geometry import compute_distances, compute_angles
from mdtraj.geometry import _geometry
import os
import sys
import numpy as np
import scipy as sp
from scipy import optimize
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import math
import itertools    
from numpy import log2, zeros, mean, var, sum, loadtxt, arange,                   array, cumsum, dot, transpose, diagonal, floor
from numpy.linalg import inv, lstsq

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# In[2]:


def block(x):
    # preliminaries
    d = log2(len(x))
    if (d - floor(d) != 0):
    #    print("Warning: Data size = %g, is not a power of 2." % floor(2**d))
    #    print("Truncating data to %g." % 2**floor(d) )
        x = x[:2**int(floor(d))]
    d = int(floor(d))
    n = 2**d
    s, gamma = zeros(d), zeros(d)
    mu = mean(x)
    # estimate the auto-covariance and variances 
    # for each blocking transformation
    for i in arange(0,d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*sum( (x[0:(n-1)]-mu)*(x[1:n]-mu) )
        # estimate variance of x
        s[i] = var(x)
        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (cumsum( ((gamma/s)**2*2**arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
    q =array([6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
              16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
              24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
              31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
              38.932173, 40.289360, 41.638398, 42.979820, 44.314105,
              45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in arange(0,d):
        if(M[k] < q[k]):
            break
    if (k >= d-1):
        print("Warning: Use more data")

    return (s[k]/2**(d-k))


# In[3]:


trajdir='/dartfs-hpc/rc/home/k/f0044gk/labhome/DESRES_Trajectories/biorxiv2021-5475823-no-water-glue'
pdb='/dartfs-hpc/rc/home/k/f0044gk/labhome/DESRES_Trajectories/biorxiv2021-5475823-no-water-glue/asyn_ligand41_nowater.pdb'
trj = md.load("%s/%s"%(trajdir,'asyn121to140_fasudil.dcd'),stride=1, top=pdb)

#workdir='/Users/paulrobustelli/Asyn_LigandBinding'
#trajdir='/Users/paulrobustelli/Desktop/Trajectories/Asyn_DESRES/DESRES-Trajectory_biorxiv2021-asyn121to139_Lig41_5475823-no-water-glue/biorxiv2021-5475823-no-water-glue'
#pdb='/Users/paulrobustelli/Desktop/Trajectories/Asyn_DESRES/DESRES-Trajectory_biorxiv2021-asyn121to139_Lig41_5475823-no-water-glue/biorxiv2021-5475823-no-water-glue/asyn_ligand41_nowater.pdb'


# In[4]:


trj.center_coordinates()
top = trj.topology
first_frame = 0
last_frame = trj.n_frames
n_frames=trj.n_frames


# In[5]:


nres=[]
for res in trj.topology.residues: nres.append(res.resSeq)
sequence=(' %s' % [residue for residue in trj.topology.residues])
resname=(' %s' % [residue.name for residue in trj.topology.residues])
resindex=(' %s' % [residue.index for residue in trj.topology.residues])
prot_top=top.subset(top.select('protein'))
prot_res=[]
for res in prot_top.residues: prot_res.append(res.resSeq)
prot_resname=(' %s' % [residue.name for residue in prot_top.residues])
residues=len(set(prot_res))

#log = open("/Users/paulrobustelli/Desktop/Sa_calc.log", "w")
print("** SYSTEM INFO **\n")
print("Number of atoms: %d\n" % trj.n_atoms)
print("Number of residues: %d\n" % len(set(nres)))
print("Number of protein residues: %d\n" % len(set(prot_res)))
print("Number of frames: %d\n" % trj.n_frames)
print("Starting frame: %d\n" % first_frame)
print("Last frame: %d\n" % last_frame)
print("sequence: %s\n" % sequence)
print("residue names: %s\n" % resname)
print("residue index: %s\n" % resindex)

residues=20
residue_offset=121
residue_number = range(residue_offset, residue_offset+residues)
print("Residue Number Range:",residue_number)


# In[6]:


#Compute Phi and Psi
indices_phi, phis = md.compute_phi(trj)
indices_psi, psis = md.compute_psi(trj)
#print(indices_phi,indices_psi)
#print(phis)
phi_label=[]
  # cycle on the number of phi dihedrals (n_phi)
for i_phi in range(0, indices_phi.shape[0]):
      resindex=trj.topology.atom(indices_phi[i_phi][3]).residue.resSeq
      phi_label.append(resindex)
  # convert to numpy array
phi_label=np.array(phi_label)
psi_label=[]
for i_psi in range(0, indices_psi.shape[0]):
      resindex=trj.topology.atom(indices_psi[i_psi][3]).residue.resSeq
      psi_label.append(resindex)
  # convert to numpy array
psi_label=np.array(psi_label)
phipsi=[]
for i in range(0,len(psi_label)-1):
  current_phipsi=np.column_stack((phis[:,i+1],psis[:,i]))
  phipsi.append(current_phipsi)
  #np.savetxt(dihedraldir+"/PhiPsi.%s.dat"%psi_label[i], current_phipsi,fmt='%.4lf')
phipsi_array=np.array(phipsi)


#residue_num=14
#print(psi_label[residue_num],phi_label[residue_num],trj.topology.residue(residue_num+1))
#hist=plt.hist2d(phipsi_array[residue_num][:,0], phipsi_array[residue_num][:,1], bins=36,range=[[-3.14,3.14],[-3.14,3.14]], norm=colors.LogNorm(),cmap='jet')


# In[7]:


#Select Ligand Residues
ligand=top.select("residue 0")
#Select Protein Residues
protein=top.select("residue 121 to 140")
#table, bonds = top.to_dataframe()
#table[table['resName'] == '*']

ligand_atomid = []
for atom in ligand:
    indices = []
    indices.append(atom)
    indices.append(top.atom(atom))
    ligand_atomid.append(indices)


protein_atomid = []
for atom in protein:
    indices = []
    indices.append(atom)
    indices.append(top.atom(atom))
    protein_atomid.append(indices)

print(ligand_atomid)
print(protein_atomid)
#Ligand Charged atom is N-296
#Attached Amine Hydrogens are 318 and 331
#Sulfoynl Oxygens are O-302 and O-303
#Sulfonyl S is S-301
#Ring 1 is C-304-309
#Ring 2 is C-308-312
#Single Ring is 304-312


# In[18]:


contact_pairs=np.zeros((residues,2))
ligand_residue_index=21

for i in range (0,residues):
  contact_pairs[i]=[i,ligand_residue_index]
contact = md.compute_contacts(trj, contact_pairs,scheme='closest')
contacts= np.asarray(contact[0]).astype(float)
cutoff=0.5
contact_frames=np.where(contacts < 0.5, 1, 0)
contact_prob = np.sum(contact_frames,axis=0)/trj.n_frames

contact_rows=np.sum(contact_frames,axis=1) 
contact_frames=np.where(contact_rows>0)
contact_fraction=np.sum(np.where(contact_rows>0,1,0))/len(contact_rows)
print("Fraction Bound (Closest Atom, 5A cutof):",contact_fraction)
np.savetxt('contact_frames,closest.5A.dat',contact_frames,'%i')

#Exact Desres definition: 
#'{system=workdir/anton_start.dms stk=workdir/run.stk}@args.selection="not water"@args.std.center.selection="protein"' 
#dist --distance "protein" "not protein and not water and not ion and not hydrogen and not resname ACE"


# In[27]:


desres_bf=42828/58468.0
print("Contact Fraction:",contact_fraction,"DESRES Distmin Bound Fraction:",desres_bf)


# In[41]:


#Calculate Contact Probabilities between each protein residue and the ligand
#residues=20
#residue_offset=121
#residue_number = range(residue_offset, residue_offset+residues)
contact_pairs=np.zeros((residues,2))
ligand_residue_index=21

for i in range (0,residues):
  contact_pairs[i]=[i,ligand_residue_index]
contact = md.compute_contacts(trj, contact_pairs,scheme='closest-heavy')
contacts= np.asarray(contact[0]).astype(float)
cutoff=0.6
contact_frames=np.where(contacts < cutoff, 1, 0)
contact_prob = np.sum(contact_frames,axis=0)/trj.n_frames

contact_rows=np.sum(contact_frames,axis=1) 
contact_fraction=np.sum(np.where(contact_rows>0,1,0))/len(contact_rows)
print("Fraction Bound:",contact_fraction)


np.savetxt('contacts.traj.dat',contact_frames)
np.savetxt('contacts.dat',np.column_stack((residue_number, contact_prob)))
contacts_by_res=np.column_stack((residue_number, contact_prob))


plt.plot(residue_number, contact_prob)
plt.xlabel('Residue', size=18)
plt.ylabel('Contact Probability', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.tight_layout()
#plt.show()
plt.savefig('ContactFraction.byResidue.ClosestHeavy.png')
plt.clf()
#plt.show()
#for i in residue_number:
#    contact = md.compute_contacts(traj, [[i, 0]])
#    array = np.asarray(contact[0]).astype(float)
#    prob = np.sum(np.where(array < 0.5, 1, 0))/len(array)
#    probability.append(prob)


# In[42]:


#Calculate Charge-Charge Contacts 

#Select Ligand Charge Groups
#Ligand Charged atom is N-296
Ligand_Pos_Charges=[296]
Ligand_Neg_Charges=[]


def add_charge_pair(pairs,pos,neg,contact_prob):
  if pos not in pairs: 
    pairs[pos] = {} 
  if neg not in pairs[pos]:
    pairs[pos][neg] = {}
  pairs[pos][neg] = contact_prob

#Select Protein Charge Groups
#Add Apropriate HIS name if there is a charged HIE OR HIP in the structure 
Protein_Pos_Charges=top.select("(resname ARG and name CZ) or (resname LYS and name NZ) or (resname HIE and name NE2) or (resname HID and name ND1)")
#Protein_Neg_Charges=[]
Protein_Neg_Charges=top.select("resname ASP and name CG or resname GLU and name CD or name OXT")
neg_res=[]
pos_res=[]
                               
for i in Protein_Neg_Charges:
 neg_res.append(top.atom(i).residue.resSeq)

for i in Protein_Pos_Charges:
 pos_res.append(top.atom(i).residue.resSeq)                               
                               
print("Negatively Charged Residues:", neg_res)
print("Posiitively Charged Residues", pos_res)

                               
#contact_pairs=np.zeros((len(Ligand_Pos_Charges),len(Protein_Neg_Charges))
charge_pairs_ligpos=[]                      
for i in Ligand_Pos_Charges:
 for j in Protein_Neg_Charges:              
  charge_pairs_ligpos.append([i,j])
  pos=top.atom(i)
  neg=top.atom(j) 

charge_pairs_ligneg=[]                      
for i in Ligand_Neg_Charges:
 for j in Protein_Pos_Charges:              
  charge_pairs_ligneg.append([i,j])
  pos=top.atom(i)
  neg=top.atom(j) 

if len(charge_pairs_ligpos) != 0:
 contact  = md.compute_distances(trj, charge_pairs_ligpos)
 contacts = np.asarray(contact).astype(float)
 cutoff=0.5
 neg_res_contact_frames=np.where(contacts < 0.5, 1, 0)
 contact_prob_ligpos = np.sum(neg_res_contact_frames,axis=0)/trj.n_frames

if len(charge_pairs_ligneg) != 0:
 contact  = md.compute_distances(trj, charge_pairs_ligneg)
 contacts = np.asarray(contact).astype(float)
 cutoff=0.5
 pos_res_contact_frames=np.where(contacts < 0.5, 1, 0)
 contact_prob_ligneg = np.sum(pos_res_contact_frames,axis=0)/trj.n_frames

charge_pair_names={}
for i in range(0,len(charge_pairs_ligpos)):
  pos=top.atom(charge_pairs_ligpos[i][0])
  neg=top.atom(charge_pairs_ligpos[i][1])      
  add_charge_pair(charge_pair_names,pos,neg,contact_prob_ligpos[i])

for i in range(0,len(charge_pairs_ligneg)):
  pos=top.atom(charge_pairs_ligneg[i][1])
  neg=top.atom(charge_pairs_ligneg[i][0])      
  add_charge_pair(charge_pair_names,pos,neg,contact_prob_ligneg[i])

print(charge_pair_names)

residues=20
residue_offset=121
residue_number = range(residue_offset, residue_offset+residues)
neg_res_index=np.array(neg_res)-residue_offset
Charge_Contacts=np.zeros((n_frames,residues))

for i in range(0,len(neg_res)):
 Charge_Contacts[:,neg_res[i]-residue_offset]=neg_res_contact_frames[:,i]

charge_contact_fraction=np.average(Charge_Contacts,axis=0)
np.savetxt('charge_contacts.traj.dat', Charge_Contacts,'%i')
np.savetxt('charge_contacts.dat',np.column_stack((residue_number, charge_contact_fraction)),fmt='%.4f')
charge_by_res=np.column_stack((residue_number, charge_contact_fraction))
print(charge_by_res)


np.savetxt('charge_contacts.BF.0.7325bf.dat',np.column_stack((residue_number, charge_contact_fraction/desres_bf)),fmt='%.4f')




plt.plot(residue_number,charge_contact_fraction)
plt.xlabel('Residue', size=18)
plt.ylabel('Charge Contact Probability', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.tight_layout()
plt.savefig('ChargeContactFraction.png')
plt.clf()
#plt.show()

plt.plot(residue_number,charge_contact_fraction/desres_bf)
plt.xlabel('Residue', size=18)
plt.ylabel('Charge Contact Probability', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.tight_layout()
plt.savefig('ChargeContactFraction.0.7325bf.png')
plt.clf()


# In[49]:


#Calculate Contacts BY DESRES DEFINITION 
#desres-with -m analysislib/2.0.47c7/bin a_measure trajectory.ark dist --distance "protein" "not protein and not water and not ion and not hydrogen and not resname ACE" method:closest --out-rpk MinimumDistance/$ligand.rpk
#desres-with -m desres-python/2.7.14-02c7/bin  ./parse_csv.py MinimumDistance/$ligand.csv MinimumDistance/$ligand.dat
#awk '{if($1<6) print(n); n++;}' MinimumDistance/$ligand.dat  > MinimumDistance/$ligand.bound.frames.dat

#PER RESIDUE COMMAND
#--distance "not protein and not water and not ion and not hydrogen" "resid 139 and sidechain" method:closest

ligand_hphob=top.select("residue 0 and not element H")
protein_hphob=top.select("protein and sidechain")


ligand_hphob_atoms = []
for atom in ligand_hphob:
    ligand_hphob_atoms.append(top.atom(atom))
    
protein_hphob_atoms = []
for atom in protein_hphob:
    protein_hphob_atoms.append(top.atom(atom))
    
print(ligand_hphob)
print(protein_hphob)
print(ligand_hphob_atoms)
print(protein_hphob_atoms)


def add_contact_pair(pairs,a1,a2,a1_id,a2_id,prot_res,contact_prob):
  if prot_res not in pairs: 
    pairs[prot_res] = {} 
  if a2 not in pairs[prot_res]:
    pairs[prot_res][a2] = {}
  if a1_id not in pairs[prot_res][a2]:
    pairs[prot_res][a2][a1_id] = contact_prob
        
hphob_pairs=[]                      
for i in ligand_hphob :
 for j in protein_hphob :              
  hphob_pairs.append([i,j])


contact  = md.compute_distances(trj, hphob_pairs)
contacts = np.asarray(contact).astype(float)
cutoff=0.6
contact_frames=np.where(contacts < 0.6, 1, 0)
contact_prob_hphob = np.sum(contact_frames,axis=0)/trj.n_frames


#Hphob Contacts at Atom Pair Resolution
hphob_pair_names={}
for i in range(0,len(hphob_pairs)):
   a1_id=hphob_pairs[i][0]
   a2_id=hphob_pairs[i][1]
   a1=top.atom(hphob_pairs[i][0])
   a2=top.atom(hphob_pairs[i][1])
   prot_res=top.atom(hphob_pairs[i][1]).residue.resSeq 
   #print(hphob_pairs[i][0],hphob_pairs[i][1],a1,a2,prot_res,contact_prob_hphob[i])
   add_contact_pair(hphob_pair_names,a1,a2,a1_id,a2_id,prot_res,contact_prob_hphob[i])

#residue_number = range(residue_offset, residue_offset+residues)
hphob_max_contacts={}

#Print Most Populated Contact For Each Aliphatic Carbon in the protein
for i in residue_number:
 if i in hphob_pair_names.keys():   
  #print(hphob_pair_names[i])  
  maxi= 0
  for j in hphob_pair_names[i]: 
    #print(i,j,hphob_pair_names[i][j])
    max_contact_j=max(hphob_pair_names[i][j], key=hphob_pair_names[i][j].get)
    max_contact_fraction=hphob_pair_names[i][j][max_contact_j]
    #print(i,j,max_contact_j,max_contact_fraction)
    if max_contact_fraction > maxi:
     max_key_j=j 
     max_j_subkey=max_contact_j
     maxi=max_contact_fraction
  #print(i,max_key_j,max_j_subkey,hphob_pair_names[i][max_key_j][max_j_subkey])
  hphob_max_contacts[i]=[max_key_j,max_j_subkey,top.atom(max_j_subkey),hphob_pair_names[i][max_key_j][max_j_subkey]]

print("Most Populatd Hydrophobic Contact by Residue:")
for i in hphob_max_contacts:
 print(hphob_max_contacts[i])


# In[51]:


#Cast Contacts contacts as per residue in each frame
Hphob_res_contacts=np.zeros((n_frames,residues))
for frame in range(n_frames):
 if np.sum(contact_frames[frame]) > 0:
  contact_pairs=np.where(contact_frames[frame] == 1) 
  for j in contact_pairs[0]:
    residue=top.atom(hphob_pairs[j][1]).residue.resSeq
    Hphob_res_contacts[frame][residue-residue_offset]=1 

np.savetxt('DESRES_contacts.traj.dat', Hphob_res_contacts,'%i')
Hphob_contact_fraction= np.sum(Hphob_res_contacts,axis=0)/trj.n_frames
hphob_by_res=np.column_stack((residue_number,Hphob_contact_fraction))
np.savetxt('ContactFraction.Ligand_noH.Protein_Sidechain.dat', hphob_by_res,fmt='%.4f')

plt.plot(residue_number, Hphob_contact_fraction)
plt.xlabel('Residue', size=18)
plt.ylabel('DESRES Contact fraction', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.tight_layout()
plt.savefig('ContactFraction.Ligand_noH.Protein_Sidechain.png')
#plt.show()
plt.clf()
print(hphob_by_res)


# In[48]:





# In[52]:


#Calculate Hydrophobic contacts 
#DESRES Hphob selection: 'protein and element C and not name CA'
#VMD aliphatic selection: resname ALA GLY ILE LEU VAL 
#VMD hydrophobic selection: resname ALA LEU VAL ILE PRO PHE MET TRP

ligand_hphob=top.select("residue 0 and element C")
protein_hphob=top.select("protein and element C and not name CA")


ligand_hphob_atoms = []
for atom in ligand_hphob:
    ligand_hphob_atoms.append(top.atom(atom))
    
protein_hphob_atoms = []
for atom in protein_hphob:
    protein_hphob_atoms.append(top.atom(atom))
    
print(ligand_hphob)
print(protein_hphob)
print(ligand_hphob_atoms)
print(protein_hphob_atoms)


def add_contact_pair(pairs,a1,a2,a1_id,a2_id,prot_res,contact_prob):
  if prot_res not in pairs: 
    pairs[prot_res] = {} 
  if a2 not in pairs[prot_res]:
    pairs[prot_res][a2] = {}
  if a1_id not in pairs[prot_res][a2]:
    pairs[prot_res][a2][a1_id] = contact_prob
        
hphob_pairs=[]                      
for i in ligand_hphob :
 for j in protein_hphob :              
  hphob_pairs.append([i,j])


contact  = md.compute_distances(trj, hphob_pairs)
contacts = np.asarray(contact).astype(float)
cutoff=0.4
contact_frames=np.where(contacts < cutoff, 1, 0)
contact_prob_hphob = np.sum(contact_frames,axis=0)/trj.n_frames


#Hphob Contacts at Atom Pair Resolution
hphob_pair_names={}
for i in range(0,len(hphob_pairs)):
   a1_id=hphob_pairs[i][0]
   a2_id=hphob_pairs[i][1]
   a1=top.atom(hphob_pairs[i][0])
   a2=top.atom(hphob_pairs[i][1])
   prot_res=top.atom(hphob_pairs[i][1]).residue.resSeq 
   #print(hphob_pairs[i][0],hphob_pairs[i][1],a1,a2,prot_res,contact_prob_hphob[i])
   add_contact_pair(hphob_pair_names,a1,a2,a1_id,a2_id,prot_res,contact_prob_hphob[i])

#residue_number = range(residue_offset, residue_offset+residues)
hphob_max_contacts={}

#Print Most Populated Contact For Each Aliphatic Carbon in the protein
for i in residue_number:
 if i in hphob_pair_names.keys():   
  #print(hphob_pair_names[i])  
  maxi= 0
  for j in hphob_pair_names[i]: 
    #print(i,j,hphob_pair_names[i][j])
    max_contact_j=max(hphob_pair_names[i][j], key=hphob_pair_names[i][j].get)
    max_contact_fraction=hphob_pair_names[i][j][max_contact_j]
    #print(i,j,max_contact_j,max_contact_fraction)
    if max_contact_fraction > maxi:
     max_key_j=j 
     max_j_subkey=max_contact_j
     maxi=max_contact_fraction
  #print(i,max_key_j,max_j_subkey,hphob_pair_names[i][max_key_j][max_j_subkey])
  hphob_max_contacts[i]=[max_key_j,max_j_subkey,top.atom(max_j_subkey),hphob_pair_names[i][max_key_j][max_j_subkey]]

print("Most Populatd Hydrophobic Contact by Residue:")
for i in hphob_max_contacts:
 print(hphob_max_contacts[i])


# In[54]:


#Cast hydrophobic contacts as per residue in each frame
Hphob_res_contacts=np.zeros((n_frames,residues))
for frame in range(n_frames):
 if np.sum(contact_frames[frame]) > 0:
  contact_pairs=np.where(contact_frames[frame] == 1) 
  for j in contact_pairs[0]:
    residue=top.atom(hphob_pairs[j][1]).residue.resSeq
    Hphob_res_contacts[frame][residue-residue_offset]=1 

np.savetxt('hphob_contacts.traj.dat', Hphob_res_contacts,'%i')
Hphob_contact_fraction= np.sum(Hphob_res_contacts,axis=0)/trj.n_frames
hphob_by_res=np.column_stack((residue_number,Hphob_contact_fraction))
np.savetxt('hphob_contacts.dat', hphob_by_res)

plt.plot(residue_number, Hphob_contact_fraction)
plt.xlabel('Residue', size=18)
plt.ylabel('Hydrophobic Contact fraction', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.tight_layout()
plt.savefig("HydrophobicContactFraction.LigandCarbon.ProteinCarbon_andnotCA.png")
#plt.show()
plt.clf()
print(hphob_by_res)


hphob_by_res=np.column_stack((residue_number,Hphob_contact_fraction/desres_bf))
np.savetxt('hphob_contacts.0.7325bf.dat', hphob_by_res, fmt='%.4f')

plt.plot(residue_number, Hphob_contact_fraction/desres_bf)
plt.xlabel('Residue', size=18)
plt.ylabel('Hydrophobic Contact fraction', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.tight_layout()
plt.savefig("HydrophobicContactFraction.LigandCarbon.ProteinCarbon_andnotCA.0.7325bf.png")
#plt.show()
plt.clf()
print(hphob_by_res)

TYR_ring=top.select("resname TYR and name CG CD1 CD2 CE1 CE2 CZ")
TRP_ring=top.select("resname TRP and name CG CD1 NE1 CE2 CD2 CZ2 CE3 CZ3 CH2")
HIS_ring=top.select("resname HIS and name CG ND1 CE1 NE2 CD2")
PHE_ring=top.select("resname PHE and name CG CD1 CD2 CE1 CE2 CZ")
# In[55]:


#OLD Definitions - Don't Specify Directions
def find_plane_normal(points):
    N=points.shape[0]
    A = np.concatenate( (points[:,0:2], np.ones( (N,1))), axis=1);
    B = points[:,2]
    out = lstsq( A, B,rcond=-1);
    na_c,nb_c,d_c= out[0]
    if d_c != 0.0:
        cu = 1./d_c;
        bu = -nb_c*cu;
        au = -na_c*cu;
    else:
        cu = 1.0
        bu = -nb_c;
        au = -na_c;
    normal = array( [au,bu,cu] )
    normal /= math.sqrt( dot( normal, normal))
    return normal

def find_plane_normal2(positions):
    #Alternate approach used to check sign - could the sign check cause descrepency with desres?
    v1=positions[0]-positions[1]
    v1 /= np.sqrt(np.sum(v1**2))
    v2=positions[2]-positions[1]
    v2 /= np.sqrt(np.sum(v2**2))
    normal=np.cross(v1,v2)
    return normal

def get_ring_center_normal(positions):
    center = np.mean(positions, axis=0)
    normal = find_plane_normal(positions)
    normal2 = find_plane_normal2(positions)
    #check direction of normal using dot product convention
    comp=np.dot(normal,normal2)
    if comp < 0:
      normal=-normal
    #  print("flip")
    return center, normal

def angle(v1,v2):
  return np.arccos(np.dot(v1,v2)/(np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2))))

def get_ring_center_normal_trj(position_array):
  length=len(position_array)
  centers=np.zeros((length,3))
  normals=np.zeros((length,3))
  centers_normals=np.zeros((length,2,3))
  for i in range(0,len(position_array)): 
    center,normal=get_ring_center_normal(position_array[i])
    #centers[i]=center
    #normals[i]=normal
    centers_normals[i][0]=center
    centers_normals[i][1]=normal
    #centers_normals[frame][0]=center, centers_normals[:,0]=all centers
    #centers_normals[frame][1]=normal  centers_normals[:,1]=all normals
  return centers_normals
    
ligand_rings=[[304,305,306,307,308,309,310,311,312,313]]
n_rings=len(ligand_rings)
print("Ligand Aromatics Rings:",n_rings)

ligand_ring_params=[]
for i in range (0,n_rings):
 ring=np.array(ligand_rings[i])
 print(ring)
 positions=trj.xyz[:, ring,:]
 ligand_centers_normals=get_ring_center_normal_trj(positions) 
 ligand_ring_params.append(ligand_centers_normals)

    
#Find Protein Aromatic Rings
#Add Apropriate HIS name if there is a charged HIE OR HIP in the structure 
prot_rings=[]
aro_residues=[]
prot_ring_name=[]
prot_ring_index=[]

aro_select=top.select("resname TYR PHE HIS TRP and name CA")
for i in aro_select:
  atom = top.atom(i)  
  resname=atom.residue.name
  print(atom.index,atom.name,atom.residue.name,atom.residue,atom.residue.index)
  if resname=="TYR":
    ring=top.select("resid %s and name CG CD1 CD2 CE1 CE2 CZ" % atom.residue.index)
    #ring=top.select("resid %s and name CG CD1 CE1 CZ CE2 CD2" % atom.residue.index)
    print(atom.residue,ring)
  if resname=="TRP": 
    ring=top.select("resid %s and name CG CD1 NE1 CE2 CD2 CZ2 CE3 CZ3 CH2" % atom.residue.index)
    print(atom.residue,ring)
  if resname=="HIS": 
    ring=top.select("resid %s and name CG ND1 CE1 NE2 CD2" % atom.residue.index)
    print(atom.residue,ring)
  if resname=="PHE": 
    ring=top.select("resid %s and name CG CD1 CD2 CE1 CE2 CZ" % atom.residue.index)
    print(atom.residue,ring)
  prot_rings.append(ring)
  prot_ring_name.append(atom.residue)
  prot_ring_index.append(atom.residue.index+residue_offset)
  
    
print("Protein Aromatics Rings:",len(prot_rings),prot_ring_name)
#print("Protein Aromatics Rings:",len(prot_rings))

prot_ring_params=[]
for i in range (0,len(prot_rings)):
 ring=np.array(prot_rings[i])
 positions=trj.xyz[:, ring,:]
 ring_centers_normals=get_ring_center_normal_trj(positions) 
 prot_ring_params.append(ring_centers_normals)


# In[115]:


def find_plane_normal(points):
    N=points.shape[0]
    A = np.concatenate( (points[:,0:2], np.ones( (N,1))), axis=1);
    B = points[:,2]
    out = lstsq( A, B,rcond=-1);
    na_c,nb_c,d_c= out[0]
    if d_c != 0.0:
        cu = 1./d_c;
        bu = -nb_c*cu;
        au = -na_c*cu;
    else:
        cu = 1.0
        bu = -nb_c;
        au = -na_c;
    normal = array( [au,bu,cu] )
    normal /= math.sqrt( dot( normal, normal))
    return normal

def find_plane_normal2(positions):
    #Alternate approach used to check sign - could the sign check cause descrepency with desres?
    #Use Ligand IDs 312, 308 and 309 to check direction
    #[304 305 306 307 308 309 310 311 312 313]
    v1=positions[0]-positions[1]
    v1 /= np.sqrt(np.sum(v1**2))
    v2=positions[2]-positions[1]
    v2 /= np.sqrt(np.sum(v2**2))
    normal=np.cross(v1,v2)
    return normal

def find_plane_normal2_assign_atomid(positions,id1,id2,id3):
    #Alternate approach used to check sign - could the sign check cause descrepency with desres?
    v1=positions[id1]-positions[id2]
    v1 /= np.sqrt(np.sum(v1**2))
    v2=positions[id3]-positions[id1]
    v2 /= np.sqrt(np.sum(v2**2))
    normal=np.cross(v1,v2)
    return normal

def get_ring_center_normal_assign_atomid(positions,id1,id2,id3):
    center = np.mean(positions, axis=0)
    normal = find_plane_normal(positions)
    normal2 = find_plane_normal2_assign_atomid(positions,id1,id2,id3)
    #check direction of normal using dot product convention
    comp=np.dot(normal,normal2)
    if comp < 0:
      normal=-normal
    #  print("flip")
    return center, normal

def get_ring_center_normal_(positions):
    center = np.mean(positions, axis=0)
    normal = find_plane_normal(positions)
    normal2 = find_plane_normal2(positions)
    #check direction of normal using dot product convention
    comp=np.dot(normal,normal2)
    if comp < 0:
      normal=-normal
    #  print("flip")
    return center, normal


def angle(v1,v2):
  return np.arccos(np.dot(v1,v2)/(np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2))))

def get_ring_center_normal_trj(position_array):
  length=len(position_array)
  centers=np.zeros((length,3))
  normals=np.zeros((length,3))
  centers_normals=np.zeros((length,2,3))
  for i in range(0,len(position_array)): 
    center,normal=get_ring_center_normal(position_array[i])
    #centers[i]=center
    #normals[i]=normal
    centers_normals[i][0]=center
    centers_normals[i][1]=normal
    #centers_normals[frame][0]=center, centers_normals[:,0]=all centers
    #centers_normals[frame][1]=normal  centers_normals[:,1]=all normals
  return centers_normals

def get_ring_center_normal_trj_assign_atomid(position_array,id1,id2,id3):
  length=len(position_array)
  centers=np.zeros((length,3))
  normals=np.zeros((length,3))
  centers_normals=np.zeros((length,2,3))
  #print(position_array[id1],position_array[id2],position_array[id3])  
  for i in range(0,len(position_array)): 
    center,normal=get_ring_center_normal_assign_atomid(position_array[i],id1,id2,id3)
    centers_normals[i][0]=center
    centers_normals[i][1]=normal
  return centers_normals


ligand_rings=[[304,305,306,307,308,309,310,311,312,313]]
n_rings=len(ligand_rings)
print("Ligand Aromatics Rings:",n_rings)

ligand_ring_params=[]
for i in range (0,n_rings):
 ring=np.array(ligand_rings[i])
 print(ring)
 positions=trj.xyz[:, ring,:]
 print(ligand_rings[i][0],ligand_rings[i][1],ligand_rings[i][2])
 ligand_centers_normals=get_ring_center_normal_trj_assign_atomid(positions,0,1,2) 
 ligand_ring_params.append(ligand_centers_normals)

    
#Find Protein Aromatic Rings
#Add Apropriate HIS name if there is a charged HIE OR HIP in the structure 
prot_rings=[]
aro_residues=[]
prot_ring_name=[]
prot_ring_index=[]

aro_select=top.select("resname TYR PHE HIS TRP and name CA")
for i in aro_select:
  atom = top.atom(i)  
  resname=atom.residue.name
  print(atom.index,atom.name,atom.residue.name,atom.residue,atom.residue.index)
  if resname=="TYR":
    ring=top.select("resid %s and name CG CD1 CD2 CE1 CE2 CZ" % atom.residue.index)
    #ring=top.select("resid %s and name CG CD1 CE1 CZ CE2 CD2" % atom.residue.index)
    print(atom.residue,ring)
  if resname=="TRP": 
    ring=top.select("resid %s and name CG CD1 NE1 CE2 CD2 CZ2 CE3 CZ3 CH2" % atom.residue.index)
    print(atom.residue,ring)
  if resname=="HIS": 
    ring=top.select("resid %s and name CG ND1 CE1 NE2 CD2" % atom.residue.index)
    print(atom.residue,ring)
  if resname=="PHE": 
    ring=top.select("resid %s and name CG CD1 CD2 CE1 CE2 CZ" % atom.residue.index)
    print(atom.residue,ring)
  prot_rings.append(ring)
  prot_ring_name.append(atom.residue)
  prot_ring_index.append(atom.residue.index+residue_offset)
  
    
print("Protein Aromatics Rings:",len(prot_rings),prot_ring_name)
#print("Protein Aromatics Rings:",len(prot_rings))

prot_ring_params=[]
for i in range (0,len(prot_rings)):
 ring=np.array(prot_rings[i])
 print(ring[0],ring[1],ring[2])
 positions=trj.xyz[:, ring,:]
 #ring_centers_normals=get_ring_center_normal_trj(positions) 
 ring_centers_normals=get_ring_center_normal_trj_assign_atomid(positions,0,1,2) 
 prot_ring_params.append(ring_centers_normals)


# In[56]:


frames=n_frames
sidechains=len(prot_rings)
ligrings=len(ligand_rings)
print(frames,sidechains)
Ringstacked={}
Quadrants={}
Stackparams={}

def normvector_connect(point1,point2):
   vec=point1-point2
   vec = vec/np.sqrt(np.dot(vec,vec))
   return vec

def angle(v1,v2):
  return np.arccos(np.dot(v1,v2)/(np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2))))

print("q1: alpha<=45 and beta>=135")
print("q2: alpha>=135 and beta>=135")
print("q3: alpha<=45 and beta<=45")
print("q4: alpha>=135 and beta<=135")

for l in range(0,ligrings):
 name="Lig_ring.%s" % l
 print(name) 
 Stackparams[name]={}
 alphas=np.zeros(shape=(frames,sidechains))
 betas=np.zeros(shape=(frames,sidechains))
 dists=np.zeros(shape=(frames,sidechains))
 stacked=np.zeros(shape=(frames,sidechains))
 quadrant=np.zeros(shape=(frames,sidechains))
 for i in range(0,frames):
  ligcenter=ligand_ring_params[l][i][0]
  lignormal=ligand_ring_params[l][i][1]
  for j in range(0,sidechains):
   protcenter=prot_ring_params[j][i][0]
   protnormal=prot_ring_params[j][i][1]
   dists[i,j] = np.linalg.norm(ligcenter-protcenter)
   connect=normvector_connect(protcenter,ligcenter)
   alphas[i,j]=np.rad2deg(angle(connect,protnormal))
   betas[i,j]=np.rad2deg(angle(connect,lignormal))
 for j in range(0,sidechains):
  name2=prot_ring_index[j]
  print(name2)
  Ringstack=np.column_stack((dists[:,j],alphas[:,j],betas[:,j]))
  a=np.where(alphas[:,j] >= 135)
  b=np.where(alphas[:,j] <= 45)
  c=np.where(betas[:,j] >= 135)
  d=np.where(betas[:,j] <= 45)
  e=np.where(dists[:,j] <= 0.5)
  q1=np.intersect1d(np.intersect1d(b,c),e)
  q2=np.intersect1d(np.intersect1d(a,c),e)
  q3=np.intersect1d(np.intersect1d(b,d),e)
  q4=np.intersect1d(np.intersect1d(a,d),e)
  stacked[:,j][q1]=1
  stacked[:,j][q2]=1
  stacked[:,j][q3]=1
  stacked[:,j][q4]=1
  quadrant[:,j][q1]=1
  quadrant[:,j][q2]=2
  quadrant[:,j][q3]=3
  quadrant[:,j][q4]=4
  total_stacked=len(q1)+len(q2)+len(q3)+len(q4)
  print("q1:",len(q1),"q2:",len(q2),"q3:",len(q3),"q4:",len(q4))
  print("q1:",len(q1)/total_stacked,"q2:",len(q2)/total_stacked,"q3:",len(q3)/total_stacked,"q4:",len(q4)/total_stacked)
  print(max(len(q1),len(q2),len(q3),len(q4))/min(len(q1),len(q2),len(q3),len(q4)))
  #print(Ringstack)
  Stackparams[name][name2]=Ringstack
  print(np.average(Ringstack,axis=0))
 Ringstacked[name]=stacked
 Quadrants[name]=quadrant  
 Ringstacked[name]=stacked
 print(np.average(Ringstacked[name],axis=0))


# In[94]:


fig, ax = plt.subplots(2,2,figsize=((8,8)))

contact=np.where(Stackparams['Lig_ring.0'][125][:,0]<= 0.5)
H125=ax[0,0].hist2d(Stackparams['Lig_ring.0'][125][:,1][contact],Stackparams['Lig_ring.0'][125][:,2][contact], bins=36,range=[[0,180],[0,180]], norm=colors.LogNorm(),cmap='jet')
np.savetxt('Lig_ring.0.Y125.stackparams.all.dat',Stackparams['Lig_ring.0'][125],fmt='%.4f')
np.savetxt('Lig_ring.0.Y125.stackparams.dist_lt_0.5.dat',Stackparams['Lig_ring.0'][125][contact],fmt='%.4f')

contact=np.where(Stackparams['Lig_ring.0'][133][:,0]<= 0.5)
H133=ax[0,1].hist2d(Stackparams['Lig_ring.0'][133][:,1][contact],Stackparams['Lig_ring.0'][133][:,2][contact], bins=36,range=[[0,180],[0,180]], norm=colors.LogNorm(),cmap='jet')
np.savetxt('Lig_ring.0.Y133.stackparams.all.dat',Stackparams['Lig_ring.0'][133],fmt='%.4f')
np.savetxt('Lig_ring.0.Y133.stackparams.dist_lt_0.5.dat',Stackparams['Lig_ring.0'][133][contact],fmt='%.4f')

contact=np.where(Stackparams['Lig_ring.0'][136][:,0]<= 0.5)
H136=ax[1,0].hist2d(Stackparams['Lig_ring.0'][136][:,1][contact],Stackparams['Lig_ring.0'][136][:,2][contact], bins=36,range=[[0,180],[0,180]], norm=colors.LogNorm(),cmap='jet')
np.savetxt('Lig_ring.0.Y136.stackparams.all.dat',Stackparams['Lig_ring.0'][136],fmt='%.4f')
np.savetxt('Lig_ring.0.Y136.stackparams.dist_lt_0.5.dat',Stackparams['Lig_ring.0'][136][contact],fmt='%.4f')


plt.savefig('StackingHistograms.png')
#plt.show()
plt.clf()


# In[62]:





# In[73]:


#Cast aromatic contacts as per residue in each frame
residues=20
residue_offset=121
residue_number = range(residue_offset, residue_offset+residues)
aro_res_index=np.array(prot_ring_index)-residue_offset
aromatic_stacking_contacts=np.zeros((n_frames,residues))
print(aro_res_index)

for i in range(0,len(aro_res_index)):
 aromatic_stacking_contacts[:,aro_res_index[i]]=Ringstacked['Lig_ring.0'][:,i]

np.savetxt('aromatic_stacking.traj.dat', aromatic_stacking_contacts,'%i')
aromatic_stacking_fraction= np.sum(aromatic_stacking_contacts,axis=0)/trj.n_frames



aromatic_by_res=np.column_stack((residue_number,aromatic_stacking_fraction))
np.savetxt('aromatic_stacking.fraction.dat',aromatic_by_res,fmt='%.4f')



print("Aromatic Stacking Fraction, All Frames:",aromatic_by_res)
plt.plot(residue_number, aromatic_stacking_fraction)
plt.xlabel('Residue', size=18)
plt.ylabel('Aromatic Stacking Fraction', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.tight_layout()
plt.savefig('AromaticStackingFraction.png')
plt.clf()
#plt.show()


aromatic_by_res=np.column_stack((residue_number,aromatic_stacking_fraction/desres_bf))
np.savetxt('aromatic_stacking.fraction.0.7325bf.dat', hphob_by_res, fmt='%.4f')

print("Aromatic, DESRES BoundFraction:",aromatic_by_res)
plt.plot(residue_number, aromatic_stacking_fraction/desres_bf)
plt.xlabel('Residue', size=18)
plt.ylabel('Aromatic Stacking Fraction', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.tight_layout()
plt.savefig('AromaticStackingFraction.0.7325bf.png')
plt.clf()


# In[74]:


def _get_bond_triplets(topology,lig_donors,exclude_water=True, sidechain_only=False):
    def can_participate(atom):
        # Filter waters
        if exclude_water and atom.residue.is_water:
            return False
        # Filter non-sidechain atoms
        if sidechain_only and not atom.is_sidechain:
            return False
        # Otherwise, accept it
        return True

    def get_donors(e0, e1):
        # Find all matching bonds
        #print("get_donors e0 e1:",e0,e1)
        elems = set((e0, e1))
        #print("elems:",elems)
        atoms = [(one, two) for one, two in topology.bonds
            if set((one.element.symbol, two.element.symbol)) == elems]
        #print("atoms",atoms)
        # Filter non-participating atoms
        atoms = [atom for atom in atoms
            if can_participate(atom[0]) and can_participate(atom[1])]
        # Get indices for the remaining atoms
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            # make sure to get the pair in the right order, so that the index
            # for e0 comes before e1
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)

        return indices

    # Check that there are bonds in topology
    nbonds = 0
    for _bond in topology.bonds:
        nbonds += 1
        break # Only need to find one hit for this check (not robust)
    if nbonds == 0:
        raise ValueError('No bonds found in topology. Try using '
                         'traj._topology.create_standard_bonds() to create bonds '
                         'using our PDB standard bond definitions.')
        
    nh_donors = get_donors('N', 'H')
    #print("nh_donors",nh_donors)
    #for i in nh_donors:
    # print(top.atom(i[0]),top.atom(i[1]))    
    oh_donors = get_donors('O', 'H')
    #print("oh_donors",oh_donors)
    #for i in oh_donors:
    # print(top.atom(i[0]),top.atom(i[1]))    
    sh_donors = get_donors('S', 'H')
    #print("sh_donors",sh_donors)
    #for i in sh_donors:
    # print(top.atom(i[0]),top.atom(i[1]))  
    #for i in lig_donors:
    # print(top.atom(i[0]),top.atom(i[1]))  
    #ADD IN ADDITIONAL SPECIFIED LIGAND DONORS
    xh_donors = np.array(nh_donors + oh_donors +sh_donors+lig_donors)
    #xh_donors = np.array(nh_donors + oh_donors)

    if len(xh_donors) == 0:
        # if there are no hydrogens or protein in the trajectory, we get
        # no possible pairs and return nothing
        return np.zeros((0, 3), dtype=int)

    acceptor_elements = frozenset(('O', 'N', 'S'))
    acceptors = [a.index for a in topology.atoms
        if a.element.symbol in acceptor_elements and can_participate(a)]
    #print("acceptors")
    #print(acceptors)
    #for i in acceptors:
    #    print(top.atom(i))
    # Make acceptors a 2-D numpy array
    acceptors = np.array(acceptors)[:, np.newaxis]

    # Generate the cartesian product of the donors and acceptors
    xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
    acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
    bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))

    # Filter out self-bonds
    self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
    return bond_triplets[np.logical_not(self_bond_mask), :]

    
def _compute_bounded_geometry(traj, triplets, distance_cutoff, distance_indices,
                              angle_indices, freq=0.0, periodic=True):
    """
    Returns a tuple include (1) the mask for triplets that fulfill the distance
    criteria frequently enough, (2) the actual distances calculated, and (3) the
    angles between the triplets specified by angle_indices.
    """
    # First we calculate the requested distances
    distances = md.compute_distances(traj, triplets[:, distance_indices], periodic=periodic)

    # Now we discover which triplets meet the distance cutoff often enough
    prevalence = np.mean(distances < distance_cutoff, axis=0)
    mask = prevalence > freq

    # Update data structures to ignore anything that isn't possible anymore
    triplets = triplets.compress(mask, axis=0)
    distances = distances.compress(mask, axis=1)

    # Calculate angles using the law of cosines
    abc_pairs = zip(angle_indices, angle_indices[1:] + angle_indices[:1])
    abc_distances = []

    # Calculate distances (if necessary)
    for abc_pair in abc_pairs:
        if set(abc_pair) == set(distance_indices):
            abc_distances.append(distances)
        else:
            abc_distances.append(md.compute_distances(traj, triplets[:, abc_pair],
                periodic=periodic))

    # Law of cosines calculation
    a, b, c = abc_distances
    cosines = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    np.clip(cosines, -1, 1, out=cosines) # avoid NaN error
    angles = np.arccos(cosines)
    #print(distances)
    #print(np.rad2deg(angles))
    return mask, distances, angles

def baker_hubbard2(traj, freq=0.1, exclude_water=True, periodic=True, sidechain_only=False,
                  distance_cutoff=0.35, angle_cutoff=150):

    angle_cutoff = np.radians(angle_cutoff)

    if traj.topology is None:
        raise ValueError('baker_hubbard requires that traj contain topology '
                         'information')

    # Get the possible donor-hydrogen...acceptor triplets
    
    #ADD IN LIGAND HBOND DONORS
    add_donors=[[296,318],[296,331]]
    #for pair in add_donors:
    # print(pair)
    # topology.add_bond(topology.atom(pair[0]),topology.atom(pair[1]))

    bond_triplets = _get_bond_triplets(traj.topology,
        exclude_water=exclude_water,lig_donors=add_donors, sidechain_only=sidechain_only)

    mask, distances, angles = _compute_bounded_geometry(traj, bond_triplets,
        distance_cutoff, [1, 2], [0, 1, 2], freq=freq, periodic=periodic)


    # Find triplets that meet the criteria
    presence = np.logical_and(distances < distance_cutoff, angles > angle_cutoff)
    mask[mask] = np.mean(presence, axis=0) > freq
    #bool= np.mean(presence, axis=0) > freq 
    #print(bond_triplets.compress(mask, axis=0))
    #for i in bond_triplets.compress(mask, axis=0):
    # print(top.atom(i[0]),top.atom(i[1]),top.atom(i[2]))
    #print(np.degrees(angles[0][bool]))
    #print(distances[0][bool])
    return bond_triplets.compress(mask, axis=0)


# In[75]:



HBond_PD=np.zeros((n_frames,residues))
HBond_LD=np.zeros((n_frames,residues))
Hbond_pairs_PD={}
Hbond_pairs_LD={}

def add_hbond_pair(donor,acceptor,hbond_pairs,donor_res):
  if donor_res not in hbond_pairs: 
    hbond_pairs[donor_res] = {} 
  if donor not in hbond_pairs[donor_res]:
    hbond_pairs[donor_res][donor] = {}
  if acceptor not in hbond_pairs[donor_res][donor]:
    hbond_pairs[donor_res][donor][acceptor] = 0
  hbond_pairs[donor_res][donor][acceptor] += 1

#Donor & Acceptors Definitions from DESRES paper:
#ligdon = mol.select('chain B and (nitrogen or oxygen or sulfur) and (withinbonds 1 of hydrogen)')
#ligacc = mol.select('chain B and (nitrogen or oxygen or sulfur)')
#protdon = mol.select('chain A and (nitrogen or oxygen or sulfur) and (withinbonds 1 of hydrogen)')
#protacc = mol.select('chain A and (nitrogen or oxygen or sulfur)')

for frame in range(n_frames):
  hbonds = baker_hubbard2(trj[frame],angle_cutoff=150,distance_cutoff=0.35)
  for hbond in hbonds:
    if ((hbond[0] in protein) and (hbond[2] in ligand)):
     donor=top.atom(hbond[0])
     donor_id=hbond[0]
     donor_res=top.atom(hbond[0]).residue.resSeq
     acc=top.atom(hbond[2])
     acc=top.atom(hbond[2])
     acc_res=top.atom(hbond[2]).residue.resSeq
     HBond_PD[frame][donor_res-residue_offset]=1
     add_hbond_pair(donor,acc,Hbond_pairs_PD,donor_res)
    if ((hbond[0] in ligand) and (hbond[2] in protein)):
     donor=top.atom(hbond[0])
     donor_id=hbond[0]
     donor_res=top.atom(hbond[0]).residue.resSeq
     acc=top.atom(hbond[2])
     acc_id=hbond[2]
     acc_res=top.atom(hbond[2]).residue.resSeq
     HBond_LD[frame][acc_res-residue_offset]=1    
     add_hbond_pair(donor,acc,Hbond_pairs_LD,acc_res)
   


# In[76]:


residues=20
residue_offset=121
residue_number = range(residue_offset, residue_offset+residues)

HB_Total=HBond_PD+HBond_LD
HB_Total_ave=np.mean(HB_Total, axis = 0)


PD_ave=np.mean(HBond_PD, axis = 0)
LD_ave=np.mean(HBond_LD, axis = 0)
for i in Hbond_pairs_PD:
    print(i,Hbond_pairs_PD[i])
for i in Hbond_pairs_LD:
    print(i,Hbond_pairs_LD[i])

print("HBond_LD",Hbond_pairs_LD,'%i')
np.savetxt('Hbond.PD.traj.dat',HBond_PD,'%i')
np.savetxt('Hbond.LD.traj.dat',HBond_LD,'%i')
np.savetxt('Hbond.all.traj.dat',HB_Total_ave,'%i')
hbond_by_res=np.column_stack((residue_number, HB_Total_ave))
np.savetxt('HbondFraction.all.dat',hbond_by_res,fmt='%.4f')

#residue_number = range(residue_offset, residue_offset+residues)
plt.plot(residue_number, HB_Total_ave)
plt.xlabel('Residue', size=18)
plt.ylabel('Hydrogen Bond Fraction', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues)
plt.tight_layout()
plt.savefig('HbondFraction.allframes.png')
#plt.show()
plt.clf()
#print(hbond_by_res)


# In[78]:


np.savetxt('HbondFraction.all.0.7325bf.dat',hbond_by_res/desres_bf,fmt='%.4f')

plt.plot(residue_number, HB_Total_ave/desres_bf)
plt.xlabel('Residue', size=18)
plt.ylabel('Hydrogen Bond Fraction', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues)
plt.tight_layout()
plt.savefig('HbondFraction.allframes.0.7325bf.png')
#plt.show()
plt.clf()


# In[80]:


plt.plot(residue_number,Hphob_contact_fraction,label='Hydrophobic',c='green')
plt.plot(residue_number,aromatic_stacking_fraction,label='Aromatic',c='black')
plt.plot(residue_number,charge_contact_fraction,label='Charge',c='blue')
plt.plot(residue_number, HB_Total_ave,label='Hydrogen Bond',c='red')

plt.xlabel('Residue', size=18)
plt.ylabel('Interaction Probability', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('LigandInteractions.allframes.png')
plt.clf()
#plt.show()


# In[81]:


#desres_bf
plt.plot(residue_number,Hphob_contact_fraction/desres_bf,label='Hydrophobic',c='green')
plt.plot(residue_number,aromatic_stacking_fraction/desres_bf,label='Aromatic',c='black')
plt.plot(residue_number,charge_contact_fraction/desres_bf,label='Charge',c='blue')
plt.plot(residue_number, HB_Total_ave/desres_bf,label='Hydrogen Bond',c='red')

plt.xlabel('Residue', size=18)
plt.ylabel('Interaction Probability', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('LigandInteractions.0.7325bf.png')
plt.clf()
#plt.show()


# In[129]:


#Conditional Interaction Probabilities 
#Gather Matrices:
print(np.shape(HB_Total),np.shape(HBond_PD),np.shape(HBond_LD),np.shape(aromatic_stacking_contacts))
print(np.shape(Hphob_res_contacts),np.shape(Charge_Contacts))


# In[84]:





# In[86]:


#Define condition to select frames:
#Where D135 has charge contact
D135contacts=np.where(Charge_Contacts[:,14]==1)
len(D135contacts[0])
HB_Cond=np.mean(HB_Total[D135contacts], axis = 0)
Stack_Cond=np.mean(aromatic_stacking_contacts[D135contacts], axis = 0)
Hphob_Cond=np.mean(Hphob_res_contacts[D135contacts],axis=0)
Charge_Cond=np.mean(Charge_Contacts[D135contacts],axis=0)


np.savetxt('Hbond.conditionalprob.D135charge.dat',HB_Cond,fmt='%.4f')
np.savetxt('Stacking.conditionalprob.D135charge.dat',Stack_Cond,fmt='%.4f')
np.savetxt('Hphob.conditionalprob.D135charge.dat',Hphob_Cond,fmt='%.4f')
np.savetxt('Charge..conditionalprob.D135charge.dat',Charge_Cond,fmt='%.4f')


plt.plot(residue_number,Hphob_Cond,label='Hydrophobic',c='green')
plt.plot(residue_number,Stack_Cond,label='Aromatic',c='black')
plt.plot(residue_number,HB_Cond,label='Hydrogen Bond',c='red')
plt.plot(residue_number,Charge_Cond,label='Charge',c='blue')

plt.xlabel('Residue', size=18)
plt.ylabel('Conditional Interaction Probability', size=18)
plt.tick_params(labelsize=18)
plt.xlim(residue_offset, residue_offset+residues-1)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('LigandInteractions.D135ChargeContacts.png')
#plt.show()
plt.clf()


# In[88]:


fig, ax = plt.subplots(2,2,figsize=((8,8)))

contact=D135contacts
contact2=np.where(Stackparams['Lig_ring.0'][133][:,0]<= 0.5)

H125=ax[0,0].hist2d(Stackparams['Lig_ring.0'][125][:,1][contact],Stackparams['Lig_ring.0'][125][:,2][contact], bins=36,range=[[0,180],[0,180]], norm=colors.LogNorm(),cmap='jet')
H133=ax[0,1].hist2d(Stackparams['Lig_ring.0'][133][:,1][contact],Stackparams['Lig_ring.0'][133][:,2][contact], bins=36,range=[[0,180],[0,180]], norm=colors.LogNorm(),cmap='jet')
H136=ax[1,0].hist2d(Stackparams['Lig_ring.0'][136][:,1][contact],Stackparams['Lig_ring.0'][136][:,2][contact], bins=36,range=[[0,180],[0,180]], norm=colors.LogNorm(),cmap='jet')

plt.savefig('StackingHistograms.D135.png')
plt.clf()


# In[93]:
fig, ax = plt.subplots(2,2,figsize=((8,8)))

contact=D135contacts
contact2=np.where(Stackparams['Lig_ring.0'][125][:,0]<= 0.5)
contact3=np.intersect1d(contact,contact2)
H125=ax[0,0].hist2d(Stackparams['Lig_ring.0'][125][:,1][contact3],Stackparams['Lig_ring.0'][125][:,2][contact3], bins=36,range=[[0,180],[0,180]], norm=colors.LogNorm(),cmap='jet')
np.savetxt('Lig_ring.0.Y125.D135contact.stackparams.all.dat',Stackparams['Lig_ring.0'][125][contact3][:,1:3],fmt='%.4f')
np.savetxt('Lig_ring.0.Y125.D135contact.stackparams.dist_lt_0.5.dat',Stackparams['Lig_ring.0'][125][contact3][:,1:3],fmt='%.4f')

contact2=np.where(Stackparams['Lig_ring.0'][133][:,0]<= 0.5)
contact3=np.intersect1d(contact,contact2)
H133=ax[0,1].hist2d(Stackparams['Lig_ring.0'][133][:,1][contact3],Stackparams['Lig_ring.0'][133][:,2][contact3], bins=36,range=[[0,180],[0,180]], norm=colors.LogNorm(),cmap='jet')

np.savetxt('Lig_ring.0.Y133.D135contact.stackparams.all.dat',Stackparams['Lig_ring.0'][133][contact3][:,1:3],fmt='%.4f')
np.savetxt('Lig_ring.0.Y133.D135contact.stackparams.dist_lt_0.5.dat',Stackparams['Lig_ring.0'][133][contact3][:,1:3],fmt='%.4f')

contact2=np.where(Stackparams['Lig_ring.0'][136][:,0]<= 0.5)
contact3=np.intersect1d(contact,contact2)
H136=ax[1,0].hist2d(Stackparams['Lig_ring.0'][136][:,1][contact3],Stackparams['Lig_ring.0'][136][:,2][contact3], bins=36,range=[[0,180],[0,180]], norm=colors.LogNorm(),cmap='jet')


np.savetxt('Lig_ring.0.Y136.D135contact.stackparams.all.dat',Stackparams['Lig_ring.0'][136][contact3][:,1:3],fmt='%.4f')
np.savetxt('Lig_ring.0.Y136.D135contact.stackparams.dist_lt_0.5.dat',Stackparams['Lig_ring.0'][136][contact3][:,1:3],fmt='%.4f')

plt.savefig('StackingHistograms.D135.dist_lt_0.5.png')
plt.show()
plt.clf()

# In[103]:


#contact=np.where(Stackparams['Lig_ring.0'][125][:,0]<= 0.5)
#H125=ax[0,0].hist2d(Stackparams['Lig_ring.0'][125][:,1][contact],Stackparams['Lig_ring.0'][125][:,2][contact], bins=36,range=[[0,180],[0,180]], norm=colors.LogNorm(),cmap='jet')
#np.savetxt('Lig_ring.0.Y125.stackparams.all.dat',Stackparams['Lig_ring.0'][125],fmt='%.4f')
#contact=np.where(Stackparams['Lig_ring.0'][133][:,0]<= 0.5)
#np.savetxt('Lig_ring.0.Y136.stackparams.dist_lt_0.5.dat',Stackparams['Lig_ring.0'][136][contact],fmt='%.4f')


for l in Stackparams:
 print(l) 
 for j in Stackparams[l]:
  print(j)
  dists=Stackparams[l][j][D135contacts][:,0]
  alphas=Stackparams[l][j][D135contacts][:,1]
  betas=Stackparams[l][j][D135contacts][:,2]
  a=np.where(alphas >= 135)
  b=np.where(alphas <= 45)
  c=np.where(betas >= 135)
  d=np.where(betas <= 45)
  e=np.where(dists <= 0.5)
  q1=np.intersect1d(np.intersect1d(b,c),e)
  q2=np.intersect1d(np.intersect1d(a,c),e)
  q3=np.intersect1d(np.intersect1d(b,d),e)
  q4=np.intersect1d(np.intersect1d(a,d),e)
  total_stacked=len(q1)+len(q2)+len(q3)+len(q4)
  print("q1:",len(q1),"q2:",len(q2),"q3:",len(q3),"q4:",len(q4))
  print("q1:",len(q1)/total_stacked,"q2:",len(q2)/total_stacked,"q3:",len(q3)/total_stacked,"q4:",len(q4)/total_stacked)
  print(max(len(q1),len(q2),len(q3),len(q4))/min(len(q1),len(q2),len(q3),len(q4)))

