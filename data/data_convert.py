from __future__ import division
from __future__ import print_function

import tqdm
import argparse
import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# These are forked from MoleculeNet letter
atom_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
             'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn', 'B', 'Si', 'Se']
degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
valence_list = [0, 1, 2, 3, 4, 5, 6]
formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
radical_list = [0, 1, 2]
hybridization_list = [Chem.rdchem.HybridizationType.SP,
                      Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3,
                      Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2]
aromatic_list = [0, 1]
num_h_list = [0, 1, 2, 3, 4]

bond_list = [Chem.rdchem.BondType.SINGLE,
             Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
conjugate_list = [0, 1]
ring_list = [0, 1]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return(list(map(lambda s: float(x == s), allowable_set)))

def atom_features(atom, explicit_H=False):
    out = one_of_k_encoding(atom.GetSymbol(),
                            atom_list)
    out += one_of_k_encoding(atom.GetDegree(),
                            degree_list)
    out += one_of_k_encoding(atom.GetImplicitValence(),
                            valence_list)
    out += one_of_k_encoding(atom.GetFormalCharge(),
                            formal_charge_list)
    out += one_of_k_encoding(atom.GetNumRadicalElectrons(),
                            radical_list)
    out += one_of_k_encoding(atom.GetHybridization(),
                            hybridization_list)
    out += one_of_k_encoding(atom.GetIsAromatic(),
                            aromatic_list)
    if not explicit_H:
        out += one_of_k_encoding(atom.GetTotalNumHs(),
                                num_h_list)
    return(out)


def bond_features(bond, explicit_H=False):
    out = one_of_k_encoding(bond.GetBondType(),
                            bond_list)
    out += one_of_k_encoding(bond.GetIsConjugated(),
                            conjugate_list)
    out += one_of_k_encoding(bond.IsInRing(),
                            ring_list)
    return(out)


def get_max_n_nodes(smiles):
    n_atoms = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        atoms = mol.GetAtoms()
        n_atoms += [len(atoms)]
    return(max(n_atoms))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    help='Dataset name.')
args = parser.parse_args()

dataset_dict = {'freesolv': ['freesolv.csv', 'freesolv.npz'],
                'esol': ['esol.csv', 'esol.npz'],
                'lipophilicity': ['lipophilicity.csv', 'lipophilicity.npz'],
                }

if args.dataset.lower() not in dataset_dict.keys():
    print("Choose supported datasets.")
    exit(1)

ifilename = os.path.join('dataset', dataset_dict[args.dataset.lower()][0])
ofilename = os.path.join('dataset', dataset_dict[args.dataset.lower()][1])

raw_dataset = pd.read_csv(ifilename)
#if args.dataset == 'freesolv':
#    raw_dataset = raw_dataset[raw_dataset['expt'] > -10]

if args.dataset == 'freesolv':
    smiles = np.array(raw_dataset['smiles'])
    outs = np.array(raw_dataset['expt'])

if args.dataset == 'esol':
    smiles = np.array(raw_dataset['smiles'])
    outs = np.array(raw_dataset['measured log solubility in mols per litre'])

if args.dataset == 'lipophilicity':
    smiles = np.array(raw_dataset['smiles'])
    outs = np.array(raw_dataset['exp'])

max_n_nodes = get_max_n_nodes(smiles)
n_feas = len(atom_list + degree_list + valence_list + formal_charge_list +
             radical_list + hybridization_list + aromatic_list + num_h_list)
n_edge_feas = len(bond_list + conjugate_list + ring_list)


Xs = []; As = []; Es_idx = []; Es_fea = []; Ns = []; Ys = []; Gs = []
A2s = []; E2s_idx = []; E2s_fea = []
for i, smile in tqdm.tqdm(enumerate(smiles), total=len(smiles)):
    mol = Chem.MolFromSmiles(smile)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    n_nodes = len(atoms)

    X = np.zeros((n_nodes, n_feas))
    A = np.zeros((n_nodes, n_nodes))
    A2 = np.zeros((n_nodes, n_nodes))
    E_idx = []
    E2_idx = []
    E_fea = []
    E2_fea = []
    N = len(atoms)

    if N <= 2:
        continue

    for j, atom in enumerate(atoms):
        X[j] = atom_features(atom)

    for j in range(N):
        for k in range(N):
            bond = mol.GetBondBetweenAtoms(j, k)
            if bond is not None:
                E_idx.append([j, k])
                E_fea.append(bond_features(bond)) 

    A = GetAdjacencyMatrix(mol)

    
    Y = [outs[i]]

    # global properties
    g = nx.Graph()
    g.add_nodes_from(list(range(N)))
    g.add_edges_from(E_idx)

    if not nx.is_connected(g):
        print("{} is not connected".format(smile))
        continue

    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            shortest_path = nx.shortest_path(g, source=j, target=k)
            if len(shortest_path) == 3:
                A2[j, k] = 1

                edge1 = shortest_path[:2]
                edge2 = shortest_path[1:]
                edge1_fea = E_fea[E_idx.index(edge1)]
                edge2_fea = E_fea[E_idx.index(edge2)]
                edge_fea = list((np.array(edge1_fea) + np.array(edge2_fea))*0.5)

                E2_idx.append([shortest_path[0], shortest_path[2]])
                E2_fea.append(edge_fea)


    Xs.append(X)
    As.append(A)
    A2s.append(A2)
    Es_idx.append(E_idx)
    E2s_idx.append(E2_idx)
    Es_fea.append(E_fea)
    E2s_fea.append(E2_fea)
    Ys.append(Y)
    Ns.append(N)

np.savez_compressed(ofilename, Xs=Xs, 
    As=As, 
    A2s=A2s,
    Es_idx=Es_idx, 
    E2s_idx=E2s_idx,
    Es_fea=Es_fea, 
    E2s_fea=E2s_fea,
    Ys=Ys, Ns=Ns, max_n_nodes=max_n_nodes, allow_pickle=True)
