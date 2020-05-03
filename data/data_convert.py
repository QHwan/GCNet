from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from collections import Counter

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix


raw_dataset_filename = './dataset/freesolv.csv'
raw_dataset = pd.read_csv(raw_dataset_filename)

smiles = raw_dataset['smiles']
solv_energies = raw_dataset['expt']

atom_types = []
num_hs = []
num_valences = []
aromaticities = []
num_atoms = []
for i, smile in enumerate(smiles):
    mol = Chem.MolFromSmiles(smile)
    atoms = mol.GetAtoms()

    atom_types += [atom.GetSymbol() for atom in atoms]
    num_hs += [atom.GetTotalNumHs() for atom in atoms]
    num_valences += [atom.GetImplicitValence() for atom in atoms]
    aromaticities += [int(atom.GetIsAromatic()) for atom in atoms]
    num_atoms += [len(atoms)]

atom_type_dict = {}
num_h_dict = {}
num_valence_dict = {}
aromaticity_dict = {}
for i, key in enumerate(Counter(atom_types).keys()):
    atom_type_dict[key] = i
for i, key in enumerate(sorted(Counter(num_hs).keys())):
    num_h_dict[key] = i
for i, key in enumerate(sorted(Counter(num_valences).keys())):
    num_valence_dict[key] = i
for i, key in enumerate(sorted(Counter(aromaticities).keys())):
    aromaticity_dict[key] = i


max_num_atoms = max(num_atoms)
num_fea = len(atom_type_dict) + len(num_h_dict) + len(num_valence_dict) + len(aromaticity_dict)

Xs = []; As = []; Ys = []
for i, smile in enumerate(smiles):
    mol = Chem.MolFromSmiles(smile)
    atoms = mol.GetAtoms()

    atom_types = [atom.GetSymbol() for atom in atoms]
    num_hs = [atom.GetTotalNumHs() for atom in atoms]
    num_valences = [atom.GetImplicitValence() for atom in atoms]
    aromaticities = [int(atom.GetIsAromatic()) for atom in atoms]

    X = np.zeros((max_num_atoms, num_fea))
    for j, (atom_type, num_h, num_valence, aromaticity) in enumerate(zip(atom_types,
                                                                         num_hs,
                                                                         num_valences,
                                                                         aromaticities)):
        x_atom_type = np.zeros(len(atom_type_dict))
        x_num_h = np.zeros(len(num_h_dict))
        x_num_valence = np.zeros(len(num_valence_dict))
        x_aromaticity = np.zeros(len(aromaticity_dict))

        x_atom_type[atom_type_dict[atom_type]] += 1
        x_num_h[num_h_dict[num_h]] += 1
        x_num_valence[num_valence_dict[num_valence]] += 1
        x_aromaticity[aromaticity_dict[aromaticity]] += 1

        x = np.concatenate((x_atom_type,
                            x_num_h,
                            x_num_valence,
                            x_aromaticity))
        X[j] = x

    A = np.zeros((max_num_atoms, max_num_atoms))
    A_mol = GetAdjacencyMatrix(mol)
    A[0:len(A_mol), 0:len(A_mol)] += A_mol + np.eye(len(A_mol))
    Y = solv_energies[i]

    Xs.append(np.array(X))
    As.append(np.array(A))
    Ys.append(np.array(Y))


np.savez('dataset/freesolv.npz', Xs=Xs, As=As, Ys=Ys)