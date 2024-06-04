import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

def MolFromXYZWithSMILES(xyz_string, smiles):
    # Create a molecule from the SMILES representation
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError("Invalid SMILES representation")
        
    # Prepare the molecule
    mol = Chem.AddHs(mol)
    Chem.rdDistGeom.EmbedMolecule(mol, useRandomCoords=True)
    conformer = mol.GetConformer()
    
    # Parse the XYZ coordinates
    lines = xyz_string.strip().split('\n')[2:]  # Skip the first two lines
    new_pos = np.zeros((len(mol.GetAtoms()), 3), dtype=float)
    
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 4:
            symbol, x, y, z = parts
            new_pos[i] = [float(x), float(y), float(z)]
    
    # Set the x, y, z coordinates for each atoms
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, new_pos[i])

    return mol

def AddBonds(mol, bond_factor = 1.25):
    # Create a new empty molecule
    mol_new = Chem.RWMol()
    
    # Add atoms
    for atom in mol.GetAtoms():
        mol_new.AddAtom(atom)

    # Add conformer
    conf = mol.GetConformer()
    mol_new.AddConformer(conf)

    # Add bonds based on covalent radii
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            dist = np.linalg.norm(np.array(conf.GetAtomPosition(i)) - np.array(conf.GetAtomPosition(j)))
            pt = Chem.GetPeriodicTable()
            r_cov_i = pt.GetRcovalent(mol.GetAtomWithIdx(i).GetSymbol())
            r_cov_j = pt.GetRcovalent(mol.GetAtomWithIdx(j).GetSymbol())
            if dist < (r_cov_i + r_cov_j) * bond_factor:
                mol_new.AddBond(i, j, Chem.BondType.SINGLE)

    # Convert to a regular Mol object
    mol_new = mol_new.GetMol()
    
    return mol_new

def View3DModel(mol):
    # Visualize the molecule with Py3DMol
    view = py3Dmol.view(width=800, height=400)
    view.addModel(Chem.MolToMolBlock(mol), "molecule", {'keepH': True})
    view.setBackgroundColor('white')
    view.setStyle({'stick': {'scale': 0.3}, 'sphere': {'scale': 0.3}})
    view.zoomTo()
    view.show()

def WritePDBTrajectory(mol, coordinates_list, change_bonding=False, bond_factor=1.25):
    # Initialize the trajectory string
    trajectory = ""
        
    # Loop over each set of coordinates and generate the corresponding PDB format
    for idx, coordinates in enumerate(coordinates_list):
        # Adjust the molecule's atom positions to match the current geometry
        conf = mol.GetConformer()
        for i, coord in enumerate(coordinates):
            conf.SetAtomPosition(i, coord)

        # Change bonding
        if change_bonding:
            # Remove bonds
            mol_copy = Chem.MolFromXYZBlock(Chem.MolToXYZBlock(mol))
            # Add bonds
            mol = AddBonds(mol_copy, bond_factor)

        # Add the records for atoms
        trajectory += "MODEL     {:4d}\n".format(idx + 1)
        trajectory += Chem.MolToPDBBlock(mol)
        trajectory += "ENDMDL\n"

    return trajectory

def WriteXYZString(mol, charge, multiplicity):
    # Get atom information
    atoms = mol.GetAtoms()
    xyz_lines = []
    for atom in atoms:
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        xyz_lines.append(f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}")

    # Construct the XYZ string
    xyz_string = f"{charge} {multiplicity}\n" + "\n".join(xyz_lines)
    return xyz_string

def ReadXYZString(xyz_string, bond_factor=1.25, return_charge_and_multiplicity=False):
    lines = xyz_string.split('\n')
    charge, multiplicity = map(int, lines[0].split())

    # Create a new empty molecule
    mol = Chem.RWMol()

    # Parse the atomic coordinates and add atoms
    coords = []
    elements = []
    for line in lines[1:]:
        if line.strip():
            parts = line.split()
            element = parts[0].capitalize()  # Correctly format the element symbol
            atom = Chem.Atom(element)
            mol.AddAtom(atom)
            elements.append(element)
            x, y, z = map(float, parts[1:4])
            coords.append([x, y, z])

    # Add 3D coordinates to the molecule
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, line in enumerate(lines[1:]):
        if line.strip():
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            conf.SetAtomPosition(i, (x, y, z))
    mol.AddConformer(conf)
    
    # Add bonds based on covalent radii
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            pt = Chem.GetPeriodicTable()
            r_cov_i = pt.GetRcovalent(elements[i])
            r_cov_j = pt.GetRcovalent(elements[j])
            if dist < (r_cov_i + r_cov_j) * bond_factor:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
                
    # Convert to a regular Mol object
    mol = mol.GetMol()
    
    if return_charge_and_multiplicity:
        return mol, charge, multiplicity
    else:
        return mol