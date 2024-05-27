from openbabel import openbabel

def convert_sdf_to_mol2(sdf_file, mol2_file):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "mol2")
    
    mol = openbabel.OBMol()
    if obConversion.ReadFile(mol, sdf_file):
        obConversion.WriteFile(mol, mol2_file)
    else:
        print("Failed to read the SDF file.")