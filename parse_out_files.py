# %%
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.rdmolops import GetShortestPath
from rdkit.Chem import Descriptors,rdDetermineBonds

# %%
# Diccionario
MASAS = {
    1:"H", 2:"He", 3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne",
    11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 18:"Ar",
    19:"K", 20:"Ca", 21:"Sc", 22:"Ti", 23:"V", 24:"Cr", 25:"Mn", 26:"Fe",
    27:"Co", 28:"Ni", 29:"Cu", 30:"Zn", 31:"Ga", 32:"Ge", 33:"As", 34:"Se",
    35:"Br", 36:"Kr", 37:"Rb", 38:"Sr", 39:"Y", 40:"Zr", 41:"Nb", 42:"Mo",
    43:"Tc", 44:"Ru", 45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd", 49:"In", 50:"Sn",
    51:"Sb", 52:"Te", 53:"I", 54:"Xe", 55:"Cs", 56:"Ba", 57:"La", 58:"Ce",
    59:"Pr", 60:"Nd", 61:"Pm", 62:"Sm", 63:"Eu", 64:"Gd", 65:"Tb", 66:"Dy",
    67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 71:"Lu", 72:"Hf", 73:"Ta", 74:"W",
    75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg", 81:"Tl", 82:"Pb",
    83:"Bi", 84:"Po", 85:"At", 86:"Rn", 87:"Fr", 88:"Ra", 89:"Ac", 90:"Th",
    91:"Pa", 92:"U", 93:"Np", 94:"Pu"
}

# %% [markdown]
# ### Funciones

# %%
def procesar_log(file):
    with open(file, "r", errors="ignore") as f:
        contenido = f.read()
    return contenido

def get_total_electronic_energy(file):
    energies = []
    pattern = re.compile(r'SCF Done:\s+E\(\w+\)\s+=\s+(-?\d+\.\d+)')
    with open(file, "r", encoding="latin-1") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                energies.append(float(match.group(1)))
    if energies:
        return energies[-1]
    else:
        return None
    
def extract_floats(texto):
    FLOAT_RE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?')
    vals = []
    for tok in FLOAT_RE.findall(texto):
        tok = tok.replace('D', 'E').replace('d', 'E')
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return vals

def get_homo_lumo_ua(file):
    with open(file, encoding='utf-8') as f:
        log = f.read()
    alpha_occ_pat = re.compile(r"Alpha\s+occ\.\s+eigenvalues\s*--\s*(.*)")
    alpha_vir_pat = re.compile(r"Alpha\s+virt\.\s+eigenvalues\s*--\s*(.*)")
    #
    alpha_occ, alpha_vir = [], []
    for line in log.splitlines():
        m = alpha_occ_pat.search(line)
        if m:
            vals = [float(extract_floats(x)[0]) for x in m.group(1).split()]
            alpha_occ.extend(vals)
            continue
        m = alpha_vir_pat.search(line)
        if m:
            vals = [float(extract_floats(x)[0]) for x in m.group(1).split()]
            alpha_vir.extend(vals)
            continue
    return [alpha_occ,alpha_vir]

def get_molecular_orbitals(file):
    occ = []
    virt = []
    occ_pattern = re.compile(r'Alpha\s+occ\.\s+eigenvalues\s+--\s+(.+)')
    virt_pattern = re.compile(r'Alpha\s+virt\.\s+eigenvalues\s+--\s+(.+)')

    in_block = False
    all_occ_blocks = []
    all_virt_blocks = []
    current_occ = []
    current_virt = []

    with open(file, "r", encoding="latin-1") as f:
        for line in f:
            if "The electronic state is 1-A." in line:
                # Empieza un bloque
                in_block = True
                current_occ = []
                current_virt = []
                continue

            if "Condensed to atoms (all electrons):" in line and in_block:
                # Termina el bloque
                all_occ_blocks.append(current_occ)
                all_virt_blocks.append(current_virt)
                in_block = False
                continue

            if in_block:
                m_occ = occ_pattern.search(line)
                m_virt = virt_pattern.search(line)

                if m_occ:
                    values = [float(extract_floats(x)[0]) for x in m_occ.group(1).split()]
                    current_occ.extend(values)
                elif m_virt:
                    values = [float(extract_floats(x)[0]) for x in m_virt.group(1).split()]
                    current_virt.extend(values)

    if all_occ_blocks and all_virt_blocks:
        return all_occ_blocks[-1], all_virt_blocks[-1]
    else:
        return [], []
############################# COORDENADAS ############################
def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False

def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _extract_blocks(lines, header_key="Standard orientation:"):
    blocks = []
    i = 0
    n = len(lines)
    dash = re.compile(r'^\s*-{5,}\s*$')

    while i < n:
        if lines[i].strip().startswith(header_key):
            # buscar primera línea de guiones
            i += 1
            while i < n and not dash.match(lines[i]):
                i += 1
            if i >= n: break
            # saltar encabezado hasta segunda línea de guiones
            i += 1
            while i < n and not dash.match(lines[i]):
                i += 1
            if i >= n: break  # no hay segunda línea de guiones
            # ahora vienen las filas hasta la tercera línea de guiones
            i += 1
            current = []
            while i < n and not dash.match(lines[i]):
                line = lines[i].strip()
                if line:
                    parts = line.split()
                    # Fila válida: al menos 6 columnas y 3 ints seguidos de 3 floats
                    if len(parts) >= 6 and _is_int(parts[0]) and _is_int(parts[1]) and _is_int(parts[2]) \
                       and _is_float(parts[3]) and _is_float(parts[4]) and _is_float(parts[5]):
                        indx = int(parts[0])
                        Z = int(parts[1])
                        x = float(parts[3]); y = float(parts[4]); z = float(parts[5])
                        current.append((indx,Z, x, y, z))
                i += 1
            if current:
                blocks.append(current)
        else:
            i += 1
    return blocks

def get_last_coordinates(log_file):
    with open(log_file, "r", encoding="latin-1") as f:
        lines = f.readlines()
    # 1) Intento con Standard orientation
    blocks = _extract_blocks(lines, header_key="Standard orientation:")
    # 2) Fallback con Input orientation si no hubo bloques
    if not blocks:
        blocks = _extract_blocks(lines, header_key="Input orientation:")
    if not blocks:
        # Nada encontrado
        return [], [], [], []

    last = blocks[-1]
    index = [idx for (idx,_ ,_, _, _) in last]
    atoms = [MASAS.get(Z, str(Z)) for (_,Z, _, _, _) in last]
    xs = [x for (_,_, x, _, _) in last]
    ys = [y for (_,_, _, y, _) in last]
    zs = [z for (_,_, _, _, z) in last]
    return index,atoms, xs, ys, zs
############################# COORDENADAS #############################
def write_xyz(path,file):
    if 'sp' in file.split("\\")[-1].replace(".log",''):

        name = file.split("\\")[-1].replace(".log",'').replace('sp_','')
    else:
        name = file.split("\\")[-1].replace(".log",'')
    #
    idx,atomos,x,y,z = get_last_coordinates(file)
    out = Path(os.path.join(path, f"{name}.xyz"))
    with out.open("w", encoding="utf-8") as f:
        f.write(f"{len(atomos)}\n")
        comentario = f"{name} | último Standard orientation | Å"
        f.write(comentario + "\n")
        for atm, xi, yi, zi in zip(atomos, x, y, z):
            f.write(f"{atm:2s} {xi: .6f} {yi: .6f} {zi: .6f}\n")
    return str(out)

def get_radius(file):
    idx,atomos,x,y,z = get_last_coordinates(file)
    xcg = np.average(x)
    ycg = np.average(y)
    zcg = np.average(z)
    radius_cg = []
    for i in range(len(x)):
        ri = ((x[i]-xcg)**2 + (y[i]-ycg)**2 + (z[i]-zcg)**2)**0.5
        radius_cg.append(ri)
    Rg = max(radius_cg) #Angstrom
    # Radio con centro de masa
    MASAS_2 = {v: k for k, v in MASAS.items()}
    Xcm = []
    Ycm = []
    Zcm = []
    M = []
    for at, xi, yi, zi  in zip(atomos,x,y,z):
        Xcm.append(MASAS_2[at]*xi)
        Ycm.append(MASAS_2[at]*yi)
        Zcm.append(MASAS_2[at]*zi)
        M.append(MASAS_2[at])
    xcm = sum(Xcm)/sum(M)
    ycm = sum(Ycm)/sum(M)
    zcm = sum(Zcm)/sum(M)
    #
    radius_cm = []
    for i in range(len(x)):
        ri = ((x[i]-xcm)**2 + (y[i]-ycm)**2 + (z[i]-zcm)**2)**0.5
        radius_cm.append(ri)
    Rm = max(radius_cm) #Angstrom
    return Rg, Rm

def get_cavity_volume(log_file):
    cavity_pattern = re.compile(r'Cavity volume\s*=\s*([0-9.+-Ee]+)')
    values = []

    with open(log_file, "r", encoding="latin-1") as f:
        for line in f:
            match = cavity_pattern.search(line)
            if match:
                values.append(float(match.group(1)))

    if values:
        return values[-1]
    else:
        return None
############################# CARGA Y ÁTOMOS VECINOS #############################
def extract_nbo_natural_charges(file):
    # Variables de salida: Átomo, Número atómico, carga, Coordenadas (x,y,z)
    results = []
    heteroatoms = {"O", "N","P","S"}
    idx, atom, x, y, z = get_last_coordinates(file)
    with open(file, 'r', errors='ignore') as f:
        lines = f.readlines()
    # 1) ubicar el encabezado de la sección
    i = 0
    n = len(lines)
    while i < n and "Summary of Natural Population Analysis" not in lines[i]:
        i += 1
    if i == n:
        return results  # sección no encontrada
    i += 1
    # 2) leer filas hasta el separador de ===
    row_re = re.compile(r'^\s*([A-Z][a-z]?)\s+(\d+)\s+([+-]?\d+\.\d+)\b')
    while i < n:
        line = lines[i]
        if line.strip().startswith('==='):  # fin de la tabla
            break
        m = row_re.match(line)
        if m:
            atom, atom_no, q = m.groups()
            if atom in heteroatoms:
                results.append((atom,int(atom_no), float(q),x[int(atom_no)-1],y[int(atom_no)-1],z[int(atom_no)-1])) # átomo, label, q, xi,yi,zi
        i += 1
    sorted_results = sorted(results, key=lambda x: x[2])
    return sorted_results

def mol_from_gaussian_log(file, charge=0, use_hueckel=True):
    idx,atomos,x,y,z = get_last_coordinates(file)
    MASAS_2 = {v: k for k, v in MASAS.items()}
    rw = Chem.RWMol()
    for atomo in atomos:
        rw.AddAtom(Chem.Atom(MASAS_2[atomo]))
    mol = rw.GetMol()
    conf = Chem.Conformer(len(atomos))
    for i, xi, yi, zi in zip(idx,x,y,z):
        conf.SetAtomPosition(i-1, (xi, yi, zi))
    mol.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineBonds(mol, charge=charge, useHueckel=bool(use_hueckel), allowChargedFragments=True, useVdw=True)
    return mol

def intermediates_count(file, atom_i_gauss, atom_j_gauss, charge=0):
    mol = mol_from_gaussian_log(file, charge=charge)
    i = atom_i_gauss - 1
    j = atom_j_gauss - 1
    path0 = GetShortestPath(mol, i, j)  # tupla de índices 0..N-1
    if not path0:
        return None, None
    path_gauss = [p+1 for p in path0]   # a 1..N como Gaussian
    inter = max(0, len(path_gauss) - 2)
    return path_gauss, inter

############################# PARÁMETROS DEL CÁLCULO  OPT O SP #############################
def parse_gaussian_params(file):
    # Extrae funcional/método, base, carga, multiplicidad y solvente
    with open(file, "r", encoding="latin-1") as f:
        lines = f.readlines()
    route_blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.lstrip().startswith("#"):  # inicio de route
            route = [line.strip()]
            i += 1
            # la route sigue hasta una línea en blanco
            while i < n and lines[i].strip() != "":
                route.append(lines[i].strip())
                i += 1
            route_blocks.append(" ".join(route))
        else:
            i += 1

    route_str = route_blocks[-1] if route_blocks else ""

    # --- 2) Extraer método/funcional y base
    functional = None
    basis = None
    if route_str:
        # tokens separados por espacios
        tokens = route_str.split()
        # buscar el primer token con "/" que parezca "metodo/base"
        for tok in tokens:
            if "/" in tok and not tok.upper().startswith("IOP") and not tok.upper().startswith("EXPERT"):
                # quitar posibles sufijos como trailing commas o paréntesis abiertos/cerrados pegados
                cleaned = tok.strip(",")
                left, right = cleaned.split("/", 1)
                # sanity check: evitar que right empiece con opciones (ej. "Opt" por error)
                # Aceptamos bases como Gen, 6-31G*, 6-31+G(d,p), def2-TZVP, cc-pVTZ, etc.
                if left:  # método/funcional
                    functional = left
                if right and not any(k in right for k in ["Opt", "Freq", "SP", "TD", "SCRF", "EmpiricalDispersion", "Guess", "Pop", "Integral", "Counterpoise"]):
                    basis = right
                # si la base salió rara (None), pero left/right existen, igual guardamos right
                if basis is None and right:
                    basis = right
                break

    # --- 3) Extraer carga y multiplicidad ---
    charge = None
    multiplicity = None

    # Intento 1: línea clásica " Charge = 0 Multiplicity = 1"
    cm_pat = re.compile(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", re.IGNORECASE)
    for line in lines:
        m = cm_pat.search(line)
        if m:
            charge = int(m.group(1))
            multiplicity = int(m.group(2))
    # Si no se encontró, intento 2: buscar cerca del bloque de entrada después de la última route
    if charge is None or multiplicity is None:
        # localizar índice fin de la última route para escanear luego ~30 líneas
        idx_route_end = 0
        if route_blocks:
            # volver a encontrar el último bloque para su final
            # aproximación: buscar última línea que empieza con '#', luego avanzar hasta línea en blanco
            for j, line in enumerate(lines):
                if line.lstrip().startswith("#"):
                    idx_route_candidate = j
            # avanzar a fin del bloque
            k = idx_route_candidate + 1
            while k < n and lines[k].strip() != "":
                k += 1
            idx_route_end = k
        # escanear unas cuantas líneas buscando "X Y"
        simple_cm = re.compile(r"^\s*(-?\d+)\s+(\d+)\s*$")
        for l in lines[idx_route_end: min(idx_route_end + 50, n)]:
            m2 = simple_cm.match(l)
            if m2:
                charge = int(m2.group(1))
                multiplicity = int(m2.group(2))
                break

    # --- 4) Extraer solvente ---
    solvent = None
    if route_str:
        scrf_content = re.search(r"SCRF=\(([^)]*)\)", route_str, flags=re.IGNORECASE)
        if scrf_content:
            inside = scrf_content.group(1)
            msol = re.search(r"Solvent\s*=\s*([A-Za-z0-9_\-\+/]+)", inside, flags=re.IGNORECASE)
            if msol:
                solvent = msol.group(1)
    if solvent is None:
        sol_pat = re.compile(r"Solvent\s*=\s*([A-Za-z0-9_\-\+/]+)", re.IGNORECASE)
        for line in lines:
            m3 = sol_pat.search(line)
            if m3:
                solvent = m3.group(1)

    return {
        "functional": functional,
        "basis": basis,
        "charge": charge,
        "multiplicity": multiplicity,
        "solvent": solvent
    }
############################# CÓDIGO SMILES #############################
def get_smiles_from_log(file, charge=0):
    mol = mol_from_gaussian_log(file, charge=charge)
    return Chem.MolToSmiles(mol)
############################# ENCUENTRA ÁTOMOS LIGANTES SI EL LIGANDO SALEN TIENE MÁS DE 3 HETEROÁTOMOS #############################
def get_salen_ligand_charges(file,ha):
    n = len(ha)
    # HETEROÁTOMOS CENTRALES
    coords = [x[3:6] for x in ha]
    coords = np.array(coords, dtype=float)
    center = coords.mean(axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    sort_ind = np.argsort(distances)
    sort_distances = [(int(i), float(distances[i])) for i in sort_ind][0:2]
    ligand_ind = [i[0] for i in sort_distances[0:4]]
    center_atm = []
    for i in ligand_ind:
        center_atm.append(ha[i][1])
    center_atm
    # HETEROÁTOMOS LIGANTES
    if len(ha) > 4:
        index = []
        for h in ha:
            index.append(h[1])
        ligantes = []
        for c in center_atm:
            for i in index:
                if i != c:
                    params = parse_gaussian_params(file)
                    path, inter = intermediates_count(file,c,i,params['charge'])
                    if inter == 2 or inter == 3:
                        ligantes.append(c)
                        ligantes.append(i)
        ligantes = list(set(ligantes))
    charges = []
    for l in ligantes:
        charge = [(x[0]+'-'+str(x[1]),x[2]) for x in ha if x[1] == l]
        charge = list(charge[0])
        charges.append(charge)
    charges = sorted(charges, key=lambda x: x[1])
    return charges[0:4]
############################# ENCUENTRA ÁTOMOS LIGANTES SI EL LIGANDO PINCER TIENE MÁS DE 3 HETEROÁTOMOS #############################
def get_pincer_ligand_charges(file,ha):
    index = []
    ligantes = {}
    for h in ha:
        index.append(h[1])
    rutas = {}
    for i in range(len(index)-1):
        ligante = []
        for j in range(len(index)):
            if i != j:
                params = parse_gaussian_params(file)
                path, inter = intermediates_count(file,index[i],index[j],params['charge'])
                if inter == 2 or inter == 3:
                    ligante.append([index[j],inter])
                    rutas[str(index[i])+','+str(index[j])] = path
                    #print('Átomo i: ',index[i],'Átomo j',index[j],"Ruta:", path, "| Átomos intermedios:", inter)
        if ligante:
            ligantes[index[i]] = ligante
    results = []
    for lg in ligantes: # átomo i
        if len(ligantes[lg]) > 1:
            for j in range(len(ligantes[lg])-1):
                for k in range(j+1,len(ligantes[lg])):
                    atomo_j = ligantes[lg][j]   #(atomo j, #intermediarios al átomo i)
                    atomo_k = ligantes[lg][k]   #(atomo j, #intermediarios al átomo i)
                    if atomo_j[1] == atomo_k[1]:    # el átomo j y el k están equidistantes al átomo i
                        n = atomo_j[1]
                        path_jk, inter_jk = intermediates_count(file,atomo_j[0],atomo_k[0],params['charge'])
                        if inter_jk == 2*n+1:
                            results.append([lg,atomo_j[0],atomo_k[0],n]) # atomo i, j, k, # átomos intermedios
                            # print(name,lg,atomo_j[0],atomo_k[0])
    if results:
        if len(results) > 1:
            bonds = []
            for res in results:
                bonds.append(res[3])
            max_bonds = max(bonds)
            final_results = [sub for sub in results if sub[3] == max_bonds][0][:3]
        else:
            final_results = results[0][:3]

    charges = []
    for l in final_results:
        charge = [(x[0]+'-'+str(x[1]),x[2]) for x in ha if x[1] == l]
        charge = list(charge[0])
        charges.append(charge)
    charges = sorted(charges, key=lambda x: x[1])
    return charges
############################# DEVUELVE LA CARGA NATURAL DE LOS ÁTOMOS LIGANTES #############################
#------------------- Si el ligando tiene menos de cuatro ligantes, adiciona la carga de la moécula de agua
def get_ligand_natural_charge(file,water_charge):
    name = file.split("\\")[-1].replace(".log",'')
    if name == 'agua':
        charges = [water_charge]*4
    else:
        ha = extract_nbo_natural_charges(file)
        if 'PINCER' in name and len(ha) > 3:
            charges = get_pincer_ligand_charges(file,ha)
            charges.append(water_charge)
            charges = sorted(charges, key=lambda x: x[1])
        elif len(ha) == 3:
            charges = [[h[0]+'-'+str(h[1]),h[2]] for h in ha]
            charges.append(water_charge)
            charges = sorted(charges, key=lambda x: x[1])
        elif 'SALEN' in name and len(ha) > 4:
            charges = get_salen_ligand_charges(file,ha)
        elif 'PINCER' not in name and 'SALEN' not in name and len(ha) > 4:
            charges = get_salen_ligand_charges(file,ha)
        elif len(ha) == 2:
            charges = [[h[0]+'-'+str(h[1]),h[2]] for h in ha]
            charges = charges + [water_charge]*2
            charges = sorted(charges, key=lambda x: x[1])
        else:
            charges = [[h[0]+'-'+str(h[1]),h[2]] for h in ha]
    return charges


def get_descriptors(file,water_charge,ligands,errors_descriptors):
    name = file.split("\\")[-1].replace(".log",'')
    try:
        params = parse_gaussian_params(file)
        smiles = get_smiles_from_log(file, params['charge'])
        #
        total_electronic_energy = get_total_electronic_energy(file)
        alpha_occ, alpha_vir = get_homo_lumo_ua(file)
        homo = round(max(alpha_occ),5)
        lumo = round(min(alpha_vir),5)
        I = round(-1* homo,5)   # Energía de ionización
        A = round(-1*lumo,5)    # Afinidad electrónica
        mu = round(-0.5*(I+A),5)    # Potencial químico
        x = round(-1*mu,5)  # Electronegatividad
        H = round(I-A,5)    # Dureza
        S = round(1/H,5)    # Suavidad
        w = round(mu**2/(2*H),5)    # Electrophilicty
        wp = round((I+3*A)**2/(16*(I-A)),5) # Electroaccepting Power w+
        wn = round((3*I-A)**2/(16*(I-A)),5) # Electrodonating Power w-
        delta_w = round(wp+wn,5)    # Net electrophilicity
        # Radios
        Rg, Rm = get_radius(file)
        #
        Vc = get_cavity_volume(file)
        # Natural Population Analysis
        charges = get_ligand_natural_charge(file,water_charge)
        # kier_hall_index
        mol = Chem.MolFromSmiles(smiles)
        chi_index = {}
        chi_functions = ["Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n"]
        for chi in chi_functions:
            func = getattr(Descriptors, chi)
            chi_index[chi] = func(mol)
        #
        ligand = {}
        ligand = {
            "SMILES_CODE": smiles,
            "IONIZATION_POTENTIAL": I,
            "ELECTROAFFINITY": A,
            "CHEMICAL_POTENTIAL": mu,
            'HARDNESS': H,
            'SOFTNESS': S,
            'ELECTRONEGATIVITY': x,
            'ELECTROPHILICITY w': w,
            'w+ ELECTRON ACCEPTOR': wp,
            'w- ELECTRON DONATOR': wn,
            'ELECTROPHILICITY NET': delta_w,
            "HOMO": homo,
            "LUMO":lumo,
            "GAP": lumo - homo,
            "TOTAL__ELECTRONIC_ENERGY": total_electronic_energy,
            "RADIO GEOMETRICO": round(Rg,5),
            "RADIO CENTRO MASA": round(Rm,5),
            "VOLUMEN DE CAVIDAD": Vc,
            "ÁTOMOS LIGANTES": " ".join([ch[0] for ch in charges]),
                "NATURAL_CHARGE_L1": [ch[1] for ch in charges][0],
                "NATURAL_CHARGE_L2": [ch[1] for ch in charges][1],
                "NATURAL_CHARGE_L3": [ch[1] for ch in charges][2],
                "NATURAL_CHARGE_L4": [ch[1] for ch in charges][3],
                "CHI0n" : chi_index["Chi0n"],
                "CHI1n" : chi_index["Chi1n"],
                "CHI2n" : chi_index["Chi2n"],
                "CHI3n" : chi_index["Chi3n"],
                "CHI4n" : chi_index["Chi4n"]
        }
        #
        ligands[name] = ligand
    except:
        errors_descriptors.append(name)
    return ligands, errors_descriptors


