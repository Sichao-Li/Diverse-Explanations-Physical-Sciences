import collections
import re
import pandas as pd
import numpy as np
import tqdm
import os
import os
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict, defaultdict

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch import nn


from sklearn.preprocessing import StandardScaler, Normalizer

import json
dirpath = os.getcwd()
plt.rcParams.update({'font.size': 16})

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

# %%
fig_dir = r'figures/Classics/'
data_type_torch = torch.float32
data_type_np = np.float32

class CompositionError(Exception):
    """Exception class for composition errors"""
    pass


def get_sym_dict(f, factor):
    sym_dict = collections.defaultdict(float)
    # compile regex for speedup
    regex = r"([A-Z][a-z]*)\s*([-*\.\d]*)"
    r = re.compile(regex)
    for m in re.finditer(r, f):
        el = m.group(1)
        amt = 1
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        f = f.replace(m.group(), "", 1)
    if f.strip():
        raise CompositionError(f'{f} is an invalid formula!')
    return sym_dict


def parse_formula(formula):
    '''
    Parameters
    ----------
        formula: str
            A string formula, e.g. Fe2O3, Li3Fe2(PO4)3.
    Return
    ----------
        sym_dict: dict
            A dictionary recording the composition of that formula.
    Notes
    ----------
        In the case of Metallofullerene formula (e.g. Y3N@C80),
        the @ mark will be dropped and passed to parser.
    '''
    # for Metallofullerene like "Y3N@C80"
    formula = formula.replace('@', '')
    formula = formula.replace('[', '(')
    formula = formula.replace(']', ')')
    # compile regex for speedup
    regex = r"\(([^\(\)]+)\)\s*([\.\d]*)"
    r = re.compile(regex)
    m = re.search(r, formula)
    if m:
        factor = 1
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join(["{}{}".format(el, amt)
                                for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return parse_formula(expanded_formula)
    sym_dict = get_sym_dict(formula, 1)
    return sym_dict


def _fractional_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 1e-6:
            elamt[k] = v
            natoms += abs(v)
    comp_frac = {key: elamt[key] / natoms for key in elamt}
    return comp_frac


def _fractional_composition_L(formula):
    comp_frac = _fractional_composition(formula)
    atoms = list(comp_frac.keys())
    counts = list(comp_frac.values())
    return atoms, counts


def _element_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 1e-6:
            elamt[k] = v
            natoms += abs(v)
    return elamt


def _element_composition_L(formula):
    comp_frac = _element_composition(formula)
    atoms = list(comp_frac.keys())
    counts = list(comp_frac.values())
    return atoms, counts


def _assign_features(matrices, elem_info, formulae, sum_feat=False):
    formula_mat, count_mat, frac_mat, elem_mat, target_mat = matrices
    elem_symbols, elem_index, elem_missing = elem_info

    if sum_feat:
        sum_feats = []
    avg_feats = []
    range_feats = []
    # var_feats = []
    dev_feats = []
    max_feats = []
    min_feats = []
    mode_feats = []
    targets = []
    formulas = []
    skipped_formula = []

    for h in tqdm.tqdm(range(len(formulae)), desc='Assigning Features...'):
        elem_list = formula_mat[h]
        target = target_mat[h]
        formula = formulae[h]
        comp_mat = np.zeros(shape=(len(elem_list), elem_mat.shape[-1]))
        skipped = False

        for i, elem in enumerate(elem_list):
            if elem in elem_missing:
                skipped = True
            else:
                row = elem_index[elem_symbols.index(elem)]
                comp_mat[i, :] = elem_mat[row]

        if skipped:
            skipped_formula.append(formula)

        range_feats.append(np.ptp(comp_mat, axis=0))
        # var_feats.append(comp_mat.var(axis=0))
        max_feats.append(comp_mat.max(axis=0))
        min_feats.append(comp_mat.min(axis=0))

        comp_frac_mat = comp_mat.T * frac_mat[h]
        comp_frac_mat = comp_frac_mat.T
        avg_feats.append(comp_frac_mat.sum(axis=0))

        dev = np.abs(comp_mat - comp_frac_mat.sum(axis=0))
        dev = dev.T * frac_mat[h]
        dev = dev.T.sum(axis=0)
        dev_feats.append(dev)

        prominant = np.isclose(frac_mat[h], max(frac_mat[h]))
        mode = comp_mat[prominant].min(axis=0)
        mode_feats.append(mode)

        comp_sum_mat = comp_mat.T * count_mat[h]
        comp_sum_mat = comp_sum_mat.T
        if sum_feat:
            sum_feats.append(comp_sum_mat.sum(axis=0))

        targets.append(target)
        formulas.append(formula)

    if len(skipped_formula) > 0:
        print('\nNOTE: Your data contains formula with exotic elements.',
              'These were skipped.')
    if sum_feat:
        conc_list = [sum_feats, avg_feats, dev_feats,
                     range_feats, max_feats, min_feats, mode_feats]
        feats = np.concatenate(conc_list, axis=1)
    else:
        conc_list = [avg_feats, dev_feats,
                     range_feats, max_feats, min_feats, mode_feats]
        feats = np.concatenate(conc_list, axis=1)

    return feats, targets, formulas, skipped_formula


def generate_features(df, elem_prop='oliynyk',
                      drop_duplicates=False,
                      extend_features=False,
                      sum_feat=False,
                      mini=False):
    '''
    Parameters
    ----------
    df: Pandas.DataFrame()
        X column dataframe of form:
            df.columns.values = array(['formula', 'target',
                                       'extended1', 'extended2', ...],
                                      dtype=object)
    elem_prop: str
        valid element properties:
            'oliynyk',
            'jarvis',
            'magpie',
            'mat2vec',
            'onehot',
            'random_200'
    drop_duplicates: boolean
        Decide to keep or drop duplicate compositions
    extend_features: boolean
        Decide whether to use non ["formula", "target"] columns as additional
        features.
    Return
    ----------
    X: pd.DataFrame()
        Feature Matrix with NaN values filled using the median feature value
        for dataset
    y: pd.Series()
        Target values
    formulae: pd.Series()
        Formula associated with X and y
    '''
    if drop_duplicates:
        if df['formula'].value_counts()[0] > 1:
            df.drop_duplicates('formula', inplace=True)
            print('Duplicate formula(e) removed using default pandas function')

    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    cbfv_path = (dirpath
                + '/data/element_properties/'
                + elem_prop
                + '.csv')

    if not os.path.exists(cbfv_path):
        cbfv_path = (dirpath
                    + '/data/element_properties/bm_element_props/'
                    + elem_prop
                    + '.csv')

    if not os.path.exists(cbfv_path):
        cbfv_path = (dirpath
                    + '/data/element_properties/mb_element_props/'
                    + elem_prop
                    + '.csv')

    elem_props = pd.read_csv(cbfv_path)

    elem_props.index = elem_props['element'].values
    elem_props.drop(['element'], inplace=True, axis=1)

    elem_symbols = elem_props.index.tolist()
    elem_index = np.arange(0, elem_props.shape[0], 1)
    elem_missing = list(set(all_symbols) - set(elem_symbols))

    elem_props_columns = elem_props.columns.values

    column_names = np.concatenate(['avg_' + elem_props_columns,
                                   'dev_' + elem_props_columns,
                                   'range_' + elem_props_columns,
                                   'max_' + elem_props_columns,
                                   'min_' + elem_props_columns,
                                   'mode_' + elem_props_columns])
    if sum_feat:
        column_names = np.concatenate(['sum_' + elem_props_columns,
                                       column_names])

    # make empty list where we will store the property value
    targets = []
    # store formula
    formulae = []
    # add the values to the list using a for loop

    elem_mat = elem_props.values

    formula_mat = []
    count_mat = []
    frac_mat = []
    target_mat = []

    if extend_features:
        features = df.columns.values.tolist()
        features.remove('target')
        extra_features = df[features]

    for index in tqdm.tqdm(df.index.values, desc='Processing Input Data'):
        formula, target = df.loc[index, 'formula'], df.loc[index, 'target']
        if 'x' in formula:
            continue
        l1, l2 = _element_composition_L(formula)
        formula_mat.append(l1)
        count_mat.append(l2)
        _, l3 = _fractional_composition_L(formula)
        frac_mat.append(l3)
        target_mat.append(target)
        formulae.append(formula)

    print('\tfeaturizing compositions...'.title())

    matrices = [formula_mat, count_mat, frac_mat, elem_mat, target_mat]
    elem_info = [elem_symbols, elem_index, elem_missing]
    feats, targets, formulae, skipped = _assign_features(matrices,
                                                         elem_info,
                                                         formulae,
                                                         sum_feat=sum_feat)

    print('\tcreating pandas objects...'.title())

    # split feature vectors and targets as X and y
    X = pd.DataFrame(feats, columns=column_names, index=formulae)
    y = pd.Series(targets, index=formulae, name='target')
    formulae = pd.Series(formulae, index=formulae, name='formula')
    if extend_features:
        extended = pd.DataFrame(extra_features, columns=features)
        extended = extended.set_index('formula', drop=True)
        X = pd.concat([X, extended], axis=1)

    # reset dataframe indices
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    formulae.reset_index(drop=True, inplace=True)

    # drop elements that aren't included in the elmenetal properties list.
    # These will be returned as feature rows completely full of NaN values.
    X.dropna(inplace=True, how='all')
    y = y.iloc[X.index]
    formulae = formulae.iloc[X.index]

    # get the column names
    cols = X.columns.values
    # find the median value of each column
    median_values = X[cols].median()
    # fill the missing values in each column with the column's median value
    X[cols] = X[cols].fillna(median_values)

    # Only return the avg/sum of element properties.

    if mini:
        np.random.seed(42)
        booleans = np.random.rand(X.shape[-1]) <= 64/X.shape[-1]
        X = X.iloc[:, booleans]

    return X, y, formulae, skipped





# %%
def get_obj_size(obj):
    size = None
    # int32 is 32 bits = 4 bytes
    # float32 is 4 bytes
    if obj is None:
        size = 0
    if isinstance(obj, np.ndarray):
        # size = obj.itemsize * obj.size
        size = obj.nbytes
    if isinstance(obj, torch.Tensor):
        size = obj.element_size() * obj.nelement()
    if size is not None:
        # return size in Megabytes (MB)
        size = size / (1024)**2
    if isinstance(obj, list):
        size = np.sum([get_obj_size(subobj) for subobj in obj])
    return size


def clear_cache(obj=None):
    if (isinstance(obj, torch.Tensor)
        or isinstance(obj, np.ndarray)
        or isinstance(obj, list)):
        obj = 0
    elif isinstance(obj, tuple):
        for item in obj:
            clear_cache(item)
    elif obj is None:
        pass
    else:
        obj.clear()
    if obj is not None:
        del obj
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def torch_memory_debug():
    memory_allocated = torch.cuda.memory_allocated() / 1024**2
    memory_reserved = torch.cuda.memory_reserved() / 1024**2
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f'\n{memory_allocated = :0.2f}')
    print(f'{memory_reserved = :0.2f}')
    print(f'{max_memory_allocated = :0.2f}')


# %%
def linear(input, weight, bias=None):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Adapted from PyTorch source code.
    """
    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    ret = output
    return ret


# %%
class CONSTANTS():
    def __init__(self):
        self.crab_red = '#f2636e'
        self.dense_blue = '#2c2cd5'
        self.colors = list(sns.color_palette("Set1", n_colors=7, desat=0.5))

        self.markers = ['o', 'x', 's', '^', 'D', 'P', '1', '2', '3',
                        '4',  'p', '*', 'h', 'H', '+', 'd', '|', '_']

        self.eps = ['oliynyk',
                    'jarvis',
                    'mat2vec',
                    'onehot',
                    'magpie',
                    'random_200']

        self.benchmark_props = [
            'aflow__ael_bulk_modulus_vrh',
            'aflow__ael_debye_temperature',
            'aflow__ael_shear_modulus_vrh',
            'aflow__agl_thermal_conductivity_300K',
            'aflow__agl_thermal_expansion_300K',
            'aflow__Egap',
            'aflow__energy_atom',
            'CritExam__Ed',
            'CritExam__Ef',
            'mp_bulk_modulus',
            'mp_elastic_anisotropy',
            'mp_e_hull',
            'mp_mu_b',
            'mp_shear_modulus',
            'OQMD_Bandgap',
            'OQMD_Energy_per_atom',
            'OQMD_Formation_Enthalpy',
            'OQMD_Volume_per_atom',
             ]

        self.benchmark_names = [
            'AFLOW Bulk modulus',
            'AFLOW Debye temperature',
            'AFLOW Shear modulus',
            'AFLOW Thermal conductivity',
            'AFLOW Thermal expansion',
            'AFLOW Band gap',
            'AFLOW Energy per atom',
            'Bartel Decomposition (Ed)',
            'Bartel Formation (Ef)',
            'MP Bulk modulus',
            'MP Elastic anisotropy',
            'MP Energy above convex hull',
            'MP Magnetic moment',
            'MP Shear modulus',
            'OQMD Band gap',
            'OQMD Energy per atom',
            'OQMD Formation enthalpy',
            'OQMD Volume per atom'
            ]

        self.matbench_props = [
            'castelli',
            'dielectric',
            'elasticity_log10(G_VRH)',
            'elasticity_log10(K_VRH)',
            'expt_gap',
            'expt_is_metal',
            'glass',
            'jdft2d',
            'mp_e_form',
            'mp_gap',
            'mp_is_metal',
            'phonons',
            'steels_yield',
            ]

        self.matbench_names = [
            'Castelli perovskites',
            'Refractive index',
            'Shear modulus (log10)',
            'Bulk modulus (log10)',
            'Experimental band gap',
            'Experimental metallicity',
            'Experimental glass formation',
            'DFT Exfoliation energy',
            'MP Formation energy',
            'MP Band gap',
            'MP Metallicity',
            'Phonon peak',
            'Steels yield'
            ]

        self.benchmark_names_dict = dict(zip(self.benchmark_props,
                                             self.benchmark_names))
        self.matbench_names_dict = dict(zip(self.matbench_props,
                                            self.matbench_names))

        self.mb_units_dict = {
            'castelli': 'eV/unit cell',
            'dielectric': 'unitless',
            'elasticity_log10(G_VRH)': 'log(GPa)',
            'elasticity_log10(K_VRH)': 'log(GPa)',
            'expt_gap': 'eV',
            'expt_is_metal': 'binary',
            'glass': 'binary',
            'jdft2d': 'meV/atom',
            'mp_e_form': 'eV/atom',
            'mp_gap': 'eV',
            'mp_is_metal': 'binary',
            'phonons': '$cm^{−1}$',
            'steels_yield': 'MPa',
            }

        self.bm_units_dict = {
            'aflow__ael_bulk_modulus_vrh': None,
            'aflow__ael_debye_temperature': None,
            'aflow__ael_shear_modulus_vrh': None,
            'aflow__agl_thermal_conductivity_300K': None,
            'aflow__agl_thermal_expansion_300K': None,
            'aflow__Egap': None,
            'aflow__energy_atom': None,
            'CritExam__Ed': None,
            'CritExam__Ef': None,
            'mp_bulk_modulus': None,
            'mp_elastic_anisotropy': None,
            'mp_e_hull': None,
            'mp_mu_b': None,
            'mp_shear_modulus': None,
            'OQMD_Bandgap': None,
            'OQMD_Energy_per_atom': None,
            'OQMD_Formation_Enthalpy': None,
            'OQMD_Volume_per_atom': None,
            }

        self.mp_units_dict = {'energy_atom': 'eV/atom',
                              'ael_shear_modulus_vrh': 'GPa',
                              'ael_bulk_modulus_vrh': 'GPa',
                              'ael_debye_temperature': 'K',
                              'Egap': 'eV',
                              'agl_thermal_conductivity_300K': 'W/m*K',
                              'agl_log10_thermal_expansion_300K': '1/K'}

        self.mp_sym_dict = {'energy_atom': '$E_{atom}$',
                            'ael_shear_modulus_vrh': '$G$',
                            'ael_bulk_modulus_vrh': '$B$',
                            'ael_debye_temperature': '$\\theta_D$',
                            'Egap': '$E_g$',
                            'agl_thermal_conductivity_300K': '$\\kappa$',
                            'agl_log10_thermal_expansion_300K': '$\\alpha$'}

        self.classification_list = ['mp_is_metal',
                                    'expt_is_metal',
                                    'glass']

        self.classic_models_dict = {'Ridge': 'Ridge',
                                    'SGDRegressor': 'SGD',
                                    'ExtraTreesRegressor': 'ExtraTrees',
                                    'RandomForestRegressor': 'RF',
                                    'AdaBoostRegressor': 'AdaBoost',
                                    'GradientBoostingRegressor': 'GradBoost',
                                    'KNeighborsRegressor': 'kNN',
                                    'SVR': 'SVR',
                                    'lSVR': 'lSVR'}

        self.atomic_symbols = ['None', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N',
                               'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P',
                               'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
                               'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                               'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
                               'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
                               'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                               'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                               'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                               'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
                               'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
                               'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
                               'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                               'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                               'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

        self.idx_symbol_dict = {(i): sym for
                                i, sym in enumerate(self.atomic_symbols)}

        self.symbol_idx_dict = {sym: (i) for
                                i, sym in enumerate(self.atomic_symbols)}


# %%
def get_cbfv(path, elem_prop='oliynyk', scale=False):
    """
    Loads the compound csv file and featurizes it, then scales the features
    using StandardScaler.

    Parameters
    ----------
    path : str
        DESCRIPTION.
    elem_prop : str, optional
        DESCRIPTION. The default is 'oliynyk'.

    Returns
    -------
    X_scaled : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    formula : TYPE
        DESCRIPTION.

    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    if 'formula' not in df.columns.values.tolist():
        df['formula'] = df['cif_id'].str.split('_ICSD').str[0]
    # elem_prop = 'mat2vec'
    # elem_prop = 'oliynyk'
    mini = False
    # mini = True
    X, y, formula, skipped = generate_features(df, elem_prop, mini=mini)
    if scale:
        # scale each column of data to have a mean of 0 and a variance of 1
        scaler = StandardScaler()
        # normalize each row in the data
        normalizer = Normalizer()

        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(normalizer.fit_transform(X_scaled),
                                columns=X.columns.values,
                                index=X.index.values)

        return X_scaled, y, formula, skipped
    else:
        return X, y, formula, skipped

# %%
def BCEWithLogitsLoss(output, log_std, target):
    loss = nn.functional.binary_cross_entropy_with_logits(output, target)
    return loss


def RobustL1(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    absolute = torch.abs(output - target)
    loss = np.sqrt(2.0) * absolute * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def RobustL2(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    squared = torch.pow(output - target, 2.0)
    loss = 0.5 * squared * torch.exp(-2.0 * log_std) + log_std
    return torch.mean(loss)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()if p.requires_grad)


# %%
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.float):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def count_gs_param_combinations(d):
    cnt_dict = OrderedDict({})
    # array = []
    if (isinstance(d, (list))
        and not isinstance(d, (bool))):
        return len(d), cnt_dict
    elif (isinstance(d, (int, float, complex))
          and not isinstance(d, (bool))):
        return 1, cnt_dict
    elif (isinstance(d, (bool))
          or isinstance(d, (str))):
        return 1, cnt_dict
    elif d is None:
        return 1, cnt_dict
    elif isinstance(d, (dict, OrderedDict)):
        keys = d.keys()
        for k in keys:
            array = []
            subd = d[k]
            array.append(count_gs_param_combinations(subd)[0])
            cnt = np.prod(array)
            cnt_dict[k] = cnt
        return np.prod(list(cnt_dict.values())), cnt_dict
    return cnt, cnt_dict


# %%
class Scaler():
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def scale(self, data):
        data = torch.as_tensor(data)
        data_scaled = (data - self.mean) / self.std
        return data_scaled

    def unscale(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled)
        data = data_scaled * self.std + self.mean
        return data

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class DummyScaler():
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def scale(self, data):
        return torch.as_tensor(data)

    def unscale(self, data_scaled):
        return torch.as_tensor(data_scaled)

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


# %%
class EDMDataset(Dataset):
    """
    Get X and y from EDM dataset.
    """

    def __init__(self, dataset, n_comp):
        self.data = dataset
        self.n_comp = n_comp

        self.X = np.array(self.data[0])
        self.y = np.array(self.data[1])
        self.formula = np.array(self.data[2])

        self.shape = [(self.X.shape), (self.y.shape), (self.formula.shape)]

    def __str__(self):
        string = f'EDMDataset with X.shape {self.X.shape}'
        return string

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx, :, :]
        y = self.y[idx]
        formula = self.formula[idx]

        X = torch.as_tensor(X, dtype=data_type_torch)
        y = torch.as_tensor(y, dtype=data_type_torch)

        return (X, y, formula)


def get_edm(path, elem_prop='mat2vec', n_elements='infer',
            inference=False, verbose=True, drop_unary=True,
            scale=True):
    """
    Build a element descriptor matrix.

    Parameters
    ----------
    path : str
        DESCRIPTION.
    elem_prop : str, optional
        DESCRIPTION. The default is 'oliynyk'.

    Returns
    -------
    X_scaled : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    formula : TYPE
        DESCRIPTION.

    """
    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    # path can either be string to csv or a dataframe with data already
    if isinstance(path, str):
        df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    else:
        df = path

    if 'formula' not in df.columns.values.tolist():
        df['formula'] = df['cif_id'].str.split('_ICSD').str[0]

    df['count'] = [len(_element_composition(form)) for form in df['formula']]
    if drop_unary:
        df = df[df['count'] != 1]  # drop pure elements
    if not inference:
        df = df.groupby(by='formula').mean().reset_index()  # mean of duplicates

    list_ohm = [OrderedDict(_element_composition(form))
                for form in df['formula']]
    list_ohm = [OrderedDict(sorted(mat.items(), key=lambda x:-x[1]))
                for mat in list_ohm]

    y = df['target'].values.astype(data_type_np)
    formula = df['formula'].values
    if n_elements == 'infer':
        # cap maximum elements at 16, and then infer n_elements
        n_elements = 16

    edm_array = np.zeros(shape=(len(list_ohm),
                                n_elements,
                                len(all_symbols)+1),
                         dtype=data_type_np)
    elem_num = np.zeros(shape=(len(list_ohm), n_elements), dtype=data_type_np)
    elem_frac = np.zeros(shape=(len(list_ohm), n_elements), dtype=data_type_np)
    for i, comp in enumerate(tqdm(list_ohm,
                                  desc="Generating EDM",
                                  unit="formulae",
                                  disable=not verbose)):
        for j, (elem, count) in enumerate(list_ohm[i].items()):
            if j == n_elements:
                # Truncate EDM representation to n_elements
                break
            try:
                edm_array[i, j, all_symbols.index(elem) + 1] = count
                elem_num[i, j] = all_symbols.index(elem) + 1
            except ValueError:
                print(f'skipping composition {comp}')

    if scale:
        # Normalize element fractions within the compound
        for i in range(edm_array.shape[0]):
            frac = (edm_array[i, :, :].sum(axis=-1)
                    / (edm_array[i, :, :].sum(axis=-1)).sum())
            elem_frac[i, :] = frac
    else:
        # Do not normalize element fractions, even for single-element compounds
        for i in range(edm_array.shape[0]):
            frac = edm_array[i, :, :].sum(axis=-1)
            elem_frac[i, :] = frac

    if n_elements == 16:
        n_elements = np.max(np.sum(elem_frac > 0, axis=1, keepdims=True))
        elem_num = elem_num[:, :n_elements]
        elem_frac = elem_frac[:, :n_elements]

    elem_num = elem_num.reshape(elem_num.shape[0], elem_num.shape[1], 1)
    elem_frac = elem_frac.reshape(elem_frac.shape[0], elem_frac.shape[1], 1)
    out = np.concatenate((elem_num, elem_frac), axis=1)

    return out, y, formula


# %%
class EDM_CsvLoader():
    """
    Parameters
    ----------
    csv_data: str
        name of csv file containing cif and properties
    csv_val: str
        name of csv file containing cif and properties
    val_frac: float, optional (default=0.75)
        train/val ratio if val_file not given
    batch_size: float, optional (default=64)
        Step size for the Gaussian filter
    random_state: int, optional (default=123)
        Random seed for sampling the dataset. Only used if validation data is
        not given.
    shuffle: bool (default=True)
        Whether to shuffle the datasets or not
    """

    def __init__(self, csv_data, batch_size=64,
                 num_workers=1, random_state=0, shuffle=True,
                 pin_memory=True, n_elements=6, inference=False,
                 verbose=True,
                 drop_unary=True,
                 scale=True):
        self.csv_data = csv_data
        self.main_data = list(get_edm(self.csv_data, elem_prop='mat2vec',
                                      n_elements=n_elements,
                                      inference=inference,
                                      verbose=verbose,
                                      drop_unary=drop_unary,
                                      scale=scale))
        self.n_train = len(self.main_data[0])
        self.n_elements = self.main_data[0].shape[1]//2

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.random_state = random_state

    def get_data_loaders(self, inference=False):
        '''
        Input the dataset, get train test split
        '''
        shuffle = not inference  # don't shuffle data when inferencing
        pred_dataset = EDMDataset(self.main_data, self.n_elements)
        pred_loader = DataLoader(pred_dataset,
                                 batch_size=self.batch_size,
                                 pin_memory=self.pin_memory,
                                 shuffle=shuffle)
        return pred_loader


# %%
class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning:
        Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes.
        _Large Batch Optimization for Deep Learning: Training BERT in 76
            minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0,
                 adam=False,
                 min_trust=None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if min_trust and not 0.0 <= min_trust < 1.0:
            raise ValueError(f"Minimum trust range from 0 to 1: {min_trust}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        self.min_trust = min_trust
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    err_msg = "Lamb does not support sparse gradients, " + \
                        "consider SparseAdam instad."
                    raise RuntimeError(err_msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_((1 - beta1) * grad)
                # v_t
                # exp_avg_sq.mul_(beta2).addcmul_((1 - beta2) * grad *
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group[
                    "lr"
                ]  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(group["weight_decay"], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                if self.min_trust:
                    trust_ratio = max(trust_ratio, self.min_trust)
                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio * adam_step)

        return loss


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if "slow_buffer" not in param_state:
                param_state["slow_buffer"] = torch.empty_like(fast_p.data)
                param_state["slow_buffer"].copy_(fast_p.data)
            slow = param_state["slow_buffer"]
            slow.add_(group["lookahead_alpha"] * (fast_p.data - slow))
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group["lookahead_step"] += 1
            if group["lookahead_step"] % group["lookahead_k"] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            "state": state_dict["state"],
            "param_groups": state_dict["param_groups"],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if "slow_state" not in state_dict:
            print("Loading state_dict from optimizer without Lookahead applied.")
            state_dict["slow_state"] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict[
                "param_groups"
            ],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = (
            self.base_optimizer.param_groups
        )  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)


# %%
if __name__ == '__main__':
    os.makedirs(fig_dir, exist_ok=True)