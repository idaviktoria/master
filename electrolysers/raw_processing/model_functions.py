"""
model_functions.py
Shared model logic for all scenario notebooks.
Each scenario notebook imports this module, sets parameters, and calls run_scenario().
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from dynamic_stock_model import DynamicStockModel
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_PATH     = Path(__file__).resolve().parent.parent
RAW_FOLDER    = BASE_PATH / 'data' / 'raw'
OUTPUT_FOLDER = BASE_PATH / 'data' / 'processed'
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ── IEA grid CO2 intensity trajectories ───────────────────────────────────────
_iea_df         = pd.read_csv(RAW_FOLDER / 'future' / 'iea_grid_co2.csv')
_IEA_GRID_YEARS = _iea_df['year'].tolist()
_IEA_GRID_DATA  = {
    'current_policies': _iea_df['current_policies'].tolist(),
    'stated_policies':  _iea_df['stated_policies'].tolist(),
    'net_zero':         _iea_df['net_zero'].tolist(),
}

def get_grid_co2_trajectory(trajectory, model_years):
    model_years = np.asarray(model_years)
    interp      = PchipInterpolator(_IEA_GRID_YEARS, _IEA_GRID_DATA[trajectory])
    values      = interp(model_years)

    # Hold 2050 value constant beyond 2050 — no data after that year
    val_2050 = float(interp(2050))
    values   = np.where(model_years > 2050, val_2050, values)

    return np.clip(values, 0, None)

GRID_TRAJECTORY_MAP = {
    'Baseline':        'current_policies',
    'DRI Scale-Up':    'current_policies',
    'Technology':      'stated_policies',
    'Full Transition': 'net_zero',
}

# ── WSA crude steel production data (Mt) ──────────────────────────────────────
_wsa_df        = pd.read_csv(RAW_FOLDER / 'Steel' / 'wsa_production.csv')
WSA_PRODUCTION = dict(zip(_wsa_df['year'].astype(int), _wsa_df['production_mt']))

# ── Helper functions ──────────────────────────────────────────────────────────
def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


def logistic_fixed_L(x, k, x0, L=11.3):
    return L / (1 + np.exp(-k * (x - x0)))


def fit_logistic_and_apply(t, stock_data, steel_stock_shares):
    K_guess  = np.max(stock_data)
    r_guess  = 0.1
    t0_guess = t[np.argmax(stock_data >= K_guess / 2)]
    popt, _  = curve_fit(logistic, np.array(steel_stock_shares['year']),
                          stock_data, p0=(K_guess, r_guess, t0_guess))
    K_fit, r_fit, t0_fit = popt
    return logistic(t, K_fit, r_fit, t0_fit)


def run_dsm(t, s, lt):
    dsm = DynamicStockModel(t=t, s=s, lt=lt)
    s_c, o_c, i = dsm.compute_stock_driven_model(NegativeInflowCorrect=True)
    o_c          = dsm.compute_o_c_from_s_c()
    o            = dsm.compute_outflow_total()
    stock_change = dsm.compute_stock_change()
    return s_c, o_c, o, stock_change, i


def scrap_collection_s_curve(years, start=0.70, end=0.90, midpoint=2035, k=0.15):
    s = 1 / (1 + np.exp(-k * (years - midpoint)))
    s_min, s_max = s[0], s[-1]
    return start + (end - start) * (s - s_min) / (s_max - s_min)


def s_curve(years, start_pct, end_pct, midpoint, steepness=0.3):
    s = 1 / (1 + np.exp(-steepness * (years - midpoint)))
    s_min, s_max = s[0], s[-1]
    return start_pct + (end_pct - start_pct) * (s - s_min) / (s_max - s_min)


# ── Load and clean raw data (done once, shared across all scenarios) ───────────
def load_raw_data():
    steel_folder = RAW_FOLDER / 'Steel'
    pop_folder   = RAW_FOLDER / 'population'

    steel_data      = pd.read_excel(steel_folder / 'steel.xlsx')
    steel_shares    = pd.read_csv(steel_folder / 'steel_shares_1900_2008.csv')
    pop_data        = pd.read_csv(pop_folder / 'population.csv')
    future_pop_data = pd.read_csv(pop_folder / 'future_population.csv', low_memory=False)

    # Clean steel stock
    steel_stock_data = steel_data.copy()
    steel_stock_data.columns = steel_data.iloc[1]
    steel_stock_data = steel_stock_data.drop(index=[0, 1])
    steel_stock_data = steel_stock_data.rename(columns={
        'total stock S': 'steel_stock', 'S - Vehicles': 'vehicles',
        'S - Machinery': 'machinery',   'S - BC': 'BC', 'S - Appliances': 'appliances'
    })

    # Clean historic population
    pop_data = pop_data[['Year', 'Code', 'Population']].copy()
    pop_data = pop_data[pop_data['Code'] == 'OWID_WRL'].drop(columns=['Code'])
    pop_data = pop_data.rename(columns={'Year': 'year', 'Population': 'population'})
    pop_data = pop_data[(pop_data['year'] >= 1700) & (pop_data['year'] <= 2008)]

    # Clean future population
    future_pop_data = future_pop_data[['Time', 'Location', 'TPopulation1Jan']].copy()
    future_pop_data = future_pop_data[future_pop_data['Location'] == 'World']
    future_pop_data = future_pop_data[(future_pop_data['Time'] >= 2009) & (future_pop_data['Time'] <= 2100)]
    future_pop_data = future_pop_data.rename(columns={'Time': 'year', 'TPopulation1Jan': 'population'})
    future_pop_data['population'] = future_pop_data['population'] * 1000  # thousands → persons

    return steel_stock_data, steel_shares, pop_data, future_pop_data


# ── Main model function ────────────────────────────────────────────────────────
def run_scenario(params: dict, scenario_name: str, verbose: bool = True, output_folder: Path = None):
    """
    Run the full model for a given set of parameters and save results.

    Parameters
    ----------
    params : dict with keys:
        L                       : float  — steel stock saturation (t/cap)
        scrap_end               : float  — scrap collection rate by 2050
        dri_share_2050          : float  — H-DRI share of primary steel by 2050
        grid_co2                : float  — electricity grid CO2 intensity (g CO2/kWh)
        eaf_efficiency          : float  — EAF electricity consumption (kWh/t steel)
        h2_conversion           : float  — H2 needed per tonne DRI (kg H2/t)
        aec_share               : float  — AEC market share (constant)
        pem_share               : float  — PEM market share (constant)
        other_share             : float  — Other market share (constant)
        lifetime_AEC            : float  — AEC stack lifetime (hours)
        lifetime_PEM            : float  — PEM stack lifetime (hours)
        lifetime_Other          : float  — Other stack lifetime (hours)
        efficiency_AEC          : float  — AEC electricity consumption (kWh/kg H2)
        efficiency_PEM          : float  — PEM electricity consumption (kWh/kg H2)
        efficiency_Other        : float  — Other electricity consumption (kWh/kg H2)
        nickel_intensity        : float  — Ni intensity AEC (kg/MW)
        platinum_intensity_2022 : float  — Pt intensity PEM in 2022 (g/kW)
        platinum_intensity_2030 : float  — Pt intensity PEM in 2030 (g/kW)
        platinum_intensity_2050 : float  — Pt intensity PEM in 2050 (g/kW)
        iridium_intensity_2022  : float  — Ir intensity PEM in 2022 (g/kW)
        iridium_intensity_2030  : float  — Ir intensity PEM in 2030 (g/kW)
        iridium_intensity_2050  : float  — Ir intensity PEM in 2050 (g/kW)
    scenario_name : str — used for output filename
    """
    if verbose:
        print(f'\n{"="*60}')
        print(f'Running scenario: {scenario_name}')
        print(f'{"="*60}')

    # ── Load raw data ──────────────────────────────────────────────────────────
    steel_stock_data, steel_shares, pop_data, future_pop_data = load_raw_data()

    categories = ['vehicles', 'machinery', 'BC', 'appliances']
    cat_col    = {'vehicles': 'vehicles', 'machinery': 'machinery', 'BC': 'bc', 'appliances': 'appliances'}

    shares_1900 = {'vehicles': 0.179350518036637, 'machinery': 0.137017233159745,
                   'BC': 0.604797209347797, 'appliances': 0.078835039455821}

    shares_2008 = {cat: float(steel_shares[steel_shares['year'] == 2008][cat].values[0])
                   for cat in categories}

    # ── Steel stock per capita — logistic projection ───────────────────────────
    steel_stock_cat = steel_stock_data[['year', 'steel_stock']].copy()
    steel_stock_cat['steel_stock'] = steel_stock_cat['steel_stock'] * 1e3

    stock_pre1900 = steel_stock_cat[steel_stock_cat['year'] < 1900][['year', 'steel_stock']].copy()
    for cat, share in shares_1900.items():
        stock_pre1900[f'stock_{cat}'] = stock_pre1900['steel_stock'] * share

    stock_post1900 = steel_stock_cat[
        (steel_stock_cat['year'] >= 1900) & (steel_stock_cat['year'] <= 2008)
    ][['year', 'steel_stock']].copy()
    stock_post1900 = stock_post1900.merge(steel_shares.reset_index(), on='year', how='left')
    for cat in categories:
        stock_post1900[f'stock_{cat}'] = stock_post1900['steel_stock'] * stock_post1900[cat]

    steel_stock_shares_hist = pd.concat([stock_pre1900, stock_post1900], ignore_index=True)
    steel_stock_shares_hist = steel_stock_shares_hist.sort_values('year').reset_index(drop=True)
    steel_stock_shares_hist = steel_stock_shares_hist[
        ['year', 'steel_stock', 'stock_vehicles', 'stock_machinery', 'stock_BC', 'stock_appliances']
    ]

    steel_stock_pop = steel_stock_shares_hist.merge(pop_data, on='year', how='left')
    steel_stock_pop['stock_per_capita'] = steel_stock_pop['steel_stock'] / steel_stock_pop['population']
    steel_stock_pop = steel_stock_pop.dropna(subset=['stock_per_capita'])

    # ── Steel stock per capita ─────────────────────────────────────────────────
    L         = params['L']
    k_scen    = float(params['k_scen'])
    last_year = int(steel_stock_pop['year'].max())
    spc_2008  = float(steel_stock_pop[steel_stock_pop['year'] == last_year]['stock_per_capita'].values[0])

    # Linear bridge 2008→2025 anchored toward 11 t/cap in 2100 (shared across scenarios)
    spc_2025     = float(np.interp(2025.0, [float(last_year), 2100.0], [spc_2008, 11.0]))
    bridge_yrs   = np.arange(last_year + 1, 2026, dtype=float)
    bridge_spc   = np.linspace(spc_2008, spc_2025, len(bridge_yrs) + 1)[1:]

    # Fit a single smooth logistic to all pre-2025 data
    pre_yr  = np.concatenate([steel_stock_pop['year'].values.astype(float), bridge_yrs])
    pre_spc = np.concatenate([steel_stock_pop['stock_per_capita'].values.astype(float), bridge_spc])

    def _log3(x, k, x0, Lh):
        return Lh / (1 + np.exp(-k * (x - x0)))

    popt, _ = curve_fit(_log3, pre_yr, pre_spc,
                        p0=[0.05, 2000.0, 15.0],
                        bounds=([0.01, 1900.0, 5.0], [0.3, 2050.0, 30.0]))
    k_hist, x0_hist, L_hist = popt
    spc_at_2025 = float(_log3(2025.0, k_hist, x0_hist, L_hist))

    # Cubic Hermite spline from 2025 → 2100: inherits historical slope at 2025 (no kink)
    deriv_at_2025 = k_hist * spc_at_2025 * (1.0 - spc_at_2025 / L_hist)
    chs_future    = CubicHermiteSpline([2025.0, 2100.0],
                                       [spc_at_2025, float(L)],
                                       [deriv_at_2025, 0.0])

    future_years = np.arange(last_year, 2101)
    anchor_idx   = int(np.searchsorted(future_years, 2025))

    future_spc = np.concatenate([
        _log3(future_years[:anchor_idx + 1].astype(float), k_hist, x0_hist, L_hist),
        chs_future(future_years[anchor_idx:].astype(float))[1:],
    ])

    future_years_trim = future_years[1:]
    future_pop_trim   = future_pop_data[future_pop_data['year'] > last_year]['population'].values
    future_spc_trim   = future_spc[1:]
    future_stock      = future_spc_trim * future_pop_trim

    stock_df = pd.DataFrame({
        'year':             np.concatenate([steel_stock_pop['year'].values, future_years_trim]),
        'stock_per_capita': np.concatenate([steel_stock_pop['stock_per_capita'].values, future_spc_trim]),
        'total_stock':      np.concatenate([steel_stock_pop['steel_stock'].values, future_stock]),
        'population':       np.concatenate([steel_stock_pop['population'].values, future_pop_trim]),
        'type':             ['historic'] * len(steel_stock_pop) +
                            ['recent' if y <= 2019 else 'predicted' for y in future_years_trim],
    })
    if verbose:
        print(f'Stock 2008: {stock_df[stock_df["year"]==2008]["total_stock"].values[0]/1e9:.2f} Gt')
        print(f'Stock 2050: {stock_df[stock_df["year"]==2050]["total_stock"].values[0]/1e9:.2f} Gt')

    # ── DSM ───────────────────────────────────────────────────────────────────
    t  = np.arange(1700, 2101)
    CV = 0.3

    lt_vehicles   = {'Type': 'Normal', 'Mean': np.full(len(t), 20), 'StdDev': np.full(len(t), 20*CV)}
    lt_machinery  = {'Type': 'Normal', 'Mean': np.full(len(t), 30), 'StdDev': np.full(len(t), 30*CV)}
    lt_bc         = {'Type': 'Normal', 'Mean': np.full(len(t), 75), 'StdDev': np.full(len(t), 75*CV)}
    lt_appliances = {'Type': 'Normal', 'Mean': np.full(len(t), 15), 'StdDev': np.full(len(t), 15*CV)}

    # Sector stocks: smooth logistic-based total stock (avoids kink from actual vs projected data)
    t_float = t.astype(float)
    spc_smooth = np.where(
        t_float <= 2025,
        _log3(t_float, k_hist, x0_hist, L_hist),
        chs_future(t_float)
    )
    pop_smooth = np.interp(t_float,
                           stock_df['year'].values.astype(float),
                           stock_df['population'].values.astype(float))
    total_stock_all = spc_smooth * pop_smooth

    def _smooth_share(s0, s1, mid=1960, k=0.07):
        sig      = 1 / (1 + np.exp(-k * (t_float - mid)))
        sig_min  = 1 / (1 + np.exp(-k * (1800.0   - mid)))
        sig_max  = 1 / (1 + np.exp(-k * (2100.0   - mid)))
        return s0 + (s1 - s0) * (sig - sig_min) / (sig_max - sig_min)

    s_vehicles   = total_stock_all * _smooth_share(shares_1900['vehicles'],   shares_2008['vehicles'])
    s_machinery  = total_stock_all * _smooth_share(shares_1900['machinery'],  shares_2008['machinery'])
    s_bc         = total_stock_all * _smooth_share(shares_1900['BC'],         shares_2008['BC'])
    s_appliances = total_stock_all * _smooth_share(shares_1900['appliances'], shares_2008['appliances'])

    _, _, o_vehicles,   sc_vehicles,   i_vehicles   = run_dsm(t, s_vehicles,   lt_vehicles)
    _, _, o_machinery,  sc_machinery,  i_machinery  = run_dsm(t, s_machinery,  lt_machinery)
    _, _, o_bc,         sc_bc,         i_bc         = run_dsm(t, s_bc,         lt_bc)
    _, _, o_appliances, sc_appliances, i_appliances = run_dsm(t, s_appliances, lt_appliances)

    results_vehicles   = pd.DataFrame({'Year': t, 'stock_vehicles':   s_vehicles,   'inflow_vehicles':   i_vehicles,   'outflow_vehicles':   o_vehicles,   'stock_change_vehicles':   sc_vehicles})
    results_machinery  = pd.DataFrame({'Year': t, 'stock_machinery':  s_machinery,  'inflow_machinery':  i_machinery,  'outflow_machinery':  o_machinery,  'stock_change_machinery':  sc_machinery})
    results_bc         = pd.DataFrame({'Year': t, 'stock_bc':         s_bc,         'inflow_bc':         i_bc,         'outflow_bc':         o_bc,         'stock_change_bc':         sc_bc})
    results_appliances = pd.DataFrame({'Year': t, 'stock_appliances': s_appliances, 'inflow_appliances': i_appliances, 'outflow_appliances': o_appliances, 'stock_change_appliances': sc_appliances})

    results = (results_vehicles.merge(results_machinery, on='Year')
                                .merge(results_bc,        on='Year')
                                .merge(results_appliances, on='Year'))

    results['total_inflow']       = results['inflow_vehicles']  + results['inflow_machinery']  + results['inflow_bc']  + results['inflow_appliances']
    results['total_outflow']      = results['outflow_vehicles'] + results['outflow_machinery'] + results['outflow_bc'] + results['outflow_appliances']
    results['total_stock_change'] = results['stock_change_vehicles'] + results['stock_change_machinery'] + results['stock_change_bc'] + results['stock_change_appliances']

    cat_col = {'vehicles': 'vehicles', 'machinery': 'machinery', 'BC': 'bc', 'appliances': 'appliances'}
    for cat in categories:
        results[f'steel_needed_{cat}'] = results[f'inflow_{cat_col[cat]}'] / (0.92 * 0.96)
    results['total_steel_needed'] = sum(results[f'steel_needed_{cat}'] for cat in categories)
    results = results[results['Year'] != 1800].copy()

    results_total = (results_vehicles.merge(results_machinery, on='Year')
                                      .merge(results_bc,        on='Year')
                                      .merge(results_appliances, on='Year'))
    results_total['total_inflow']  = results_total['inflow_vehicles']  + results_total['inflow_machinery']  + results_total['inflow_bc']  + results_total['inflow_appliances']
    results_total['total_outflow'] = results_total['outflow_vehicles'] + results_total['outflow_machinery'] + results_total['outflow_bc'] + results_total['outflow_appliances']

    future = results_total[results_total['Year'] >= 2022].copy()

    years_future = np.array(future['Year'])
    future['scrap_collection_rate'] = scrap_collection_s_curve(
        years_future, start=0.85, end=params['scrap_end'])
    future['scrap_supply']         = future['total_outflow'] * future['scrap_collection_rate']
    future['primary_steel_needed'] = np.maximum(future['total_inflow'] - future['scrap_supply'], 0)

    # ── DRI penetration ────────────────────────────────────────────────────────
    years = np.arange(2022, 2101)

    if 'bfbof_share_2050' in params:
        # Direct total-steel share parameterisation: H-DRI + BF-BOF + EAF = 1
        total_steel = np.asarray(future['total_inflow'].values, dtype=float)
        hdri_pen  = np.asarray(s_curve(years, 0.02, params['dri_share_2050'],   2038, 0.30), dtype=float)
        bfbof_pen = np.asarray(s_curve(years, 0.68, params['bfbof_share_2050'], 2038, 0.30), dtype=float)
        eaf_pen   = np.maximum(1.0 - hdri_pen - bfbof_pen, 0.0)
        dri_prod  = total_steel * hdri_pen
        bf_prod   = total_steel * bfbof_pen
        eaf_prod  = total_steel * eaf_pen
        primary   = dri_prod + bf_prod
        dri_pen   = hdri_pen
    else:
        dri_pen  = np.asarray(s_curve(years, 0.02, params['dri_share_2050'], 2038, 0.30), dtype=float)
        primary  = np.asarray(future['primary_steel_needed'].values, dtype=float)
        dri_prod = primary * dri_pen
        bf_prod  = primary * (1 - dri_pen)
        eaf_prod = np.asarray(future['scrap_supply'].values, dtype=float)

    production = {
        'DRI Production': dri_prod,
        'BF Production':  bf_prod,
        'EAF Scrap':      eaf_prod,
        'Total':          dri_prod + bf_prod + eaf_prod
    }

    # ── Emissions ──────────────────────────────────────────────────────────────
    MW_Fe = 55.845; MW_C = 12.011; MW_O = 15.999; MW_CO2 = MW_C + 2*MW_O
    emissions_factor_BF = params.get('bf_emission_factor', 2.3)  # t CO2/t steel
    electrode_rate      = 0.002
    eaf_elec_mwh_per_t  = params['eaf_efficiency'] * 1e-3  # kWh/t → MWh/t

    # Grid CO2: string → IEA trajectory, scalar → fixed, array → use directly
    grid_co2_input = params['grid_co2']
    if isinstance(grid_co2_input, str):
        grid_co2_arr = get_grid_co2_trajectory(grid_co2_input, years)
    elif np.isscalar(grid_co2_input):
        grid_co2_arr = np.full(len(years), float(grid_co2_input))
    else:
        grid_co2_arr = np.asarray(grid_co2_input)

    grid_co2_t_per_mwh   = grid_co2_arr / 1000   # g/kWh → t/MWh
    emissions_factor_EAF = eaf_elec_mwh_per_t * grid_co2_t_per_mwh \
                         + electrode_rate * (MW_CO2 / MW_C)  # array

    emissions = {
        'BF Emissions':    bf_prod  / 1e6 * emissions_factor_BF,
        'H-DRI Emissions': dri_prod / 1e6 * emissions_factor_EAF,
        'EAF Emissions':   eaf_prod / 1e6 * emissions_factor_EAF,
        'Total Emissions': bf_prod  / 1e6 * emissions_factor_BF
                         + dri_prod / 1e6 * emissions_factor_EAF
                         + eaf_prod / 1e6 * emissions_factor_EAF,
    }


    # ── Hydrogen demand ────────────────────────────────────────────────────────
    h2_conv    = params['h2_conversion'] / 1000  # kg/t → t H2/t steel
    h2_total   = dri_prod * h2_conv / 1e6        # Mt H2
    aec_share  = params['aec_share']
    pem_share  = params['pem_share']
    other_share= params['other_share']

    h2 = {
        'H2 Demand (Mt)': h2_total,
        'AEC H2':         h2_total * aec_share,
        'PEM H2':         h2_total * pem_share,
        'Other H2':       h2_total * other_share,
    }

    # ── Electrolyzer capacity ──────────────────────────────────────────────────
    available_hours     = (8760 - 11*24) * (1 - 0.03)
    avail_AEC = avail_PEM = avail_other = available_hours

    lifetime_AEC   = params['lifetime_AEC']   / avail_AEC
    lifetime_PEM   = params['lifetime_PEM']   / avail_PEM
    lifetime_Other = params['lifetime_Other'] / avail_other

    eff_AEC   = params['efficiency_AEC']
    eff_PEM   = params['efficiency_PEM']
    eff_Other = params['efficiency_Other']

    CV_e = 0.3
    lt_AEC   = {'Type': 'Normal', 'Mean': np.full(len(years), round(lifetime_AEC,   1)), 'StdDev': np.full(len(years), round(lifetime_AEC,   1)*CV_e)}
    lt_PEM   = {'Type': 'Normal', 'Mean': np.full(len(years), round(lifetime_PEM,   1)), 'StdDev': np.full(len(years), round(lifetime_PEM,   1)*CV_e)}
    lt_Other = {'Type': 'Normal', 'Mean': np.full(len(years), round(lifetime_Other, 1)), 'StdDev': np.full(len(years), round(lifetime_Other, 1)*CV_e)}

    cap_aec   = h2['AEC H2']   * 1e9 * eff_AEC   / avail_AEC   / 1e6  # GW
    cap_pem   = h2['PEM H2']   * 1e9 * eff_PEM   / avail_PEM   / 1e6
    cap_other = h2['Other H2'] * 1e9 * eff_Other / avail_other / 1e6

    _, _, o_aec,   sc_aec,   i_aec   = run_dsm(years, cap_aec,   lt_AEC)
    _, _, o_pem,   sc_pem,   i_pem   = run_dsm(years, cap_pem,   lt_PEM)
    _, _, o_other, sc_other, i_other = run_dsm(years, cap_other, lt_Other)

    cap_results = {'AEC': cap_aec, 'PEM': cap_pem, 'Other': cap_other}
    dsm_elec    = {'i_AEC': i_aec, 'o_AEC': o_aec, 'i_PEM': i_pem, 'o_PEM': o_pem,
                   'i_Other': i_other, 'o_Other': o_other}

    # ── Critical materials ─────────────────────────────────────────────────────
    ni_int = params['nickel_intensity']
    
    pt_2022 = params['platinum_intensity_2022']
    pt_2030 = params['platinum_intensity_2030']
    pt_2050 = params['platinum_intensity_2050']

    ir_2022 = params['iridium_intensity_2022']
    ir_2030 = params['iridium_intensity_2030']
    ir_2050 = params['iridium_intensity_2050']

    pt_interp = PchipInterpolator(
        [2022, 2030, 2050, 2100],
        [pt_2022, pt_2030, pt_2050, pt_2050]
    )
    pt_int_arr = np.clip(pt_interp(years), 0, None)

    interp_func = PchipInterpolator(
        [2022, 2030, 2050, 2100],
        [ir_2022, ir_2030, ir_2050, ir_2050]
    )
    ir_int_arr = np.clip(interp_func(years), 0, None)

    materials = {
    'nickel_inflow':   i_aec * 1e3 * ni_int      / 1e6, # kt/yr
    'nickel_stock':    cap_aec * 1e3 * ni_int     / 1e6,
    'nickel_outflow':  o_aec * 1e3 * ni_int       / 1e6,
    'platinum_inflow': i_pem * 1e3 * pt_int_arr   / 1e6, # kt/yr, time-varying
    'platinum_stock':  cap_pem * 1e3 * pt_int_arr / 1e6,
    'iridium_inflow':  i_pem * 1e3 * ir_int_arr   / 1e6, # kt/yr, time-varying
    'iridium_stock':   cap_pem * 1e3 * ir_int_arr / 1e6,
    }

    # ── Sanity checks ──────────────────────────────────────────────────────────
    idx_2050 = np.where(years == 2050)[0][0]
    if verbose:
        print(f'  Primary steel 2050:  {primary[idx_2050]/1e6:.0f} Mt')
        print(f'  DRI production 2050: {dri_prod[idx_2050]/1e6:.0f} Mt  ({dri_pen[idx_2050]*100:.0f}% of primary)')
        print(f'  H2 demand 2050:      {h2_total[idx_2050]:.2f} Mt')
        print(f'  CO2 total 2050:      {emissions["Total Emissions"][idx_2050]:.0f} Mt')
        print(f'  Ni inflow 2050:      {materials["nickel_inflow"][idx_2050]:.2f} kt')
        print(f'  Ir inflow 2050:      {materials["iridium_inflow"][idx_2050]*1000:.2f} t')

    # ── Save ───────────────────────────────────────────────────────────────────
    out = {
        'scenario_name':    scenario_name,
        'params':           params,
        'years':            years,
        'plot_years':       future['Year'].values,
        'stock_df':         stock_df,
        'results':          results,
        'future':           future,
        'steel_stock_pop':  steel_stock_pop,
        'production':       production,
        'emissions':        emissions,
        'h2':               h2,
        'cap_results':      cap_results,
        'dsm_elec':         dsm_elec,
        'materials':        materials,
        'emissions_factor_BF':  emissions_factor_BF,
        'emissions_factor_EAF': emissions_factor_EAF,
        'wsa_production':    WSA_PRODUCTION,
    }

    save_folder = output_folder if output_folder is not None else OUTPUT_FOLDER
    save_folder.mkdir(parents=True, exist_ok=True)
    out_path = save_folder / f'{scenario_name}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    if verbose:
        print(f'  Saved -> {out_path}')
    return out

def load_remind_data(file_paths: dict, year_cols=None):
    if year_cols is None:
        year_cols = ['2020','2025','2030','2035','2040','2045','2050',
                     '2060','2070','2080','2090','2100']

    EJ_to_MtH2 = 1000 / 142   # 1 EJ ≈ 7.04 Mt H2 (LHV)
    EJ_to_TWh  = 277.78        # 1 EJ = 277.78 TWh
    remind_data = {}

    for label, fpath in file_paths.items():
        df  = pd.read_csv(fpath, sep=';')
        dfw = df[df['Region'] == 'World'].copy()

        def get_var(variable, scale):          # ← now accepts scale
            row = dfw[dfw['Variable'] == variable][year_cols]
            if row.empty:
                return np.zeros(len(year_cols))
            return row.values[0].astype(float) * scale

        remind_data[label] = {
            'years':                  [int(y) for y in year_cols],
            'total_h2':               get_var('SE|Hydrogen',                           EJ_to_MtH2),
            'green_h2':               get_var('SE|Hydrogen|+|Electricity',             EJ_to_MtH2),
            'renewable_electricity':  get_var('SE|Electricity|Non-Biomass Renewables', EJ_to_TWh),
            'electrolysis_electricity':get_var('SE|Input|Electricity|Hydrogen',         EJ_to_TWh),
            'wind':                   get_var('SE|Electricity|+|Wind',                 EJ_to_TWh),
            'solar':                  get_var('SE|Electricity|+|Solar',                EJ_to_TWh),
            'hydro':                  get_var('SE|Electricity|+|Hydro',                EJ_to_TWh),
            'geothermal':             get_var('SE|Electricity|+|Geothermal',           EJ_to_TWh),
        }
        print(f'Loaded: {label}'
              f'  |  Green H2 2050: {remind_data[label]["green_h2"][year_cols.index("2050")]:.1f} Mt'
              f'  |  Renew elec 2050: {remind_data[label]["renewable_electricity"][year_cols.index("2050")]:.0f} TWh'
              f'  |  Electrolysis elec 2050: {remind_data[label]["electrolysis_electricity"][year_cols.index("2050")]:.0f} TWh')

    return remind_data  # ← return is at the end, after the loop

def compare_remind_h2(scenarios: dict, remind_data: dict):
    """
    Interpolate model H2 demand to REMIND year grid for comparison.

    Parameters
    ----------
    scenarios   : dict — loaded scenario results (from 05_plots.ipynb)
    remind_data : dict — loaded REMIND data (from load_remind_data())

    Returns
    -------
    dict — {scenario_name: interpolated H2 array}
    """

    remind_years = list(remind_data.values())[0]['years']

    def interp_model_h2(sc_name):
        sc        = scenarios[sc_name]
        model_yrs = sc['plot_years']
        model_h2  = sc['h2']['H2 Demand (Mt)']
        return np.interp(remind_years, model_yrs, model_h2)

    return {name: interp_model_h2(name) for name in scenarios}