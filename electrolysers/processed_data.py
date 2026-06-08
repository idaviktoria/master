import pickle, numpy as np

scenarios = [
    ('low_baseline',  'Low',    'BAU'),
    ('low_dri',       'Low',    'DRI Scale-Up'),
    ('low_tech',      'Low',    'Technology'),
    ('low_full',      'Low',    'Full Transition'),
    ('med_baseline',  'Medium', 'BAU'),
    ('med_dri',       'Medium', 'DRI Scale-Up'),
    ('med_tech',      'Medium', 'Technology'),
    ('med_full',      'Medium', 'Full Transition'),
    ('high_baseline', 'High',   'BAU'),
    ('high_dri',      'High',   'DRI Scale-Up'),
    ('high_tech',     'High',   'Technology'),
    ('high_full',     'High',   'Full Transition'),
]

years = np.arange(2022, 2101)
key_years = [2030, 2040, 2050, 2075, 2100]
idxs = [int(np.where(years == y)[0][0]) for y in key_years]

data = {}
for name, demand, label in scenarios:
    with open('data/processed/' + name + '.pkl', 'rb') as f:
        data[name] = pickle.load(f)

def make_table(caption, label, rows):
    yr_cols = ' & '.join(str(y) for y in key_years)
    L = []
    L.append(r'\begin{table}[h!]')
    L.append(r'\centering')
    L.append(r'\small')
    L.append(r'\caption{' + caption + '}')
    L.append(r'\label{tab:' + label + '}')
    L.append(r'\begin{tabular}{llrrrrr}')
    L.append(r'\toprule')
    L.append('Demand & Scenario & ' + yr_cols + r' \\')
    L.append(r'\midrule')
    prev = None
    for (name, demand, scen_label), row_vals in zip(scenarios, rows):
        if prev and demand != prev:
            L.append(r'\midrule')
        d = demand if demand != prev else ''
        L.append(d + ' & ' + scen_label + ' & ' + ' & '.join(row_vals) + r' \\')
        prev = demand
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\end{table}')
    L.append('')
    return '\n'.join(L)

tables = []

# 1. CO2 emissions (Gt/yr)
rows = []
for name, _, _ in scenarios:
    em = data[name]['emissions']['Total Emissions']
    rows.append(['{:.2f}'.format(em[i]/1000) for i in idxs])
tables.append(make_table(
    r'Annual CO\textsubscript{2} emissions by scenario (Gt\,CO\textsubscript{2}/yr).',
    'app_co2_emissions', rows))

# 2. Total steel stock (Gt)
rows = []
for name, _, _ in scenarios:
    stock_df = data[name]['stock_df']
    all_years = stock_df['year'].values
    stock = stock_df['total_stock'].values
    row = []
    for y in key_years:
        idx = np.where(all_years == y)[0]
        row.append('{:.1f}'.format(stock[idx[0]]/1e9) if len(idx) else '--')
    rows.append(row)
tables.append(make_table(
    'Total in-use steel stock by scenario (Gt).',
    'app_steel_stock', rows))

# 3. H2 demand (Mt/yr)
rows = []
for name, _, _ in scenarios:
    h2 = data[name]['h2']['H2 Demand (Mt)']
    rows.append(['{:.1f}'.format(h2[i]) for i in idxs])
tables.append(make_table(
    r'Annual hydrogen demand by scenario (Mt\,H\textsubscript{2}/yr).',
    'app_h2_demand', rows))

# 4. Electrolyzer stock (GW)
rows = []
for name, _, _ in scenarios:
    cap = data[name]['cap_results']
    total = cap['AEC'] + cap['PEM'] + cap['Other']
    rows.append(['{:.0f}'.format(total[i]) for i in idxs])
tables.append(make_table(
    'Total installed electrolyzer capacity by scenario (GW).',
    'app_elec_stock', rows))

# 5. Electrolyzer annual additions (GW/yr)
rows = []
for name, _, _ in scenarios:
    dsm = data[name]['dsm_elec']
    total = dsm['i_AEC'] + dsm['i_PEM'] + dsm['i_Other']
    rows.append(['{:.1f}'.format(total[i]) for i in idxs])
tables.append(make_table(
    'Annual electrolyzer capacity additions by scenario (GW/yr).',
    'app_elec_additions', rows))

# 6. Nickel inflow (kt/yr)
rows = []
for name, _, _ in scenarios:
    ni = data[name]['materials']['nickel_inflow']
    rows.append(['{:.2f}'.format(ni[i]) for i in idxs])
tables.append(make_table(
    'Annual nickel demand for electrolyzers by scenario (kt/yr).',
    'app_nickel', rows))

# 7. Iridium inflow (t/yr)
rows = []
for name, _, _ in scenarios:
    ir = data[name]['materials']['iridium_inflow']
    rows.append(['{:.2f}'.format(ir[i]*1000) for i in idxs])
tables.append(make_table(
    'Annual iridium demand for electrolyzers by scenario (t/yr).',
    'app_iridium', rows))

# 8. Platinum inflow (t/yr)
rows = []
for name, _, _ in scenarios:
    pt = data[name]['materials']['platinum_inflow']
    rows.append(['{:.2f}'.format(pt[i]*1000) for i in idxs])
tables.append(make_table(
    'Annual platinum demand for electrolyzers by scenario (t/yr).',
    'app_platinum', rows))

header = (
    '% Appendix tables — generated automatically\n'
    '% Include in your main document with \\input{appendix_tables}\n'
    '% Requires \\usepackage{booktabs} in preamble\n\n'
)

with open('data/processed/appendix_tables.tex', 'w', encoding='utf-8') as f:
    f.write(header + '\n\n'.join(tables))

print('Written to appendix_tables.tex')
