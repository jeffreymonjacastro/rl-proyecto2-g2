"""Construye el notebook y el runner remoto, sin acceder a credenciales."""
from pathlib import Path
import json
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

md("""
# M5 oficial: análisis exploratorio para RL multi-tienda

## Resumen
Análisis completo de las series de M5 y foco en **CA_1, CA_2 y CA_3** para el proyecto P2.
Los hallazgos numéricos se calculan al final y se guardan también en `resumen.md`.

## Contexto y método
Fuente única: [M5 Forecasting Accuracy, datos oficiales](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
Se usa `sales_train_evaluation.csv` como historia principal; `sales_train_validation.csv`
es un prefijo de esa misma historia y **no se concatena**. La observación es un producto-tienda-día;
el precio se observa por producto-tienda-semana. Las unidades vendidas no equivalen a demanda latente.

### Supuestos y límites
- Un cero de ventas no prueba un quiebre de inventario ni ausencia de demanda.
- El primer día con venta positiva es una aproximación a actividad comercial, no fecha de lanzamiento.
- Precio por unidades da ingresos brutos aproximados en USD; no margen ni beneficio neto.
- Inventario, reposición, costos, capacidad regional y acciones del agente no están registrados.
- Asociaciones con precio, eventos o SNAP son descriptivas, no efectos causales ni elasticidades identificadas.
- La inspección del último año es descriptiva; debe congelarse antes de ajustar modelos o políticas.
""")
md("""## 1. Configuración
Todos los CSV analíticos y PNG se escriben en `/kaggle/working`; cada figura tiene su tabla fuente.
""")
code("""
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

OUT = Path('/kaggle/working')
TABLES, FIGURES = OUT/'tables', OUT/'figures'
TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
STORES = ['CA_1', 'CA_2', 'CA_3']
COLORS = {'CA_1':'#0072B2', 'CA_2':'#D55E00', 'CA_3':'#009E73'}
plt.rcParams.update({'figure.figsize':(11,4.8), 'font.size':11,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.grid':True,
    'grid.alpha':0.2, 'figure.dpi':110, 'savefig.dpi':160})

def table(df, name):
    df.to_csv(TABLES/f'{name}.csv', index=False)
    return df

def finish(name):
    plt.tight_layout()
    plt.savefig(FIGURES/f'{name}.png', bbox_inches='tight')
    plt.show()
    plt.close()

print('Entorno:', platform.python_version(), 'pandas',pd.__version__,'numpy',np.__version__)
""")
md("""## 2. Archivos oficiales y trazabilidad
El runner monta la competición oficial. Fuera de Kaggle puede usarse `M5_DATA_DIR` o `kagglehub.competition_download`.
Se registran tamaños y SHA-256 para verificar la identidad de los CSV descargados localmente.
""")
code("""
import os
DATA = Path(os.environ.get('M5_DATA_DIR','/kaggle/input/m5-forecasting-accuracy'))
if not (DATA/'calendar.csv').exists():
    candidates = list(Path('/kaggle/input').rglob('sales_train_evaluation.csv')) if Path('/kaggle/input').exists() else []
    if candidates:
        DATA = candidates[0].parent
    else:
        import kagglehub
        DATA = Path(kagglehub.competition_download('m5-forecasting-accuracy'))
names = ['calendar.csv','sales_train_evaluation.csv','sales_train_validation.csv',
         'sell_prices.csv','sample_submission.csv']
inventory = []
for name in names:
    path = DATA/name
    assert path.is_file(), f'Falta {path}'
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            digest.update(chunk)
    inventory.append({'file':name,'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
display(table(pd.DataFrame(inventory),'official_file_manifest')[['file','bytes']])
""")
md("""## 3. Carga y validación estructural
Se carga toda la historia oficial, sin muestreo. Se mantiene formato ancho para evitar materializar
decenas de millones de filas. Los precios se unen solo después de agregar ventas por semana.
""")
code("""
calendar = pd.read_csv(DATA/'calendar.csv', parse_dates=['date'])
ids = ['id','item_id','dept_id','cat_id','store_id','state_id']
day_cols = pd.read_csv(DATA/'sales_train_evaluation.csv',nrows=0).columns[6:].tolist()
dtypes = {c:'int16' for c in day_cols}
dtypes.update({c:'category' for c in ids})
sales = pd.read_csv(DATA/'sales_train_evaluation.csv',dtype=dtypes)
prices = pd.read_csv(DATA/'sell_prices.csv',dtype={'store_id':'category','item_id':'category','sell_price':'float32'})
submission = pd.read_csv(DATA/'sample_submission.csv')
cal = calendar.set_index('d').loc[day_cols].reset_index()
values = sales[day_cols].to_numpy()
region = sales[sales.store_id.isin(STORES)].reset_index(drop=True)
rv = region[day_cols].to_numpy()
checks = []
def check(name, ok, detail):
    checks.append({'check':name,'passed':bool(ok),'detail':str(detail)})
    assert ok, f'{name}: {detail}'
check('unique_series',sales.id.is_unique,len(sales))
check('unique_item_store',not sales.duplicated(['item_id','store_id']).any(),'producto-tienda')
check('nonnegative_sales',(values>=0).all(),int(values.min()))
check('no_missing_sales',not sales.isna().any().any(),'todas las columnas de ventas')
check('calendar_unique',calendar.d.is_unique and calendar.date.is_unique,len(calendar))
check('consecutive_dates',cal.date.diff().dropna().eq(pd.Timedelta(days=1)).all(),len(cal))
check('unique_prices',not prices.duplicated(['store_id','item_id','wm_yr_wk']).any(),len(prices))
check('positive_prices',prices.sell_price.gt(0).all() and not prices.isna().any().any(),len(prices))
check('three_stores',set(region.store_id.unique())==set(STORES),STORES)
schema = pd.DataFrame([{'file':'sales_train_evaluation.csv','rows':len(sales),'columns':len(sales.columns)},
    {'file':'calendar.csv','rows':len(calendar),'columns':len(calendar.columns)},
    {'file':'sell_prices.csv','rows':len(prices),'columns':len(prices.columns)},
    {'file':'sample_submission.csv','rows':len(submission),'columns':len(submission.columns)}])
display(table(schema,'dataset_shapes'))
profile = pd.DataFrame([{'column':c,'dtype':str(sales[c].dtype),'unique':sales[c].nunique(),
    'missing':sales[c].isna().sum()} for c in ids])
display(table(profile,'hierarchy_profile'))
table(calendar.isna().sum().rename_axis('column').reset_index(name='missing_rows'),'calendar_missingness')
""")
md("""### Validación y evaluación: evitar historia duplicada
Se verifica por chunks que las ventas de validación coincidan exactamente con el prefijo de evaluación,
incluida la identidad producto-tienda y el sufijo de los identificadores.
""")
code("""
validation_days = pd.read_csv(DATA/'sales_train_validation.csv',nrows=0).columns[6:].tolist()
offset = 0
prefix_ok = True
for chunk in pd.read_csv(DATA/'sales_train_validation.csv',chunksize=4000,
                         dtype={c:'int16' for c in validation_days}):
    ref = sales.iloc[offset:offset+len(chunk)]
    prefix_ok &= np.array_equal(chunk[validation_days].to_numpy(),ref[validation_days].to_numpy())
    prefix_ok &= np.array_equal(chunk[ids[1:]].astype(str).to_numpy(),ref[ids[1:]].astype(str).to_numpy())
    prefix_ok &= np.array_equal(chunk.id.str.replace('_validation','',regex=False).to_numpy(),
                                ref.id.astype(str).str.replace('_evaluation','',regex=False).to_numpy())
    offset += len(chunk)
check('validation_is_prefix',prefix_ok and offset==len(sales),f'{offset} series; {len(validation_days)} días')
coverage = pd.DataFrame([
    {'scope':'historia validación','start':cal.date.min(),'end':cal.iloc[len(validation_days)-1].date,'days':len(validation_days)},
    {'scope':'historia evaluación','start':cal.date.min(),'end':cal.date.max(),'days':len(day_cols)},
    {'scope':'calendario oficial','start':calendar.date.min(),'end':calendar.date.max(),'days':len(calendar)}])
display(table(coverage,'temporal_coverage'))
print('sample_submission: plantilla para dos horizontes de 28 días; sus ceros no son ventas observadas.')
""")
md("""## 4. Panorama global y foco regional
Las comparaciones usan unidades, no ingresos. La suma regional debe coincidir con la suma de sus tiendas.
""")
code("""
daily = cal[['d','date','wm_yr_wk','wday','weekday','month','year','snap_CA','event_name_1','event_type_1',
             'event_name_2','event_type_2']].copy()
daily['all_units'] = values.sum(axis=0,dtype=np.int64)
for store in STORES:
    daily[store] = rv[region.store_id.eq(store)].sum(axis=0,dtype=np.int64)
daily['region_units'] = daily[STORES].sum(axis=1)
check('regional_reconciliation',daily.region_units.sum()==rv.sum(dtype=np.int64),daily.region_units.sum())
table(daily,'daily_sales')
store_totals = sales[['store_id','state_id']].copy()
store_totals['units'] = values.sum(axis=1,dtype=np.int64)
store_totals = store_totals.groupby(['state_id','store_id'],observed=True).units.sum().reset_index()
store_totals['share_all_units'] = store_totals.units/store_totals.units.sum()
table(store_totals,'store_totals')
fig, axes = plt.subplots(1,2,figsize=(13,5))
ordered = store_totals.sort_values('units')
axes[0].barh(ordered.store_id.astype(str),ordered.units/1e6,color='#0072B2')
axes[0].set(xlabel='Millones de unidades',title='M5 completo: ventas acumuladas por tienda')
for store in STORES:
    axes[1].plot(daily.date,daily[store].rolling(28,min_periods=28).mean(),label=store,color=COLORS[store])
axes[1].set(title='Tres tiendas: media móvil retrospectiva de 28 días',ylabel='Unidades / día',xlabel='Fecha')
axes[1].legend(); finish('01_scope_and_trend')
""")
md("""### Departamentos, categorías y concentración de productos
Se incluyen todos los productos. La curva de concentración usa ventas acumuladas del periodo completo;
no es un criterio para elegir productos de entrenamiento usando el test.
""")
code("""
series = region[ids].copy()
series['units'] = rv.sum(axis=1,dtype=np.int64)
series['mean_daily_units'] = rv.mean(axis=1)
series['zero_fraction_all_days'] = (rv==0).mean(axis=1)
first = (rv>0).argmax(axis=1)
has_sales = (rv>0).any(axis=1)
series['first_positive_day_index'] = np.where(has_sales,first,np.nan)
active_days = np.where(has_sales,len(day_cols)-first,0)
positive_days = (rv>0).sum(axis=1)
series['observed_days_since_first_sale'] = active_days
series['zero_fraction_since_first_sale'] = np.divide(active_days-positive_days,active_days,
    out=np.full(len(rv),np.nan),where=active_days>0)
series['positive_days'] = positive_days
series['std_daily_units'] = rv.std(axis=1)
series['cv_daily_units'] = series.std_daily_units/series.mean_daily_units.replace(0,np.nan)
table(series,'series_profile_ca123')
dept = series.groupby(['store_id','dept_id'],observed=True).units.sum().reset_index()
table(dept,'department_sales')
category = series.groupby(['store_id','cat_id'],observed=True).units.sum().reset_index()
table(category,'category_sales')
fig, axes = plt.subplots(1,2,figsize=(13,5))
dept.pivot(index='dept_id',columns='store_id',values='units').div(1e6).plot.bar(ax=axes[0],color=COLORS,rot=30)
axes[0].set(title='Ventas por departamento y tienda',ylabel='Millones de unidades',xlabel='Departamento')
item = series.groupby('item_id',observed=True).units.sum().sort_values(ascending=False)
concentration = pd.DataFrame({'item_id':item.index.astype(str),'units':item.values,
    'item_fraction':np.arange(1,len(item)+1)/len(item),'cumulative_unit_share':item.cumsum().values/item.sum()})
table(concentration,'item_concentration')
axes[1].plot(concentration.item_fraction*100,concentration.cumulative_unit_share*100,color='#0072B2')
axes[1].set(title='Concentración regional de ventas por producto',xlabel='Productos ordenados por ventas (%)',ylabel='Unidades acumuladas (%)',xlim=(0,100),ylim=(0,100))
finish('02_departments_concentration')
""")
md("""## 5. Ceros e intermitencia
Se separa el periodo previo a la primera venta del periodo posterior. Aun después de esa fecha,
los ceros pueden corresponder a baja demanda, indisponibilidad u otras causas no identificables.
""")
code("""
zeros = series.groupby('store_id',observed=True).agg(series=('id','size'),
    mean_zero_fraction=('zero_fraction_all_days','mean'),
    mean_zero_fraction_since_first_sale=('zero_fraction_since_first_sale','mean'),
    median_daily_units=('mean_daily_units','median')).reset_index()
display(table(zeros,'zero_summary'))
fig, axes = plt.subplots(1,2,figsize=(13,4.8))
for store in STORES:
    subset = series[series.store_id.eq(store)]
    axes[0].hist(subset.zero_fraction_since_first_sale.dropna()*100,bins=np.arange(0,105,5),
        histtype='step',linewidth=2,label=store,color=COLORS[store])
    axes[1].scatter(subset.mean_daily_units,subset.cv_daily_units,s=8,alpha=.25,label=store,color=COLORS[store])
axes[0].set(title='Ceros desde la primera venta positiva',xlabel='Días con cero ventas (%)',ylabel='Series producto-tienda')
axes[1].set(title='Volumen y variabilidad diaria',xlabel='Media de unidades / día (escala log)',ylabel='Coeficiente de variación',xscale='log')
axes[0].legend(); axes[1].legend(); finish('03_intermittency')
""")
md("""## 6. Estacionalidad semanal y mensual
Se comparan promedios diarios; el mes de calendario puede mezclar tendencia, surtido y años incompletos.
La tabla mensual conserva el número de días de cada periodo para evitar comparar totales de distinta duración.
""")
code("""
weekday = daily.groupby(['wday','weekday'])[STORES].mean().reset_index().sort_values('wday')
table(weekday,'weekday_mean_sales')
monthly = daily.assign(period=daily.date.dt.to_period('M').astype(str)).groupby('period').agg(
    days=('date','size'),region_units=('region_units','sum'),mean_daily_units=('region_units','mean')).reset_index()
table(monthly,'monthly_sales')
heat = daily.assign(year=daily.date.dt.year,month=daily.date.dt.month).groupby(['year','month']).region_units.mean().unstack()
table(heat.reset_index(),'year_month_mean_daily_sales')
fig, axes = plt.subplots(1,2,figsize=(13,4.8))
for store in STORES:
    axes[0].plot(weekday.weekday,weekday[store],marker='o',label=store,color=COLORS[store])
axes[0].tick_params(axis='x',rotation=40)
axes[0].set(title='Patrón semanal observado',ylabel='Unidades / día',xlabel='Día de la semana'); axes[0].legend()
image = axes[1].imshow(heat.to_numpy(),aspect='auto',cmap='Blues')
axes[1].set(xticks=range(12),xticklabels=range(1,13),yticks=range(len(heat)),yticklabels=heat.index,
    title='Promedio diario regional por año y mes',xlabel='Mes',ylabel='Año')
fig.colorbar(image,ax=axes[1],label='Unidades / día'); finish('04_seasonality')
""")
md("""### Dependencia temporal y coordinación entre tiendas
La ACF es descriptiva de la serie agregada sin desestacionalizar. Se reportan tanto niveles como
primeras diferencias para no confundir tendencias compartidas con acoplamiento causal.
""")
code("""
acf = pd.DataFrame({'lag_days':np.arange(1,57)})
for store in STORES:
    acf[store] = [daily[store].autocorr(lag=int(lag)) for lag in acf.lag_days]
table(acf,'sales_autocorrelation')
corr_level = daily[STORES].corr()
corr_diff = daily[STORES].diff().dropna().corr()
table(corr_level.rename_axis('store').reset_index(),'store_correlation_levels')
table(corr_diff.rename_axis('store').reset_index(),'store_correlation_first_differences')
fig, axes = plt.subplots(1,2,figsize=(13,4.8))
for store in STORES:
    axes[0].plot(acf.lag_days,acf[store],label=store,color=COLORS[store])
axes[0].set(title='Autocorrelación de unidades diarias',xlabel='Rezago (días)',ylabel='Correlación',ylim=(-1,1)); axes[0].legend()
im=axes[1].imshow(corr_diff,vmin=-1,vmax=1,cmap='RdBu_r')
axes[1].set(xticks=range(3),xticklabels=STORES,yticks=range(3),yticklabels=STORES,title='Correlación de cambios diarios')
for i in range(3):
    for j in range(3):
        axes[1].text(j,i,f'{corr_diff.iloc[i,j]:.2f}',ha='center',va='center',color='black',bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
fig.colorbar(im,ax=axes[1]); finish('05_temporal_dependence')
""")
md("""## 7. Eventos y SNAP
SNAP indica días de elegibilidad para compras con beneficios en California. Los promedios no controlan
por día de semana, tendencia, surtido ni promociones; no permiten atribuir causalidad.
Los eventos simultáneos se cuentan como día con evento una sola vez en la comparación binaria.
""")
code("""
daily['has_event'] = daily.event_name_1.notna() | daily.event_name_2.notna()
events = daily.groupby('has_event').agg(days=('date','size'),mean_daily_units=('region_units','mean')).reset_index()
snap = daily.groupby('snap_CA').agg(days=('date','size'),mean_daily_units=('region_units','mean')).reset_index()
table(events,'event_day_comparison'); table(snap,'snap_day_comparison')
event_rows = pd.concat([daily[['date','region_units',col]].rename(columns={col:'event'})
    for col in ['event_name_1','event_name_2']]).dropna().drop_duplicates(['date','event'])
event_detail = event_rows.groupby('event').agg(days=('date','size'),mean_daily_units=('region_units','mean')).reset_index()
table(event_detail,'named_event_sales')
fig, axes=plt.subplots(1,2,figsize=(11,4.8))
axes[0].bar(['Sin evento','Con evento'],events.mean_daily_units,color=['#777777','#0072B2'])
axes[1].bar(['No SNAP','SNAP'],snap.mean_daily_units,color=['#777777','#009E73'])
for ax,frame in zip(axes,[events,snap]):
    ax.set(ylabel='Unidades regionales / día')
    for i,row in frame.iterrows():
        ax.text(i,row.mean_daily_units,f'n={int(row.days)} días',ha='center',va='bottom')
axes[0].set_title('Asociación descriptiva con eventos'); axes[1].set_title('Asociación descriptiva con SNAP CA')
finish('06_events_snap')
""")
md("""## 8. Ventas y precios a la misma granularidad semanal
Se agregan únicamente días históricos observados. Las semanas parciales conservan su denominador.
La unión es many-to-one por tienda, producto y semana. Los precios faltantes no se imputan a cero.
""")
code("""
weeks = cal.wm_yr_wk.to_numpy()
unique_weeks = pd.unique(weeks)
weekly_values = np.column_stack([rv[:,weeks==week].sum(axis=1,dtype=np.int64) for week in unique_weeks])
weekly = pd.DataFrame(weekly_values,index=pd.MultiIndex.from_frame(region[['store_id','item_id']]),columns=unique_weeks)
weekly = weekly.rename_axis(columns='wm_yr_wk').stack().rename('units').reset_index()
week_meta = cal.groupby('wm_yr_wk').agg(date_start=('date','min'),date_end=('date','max'),observed_days=('date','size')).reset_index()
weekly = weekly.merge(week_meta,on='wm_yr_wk',validate='many_to_one')
weekly = weekly.merge(prices,on=['store_id','item_id','wm_yr_wk'],how='left',validate='many_to_one')
weekly['units_per_observed_day'] = weekly.units/weekly.observed_days
weekly['gross_revenue_proxy_usd'] = weekly.units*weekly.sell_price
positive_without_price = int(((weekly.units>0)&weekly.sell_price.isna()).sum())
check('positive_sales_have_price',positive_without_price==0,positive_without_price)
table(weekly,'weekly_sales_prices_ca123')
price_coverage = weekly.groupby('store_id',observed=True).agg(rows=('units','size'),
    price_missing=('sell_price',lambda s:s.isna().sum()),units=('units','sum'),
    gross_revenue_proxy_usd=('gross_revenue_proxy_usd','sum')).reset_index()
display(table(price_coverage,'weekly_price_coverage'))
print('Los faltantes de precio con cero ventas pueden reflejar surtido no activo; no son automáticamente errores.')
""")
md("""### Variación de precios por producto-tienda
El resumen usa solo semanas dentro de la historia observada y con precio disponible.
La distribución de precios no ponderada describe registros semanales, no una canasta de consumo constante.
""")
code("""
priced = weekly.dropna(subset=['sell_price']).sort_values(['store_id','item_id','date_start']).copy()
group = priced.groupby(['store_id','item_id'],observed=True)
priced['previous_price'] = group.sell_price.shift(1)
priced['changed'] = priced.previous_price.notna() & ~np.isclose(priced.sell_price,priced.previous_price)
price_series = group.agg(observed_weeks=('sell_price','size'),unique_prices=('sell_price','nunique'),
    min_price=('sell_price','min'),max_price=('sell_price','max'),mean_price=('sell_price','mean'),
    price_changes=('changed','sum')).reset_index()
table(price_series,'price_variation_by_series')
price_desc = priced.groupby('store_id',observed=True).sell_price.describe(percentiles=[.05,.5,.95]).reset_index()
table(price_desc,'weekly_price_distribution')
fig, axes=plt.subplots(1,2,figsize=(13,4.8))
for store in STORES:
    axes[0].hist(priced.loc[priced.store_id.eq(store),'sell_price'],bins=np.geomspace(float(priced.sell_price.min())*.99,float(priced.sell_price.max())*1.01,65),histtype='step',
        label=store,linewidth=1.8,color=COLORS[store])
axes[0].set(title='Precios semanales observados',xlabel='Precio USD (escala log)',ylabel='Registros producto-tienda-semana',xscale='log'); axes[0].legend()
axes[1].hist(price_series.unique_prices,bins=30,color='#0072B2')
axes[1].set(title='Variación de precios disponible por serie',xlabel='Precios distintos observados',ylabel='Series producto-tienda')
finish('07_price_variation')
""")
md("""### Un producto ilustrativo: precio y ventas
Se elige el producto de mayor volumen regional **en entrenamiento**. El doble panel evita mezclar
escalas. La asociación histórica no identifica respuesta contrafactual ante precios decididos por un agente.
""")
code("""
test_start = cal.date.max()-pd.Timedelta(days=364)
val_start = test_start-pd.Timedelta(days=90)
train_mask = cal.date.lt(val_start).to_numpy()
train_units = pd.DataFrame({'item_id':region.item_id.astype(str),'units':rv[:,train_mask].sum(axis=1,dtype=np.int64)})
chosen = train_units.groupby('item_id').units.sum().idxmax()
example = weekly[weekly.item_id.eq(chosen)].copy()
table(example,'example_product_weekly')
fig, axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
for store in STORES:
    s = example[example.store_id.eq(store)]
    axes[0].plot(s.date_start,s.units_per_observed_day,label=store,color=COLORS[store],alpha=.85)
    axes[1].plot(s.date_start,s.sell_price,label=store,color=COLORS[store])
for ax in axes:
    ax.axvline(test_start,color='black',linestyle='--',label='Inicio test')
axes[0].set(title=f'Producto seleccionado solo con entrenamiento: {chosen}',ylabel='Unidades / día observado'); axes[0].legend(ncol=4)
axes[1].set(ylabel='Precio USD',xlabel='Fecha'); finish('08_example_product')
""")
md("""## 9. Partición temporal reproducible
Propuesta: último año observado (365 días) como test; 90 días anteriores como validación;
resto como entrenamiento. Sin mezcla aleatoria ni normalización con estadísticas futuras.
Los últimos 28 días de evaluación oficial están dentro del test propuesto, no constituyen otro dataset independiente.
""")
code("""
daily['split'] = np.select([daily.date.lt(val_start),daily.date.lt(test_start)],['train','validation'],default='test')
split = daily.groupby('split',sort=False).agg(start=('date','min'),end=('date','max'),days=('date','size'),
    region_units=('region_units','sum'),mean_daily_units=('region_units','mean')).reset_index()
check('test_365_days',int(split.loc[split.split.eq('test'),'days'].iloc[0])==365,split.to_dict('records'))
check('all_days_partitioned',split.days.sum()==len(day_cols),len(day_cols))
check('split_disjoint',split.end.iloc[0]<split.start.iloc[1] and split.end.iloc[1]<split.start.iloc[2],'orden cronológico')
display(table(split,'temporal_split'))
table(daily[['d','date','split']],'day_split_assignments')
table(daily,'daily_sales')
fig,ax=plt.subplots(figsize=(12,4.8))
for name,color in [('train','#0072B2'),('validation','#E69F00'),('test','#009E73')]:
    s=daily[daily.split.eq(name)]
    ax.plot(s.date,s.region_units,label=name,color=color,linewidth=.8)
ax.set(title='Partición temporal propuesta: CA_1 + CA_2 + CA_3',ylabel='Unidades / día',xlabel='Fecha'); ax.legend()
finish('09_temporal_split')
""")
md("""## 10. Elegibilidad de productos para futuros experimentos
Diagnóstico, no selección definitiva: exige actividad positiva antes del fin de entrenamiento,
al menos 180 días desde primera venta y precios en las últimas 26 semanas de entrenamiento, en las tres tiendas.
Todos los criterios se calculan sin usar validación o test. El ranking por volumen no garantiza elasticidad estimable.
""")
code("""
train_end = cal.loc[train_mask,'date'].max()
recent_start = train_end-pd.Timedelta(days=181)
recent = weekly[weekly.date_end.le(train_end)&weekly.date_start.ge(recent_start)].copy()
availability = recent.groupby(['store_id','item_id'],observed=True).agg(weeks=('units','size'),
    priced_weeks=('sell_price','count'),unique_prices=('sell_price','nunique')).reset_index()
candidate = region[ids].copy()
candidate['train_units'] = rv[:,train_mask].sum(axis=1,dtype=np.int64)
candidate['train_positive_days'] = (rv[:,train_mask]>0).sum(axis=1)
candidate['train_days_since_first_sale'] = np.where(candidate.train_positive_days.gt(0),int(train_mask.sum())-first,0)
candidate = candidate.merge(availability,on=['store_id','item_id'],how='left',validate='one_to_one')
candidate['eligible_store'] = candidate.train_positive_days.gt(0)&candidate.train_days_since_first_sale.ge(180)&candidate.weeks.ge(25)&candidate.priced_weeks.eq(candidate.weeks)
table(candidate,'training_only_candidate_store_diagnostics')
candidates = candidate.groupby('item_id',observed=True).agg(stores=('store_id','nunique'),eligible_stores=('eligible_store','sum'),
    train_units=('train_units','sum'),min_unique_prices=('unique_prices','min')).reset_index()
candidates['eligible_all_three'] = candidates.stores.eq(3)&candidates.eligible_stores.eq(3)
candidates = candidates.sort_values(['eligible_all_three','train_units'],ascending=False)
table(candidates,'training_only_product_candidates')
display(candidates.head(10))
""")
md("""## 11. Hallazgos ejecutados e implicaciones para RL
Los siguientes resultados son descriptivos. Un simulador deberá declarar sus supuestos de inventario,
costos y respuesta al precio; no puede recuperar esas variables ausentes como hechos históricos.
""")
code("""
from IPython.display import Markdown
regional_share = daily.region_units.sum()/daily.all_units.sum()
top10_share = concentration.loc[concentration.item_fraction.le(.10),'units'].sum()/concentration.units.sum()
price_constant_share = price_series.unique_prices.eq(1).mean()
summary = f'''# Hallazgos del EDA M5 oficial

- Historia observada: {cal.date.min().date()} a {cal.date.max().date()}, {len(day_cols):,} días.
- M5 completo: {len(sales):,} series, {sales.item_id.nunique():,} productos y {sales.store_id.nunique()} tiendas.
- Foco CA_1/CA_2/CA_3: {len(region):,} series y {daily.region_units.sum():,} unidades; {regional_share:.1%} del volumen total M5.
- Ceros regionales (todos los producto-tienda-días): {(rv==0).mean():.1%}. No interpretarlos automáticamente como stockouts.
- El 10% superior de productos concentra {top10_share:.1%} de las unidades regionales (historia completa).
- Series con un solo precio observado: {price_constant_share:.1%}; la variación restante tampoco demuestra elasticidad causal.
- Ventas positivas sin precio semanal: {positive_without_price}. Los ingresos aproximados cubren las ventas positivas comprobadas.
- Test propuesto: {test_start.date()} a {cal.date.max().date()}; validación: {val_start.date()} a {(test_start-pd.Timedelta(days=1)).date()}.
- Productos que cumplen elegibilidad en las tres tiendas usando solo entrenamiento: {int(candidates.eligible_all_three.sum()):,}.

## Implicaciones y límites
1. El estado del agente puede usar calendario, SNAP y rezagos de ventas/precios conocidos al decidir; las ventanas móviles deben desplazarse al menos un día.
2. La coordinación debe evaluarse contra baselines bajo el mismo simulador, costos, capacidad y demanda latente supuesta.
3. M5 no contiene inventario, reposiciones, costos ni beneficio neto histórico. No permite demostrar directamente reducción real de stockouts ni superar el beneficio real de Walmart.
4. No equiparar ingreso bruto con beneficio. No tratar ventas observadas como demanda no censurada ni ceros como inventario agotado.
5. Estimar sensibilidad al precio exige controles y supuestos explícitos; no se ajusta una elasticidad causal a partir de correlaciones agregadas.
6. Eventos y SNAP muestran asociaciones potencialmente confundidas. No atribuir efectos causales con estas comparaciones.
7. Separar test antes de ajustar preprocesamiento, selección de productos, potencial PBRS, costos o hiperparámetros. El perfil completo del test aquí es solo descriptivo.

Fuente: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
'''
(OUT/'resumen.md').write_text(summary,encoding='utf-8')
display(Markdown(summary))
table(pd.DataFrame(checks),'quality_checks')
(OUT/'analysis_metadata.json').write_text(json.dumps({'competition':'m5-forecasting-accuracy',
    'stores':STORES,'main_sales_file':'sales_train_evaluation.csv','rows':len(sales),'days':len(day_cols),
    'region_series':len(region),'test_start':str(test_start.date()),'test_end':str(cal.date.max().date()),
    'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
    'matplotlib':matplotlib.__version__,'checks_passed':len(checks)},indent=2),encoding='utf-8')
print(f'Exportados {len(list(TABLES.glob("*.csv")))} CSV y {len(list(FIGURES.glob("*.png")))} PNG.')
""")

for i, cell in enumerate(cells):
    cell['id'] = f'm5-eda-{i:03d}'
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
nbf.validate(nb)
nbf.write(nb, ROOT/'input'/'m5_eda.ipynb')
# nbclient saves actual executed cells and inline figures as a downloadable artifact.
payload = json.dumps(nb,ensure_ascii=False)
runner = '''"""Ejecuta en Kaggle y guarda el notebook completo con sus salidas reales."""
import json, time, traceback
from pathlib import Path
import nbformat
from nbclient import NotebookClient

OUT = Path('/kaggle/working')
OUT.mkdir(exist_ok=True)
progress = {'status':'running','stage':'execute_notebook'}
progress_path = OUT/'run_summary.json'
progress_path.write_text(json.dumps(progress),encoding='utf-8')
started = time.time()
notebook = nbformat.reads(NOTEBOOK_PAYLOAD,as_version=4)
client = NotebookClient(notebook,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(OUT)}})
try:
    client.execute()
    progress.update(status='complete',stage='done')
except Exception as exc:
    progress.update(status='failed',stage='execute_notebook',error=repr(exc),traceback=traceback.format_exc())
finally:
    nbformat.write(notebook,OUT/'m5_eda_executed.ipynb')
    progress['elapsed_seconds'] = round(time.time()-started,2)
    progress['executed_code_cells'] = sum(c.cell_type=='code' and c.get('execution_count') is not None for c in notebook.cells)
    progress['artifacts'] = [str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()]
    progress_path.write_text(json.dumps(progress,indent=2),encoding='utf-8')
    print(json.dumps(progress,indent=2))
    import zipfile
    with zipfile.ZipFile(OUT/'eda_artifacts.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for p in OUT.rglob('*'):
            if p.is_file() and p.name!='eda_artifacts.zip':
                bundle.write(p,p.relative_to(OUT))
'''.replace('notebook = nbformat.reads(NOTEBOOK_PAYLOAD,as_version=4)',f'notebook = nbformat.reads({payload!r},as_version=4)')
# ASCII escapes prevent legacy CLI Windows encodings from corrupting Spanish text.
(ROOT/'input'/'run_eda.py').write_text(runner.encode('ascii','backslashreplace').decode('ascii'),encoding='ascii')
print(f'Notebook validado: {len(cells)} celdas; {sum(c.cell_type=="code" for c in cells)} de código.')
