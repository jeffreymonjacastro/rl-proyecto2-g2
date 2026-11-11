# EDA oficial de Walmart M5

Análisis exploratorio del dataset completo de `m5-forecasting-accuracy`, con foco en `CA_1`, `CA_2` y `CA_3` para la vía retail del proyecto de RL.

## Archivos

- `input/m5_eda.ipynb`: notebook fuente, con celdas separadas y explicaciones en español.
- `input/run_eda.py`: runner que ejecuta ese notebook en Kaggle mediante `nbclient` y guarda las salidas de las celdas, tablas y figuras.
- `input/kernel-metadata.json`: competición oficial adjunta; kernel privado.
- `outputs/m5_eda_executed.ipynb`: notebook ejecutado remotamente, incluidas figuras y resultados.
- `outputs/tables/`: CSV analíticos, tablas fuente de gráficos y controles de calidad.
- `outputs/figures/`: gráficos PNG.
- `outputs/resumen.md`: hallazgos calculados y límites del dataset.
- `outputs/analysis_metadata.json`: alcance, versiones y partición temporal.
- `outputs/run_summary.json`: resultado interno real de ejecución, independiente del estado externo de Kaggle.
- `data/`: los cinco CSV oficiales y su ZIP descargado, excluidos de Git por tamaño. No hay credenciales en esta carpeta.

Fuente: [datos oficiales de M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
Kernel: [jeffreyamc/rl-p2-m5-official-eda](https://www.kaggle.com/code/jeffreyamc/rl-p2-m5-official-eda).

## Reproducir en Kaggle

Desde la raíz del repositorio, con Kaggle CLI autenticada y acceso a la competición:

```powershell
python -m pip install nbformat
python kaggle/eda/build_notebook.py
Push-Location kaggle/eda/input
kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels status jeffreyamc/rl-p2-m5-official-eda
Pop-Location
```

Esperar estado `COMPLETE` antes de descargar en una carpeta vacía para esa ejecución:

```powershell
kaggle kernels output jeffreyamc/rl-p2-m5-official-eda -p kaggle/eda/download --file-pattern eda_artifacts.zip --force
Expand-Archive -LiteralPath kaggle/eda/download/eda_artifacts.zip -DestinationPath kaggle/eda/outputs -Force
```

Comprobar además que `run_summary.json` diga `status: complete`, que `quality_checks.csv` no tenga fallos y que no existan salidas de error en el notebook. El runner guarda el notebook parcial incluso si alguna celda falla; un estado externo `COMPLETE` por sí solo no prueba que el EDA haya terminado.

Para recuperar nuevamente los datos originales:

```powershell
kaggle competitions download -c m5-forecasting-accuracy -p kaggle/eda/data
Expand-Archive -LiteralPath kaggle/eda/data/m5-forecasting-accuracy.zip -DestinationPath kaggle/eda/data -Force
```

`official_file_manifest.csv` contiene SHA-256 de los CSV leídos en Kaggle, para contrastar con los descargados. Los CSV originales no se adjuntan como outputs del kernel porque ya se descargan directamente desde la competición.

La descarga agrupada comprime el CSV semanal grande durante el transporte; se entrega descomprimido en `outputs/tables/`. Para verificar los archivos locales después de descargarlos:

```powershell
python kaggle/eda/verify_delivery.py
```

## Alcance y precauciones de interpretación

Se analiza la historia de evaluación sin concatenar la historia de validación, que contiene las mismas series con un horizonte más corto. La plantilla de submission no contiene observaciones reales.

El EDA cubre jerarquía de productos, ventas por tienda y departamento, concentración, ceros e intermitencia, estacionalidad, autocorrelación, eventos, SNAP, cobertura y variación de precios, asociación ilustrativa de precio/ventas y partición temporal. Las tablas semanales indican los días observados y no imputan precios faltantes.

El test propuesto son los últimos 365 días observados, con 90 días previos de validación. El diagnóstico de productos elegibles y el producto ilustrativo usan solo entrenamiento. La exploración del test es descriptiva: no debe usarse para ajustar modelos, políticas ni supuestos del simulador.

M5 no contiene inventario, costos, reposiciones, demanda perdida ni beneficio neto histórico. Las ventas no equivalen necesariamente a demanda; los ceros no demuestran stockouts; precio por unidades es una aproximación a ingreso bruto. El análisis no estima elasticidad causal ni verifica resultados de agentes RL. Para el proyecto deberán declararse y validarse por separado los supuestos del simulador.
