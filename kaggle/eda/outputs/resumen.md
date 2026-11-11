# Hallazgos del EDA M5 oficial

- Historia observada: 2011-01-29 a 2016-05-22, 1,941 días.
- M5 completo: 30,490 series, 3,049 productos y 10 tiendas.
- Foco CA_1/CA_2/CA_3: 9,147 series y 25,014,183 unidades; 37.4% del volumen total M5.
- Ceros regionales (todos los producto-tienda-días): 64.0%. No interpretarlos automáticamente como stockouts.
- El 10% superior de productos concentra 50.8% de las unidades regionales (historia completa).
- Series con un solo precio observado: 29.1%; la variación restante tampoco demuestra elasticidad causal.
- Ventas positivas sin precio semanal: 0. Los ingresos aproximados cubren las ventas positivas comprobadas.
- Test propuesto: 2015-05-24 a 2016-05-22; validación: 2015-02-23 a 2015-05-23.
- Productos que cumplen elegibilidad en las tres tiendas usando solo entrenamiento: 2,592.

## Implicaciones y límites
1. El estado del agente puede usar calendario, SNAP y rezagos de ventas/precios conocidos al decidir; las ventanas móviles deben desplazarse al menos un día.
2. La coordinación debe evaluarse contra baselines bajo el mismo simulador, costos, capacidad y demanda latente supuesta.
3. M5 no contiene inventario, reposiciones, costos ni beneficio neto histórico. No permite demostrar directamente reducción real de stockouts ni superar el beneficio real de Walmart.
4. No equiparar ingreso bruto con beneficio. No tratar ventas observadas como demanda no censurada ni ceros como inventario agotado.
5. Estimar sensibilidad al precio exige controles y supuestos explícitos; no se ajusta una elasticidad causal a partir de correlaciones agregadas.
6. Eventos y SNAP muestran asociaciones potencialmente confundidas. No atribuir efectos causales con estas comparaciones.
7. Separar test antes de ajustar preprocesamiento, selección de productos, potencial PBRS, costos o hiperparámetros. El perfil completo del test aquí es solo descriptivo.

Fuente: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
