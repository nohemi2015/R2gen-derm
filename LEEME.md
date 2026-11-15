# R2Gen para lesiones de piel (HAM10000 y Brown Skin)

Este código parte del R2Gen original (Chen et al., EMNLP 2020, https://github.com/zhjohnchan/R2Gen) y contiene solo los cambios necesarios para los experimentos del artículo. El modelo y la idea son los mismos; lo que cambia es que ahora cada afirmación del artículo se puede reproducir y verificar.

## Qué cambió respecto a R2Gen y por qué

| Cambio | Archivo | Problema que resuelve |
| --- | --- | --- |
| ViT-large real (timm, ImageNet-21k) y `d_vf` fijado automáticamente (ViT-L 1024, ViT-B 768, ResNet-101 2048) | `modules/visual_extractor.py`, `main_*.py` | En los scripts anteriores no se pasaba `--visual_extractor`, así que se usaba ResNet-101 por defecto. Ahora, si `--d_vf` no coincide con el extractor, el programa se detiene con error |
| Normalización acorde con el extractor (ViT augreg: media y desviación 0.5) | `modules/dataloaders.py` | Usar la normalización equivocada degrada las características |
| `--variant base / rm / rm_mcln` | `modules/encoder_decoder.py` | Permite la tabla de ablación con el mismo formato que R2Gen (BASE, +RM, +RM+MCLN) |
| El vocabulario se guarda en `vocab.json` al entrenar y se reutiliza al probar | `modules/tokenizers.py`, `main_train.py`, `main_test.py` | El experimento "con clasificación" se entrenó con un archivo de anotaciones y se probó con otro, lo que desfasa los índices del vocabulario |
| `predictions.csv` (id, referencia, generado) y `metrics.json` por prueba | `modules/tester.py` | Las figuras de ejemplo se toman de aquí, emparejadas por id |
| Solo se registra el modelo elegido en validación | `modules/trainer.py` | R2Gen también guardaba "el mejor resultado en test", que equivale a elegir el modelo con el conjunto de prueba; ese resultado no se debe reportar |
| Funciona en CPU y con pandas 2 | `modules/att_model.py`, `modules/caption_model.py`, `modules/trainer.py` | Había llamadas a `.cuda()` fijas en el código y se usaba `DataFrame.append`, que ya no existe en pandas 2 |

## Herramientas nuevas

- `tools/make_splits.py audit`: revisa si una misma lesión aparece en más de una partición. Conviene correrlo primero sobre los JSON actuales.
- `tools/make_splits.py split`: crea particiones 70/10/20 en las que ninguna lesión se repite entre particiones, estratificadas por diagnóstico. En HAM10000 agrupa por `lesion_id`; en Brown Skin, las 6 versiones de una misma imagen de origen quedan juntas.
- `tools/baselines.py`: dos líneas base que no generan texto, la recuperación del vecino más cercano y clasificador + plantilla.
- `tools/evaluate.py`: calcula las métricas NLG globales, por clase y por fototipo, más tres medidas de corrección:
  - `dx_acc`: si el diagnóstico mencionado en el reporte generado es el correcto;
  - `dx_f1`: F1 macro de ese diagnóstico;
  - `pt_acc`: si el fototipo mencionado es el correcto.

  Con varias semillas reporta media ± desviación estándar.
- `tools/collect_tables.py`: arma la tabla principal y las tablas por fototipo y por clase en CSV y Markdown.
- `tools/build_brown_skin.py`: arma las particiones por lesión y los pares claro/oscuro de Brown Skin.
- `tools/paired_tone.py`: compara la misma lesión en piel clara y en piel oscura.
- `scripts/run_experiments.py <ham10000|ham10000_bs|iu_xray>`: corre todo el diseño experimental; funciona en Windows.

## Orden de trabajo (Windows)

Todo se corre desde dentro de la carpeta `r2gen-derm`. Las rutas son las de su computadora.

```
pip install -r requirements.txt

REM 1. Copiar las carpetas de Brown Skin junto a las imágenes de HAM10000
REM    (data\ham10000\images debe contener HAM10000_images_all, ISIC2018_Task3_Test_Images,
REM     "100 malignas" y "100 benignas")

REM 2. Construir los archivos de anotaciones (particiones por lesión + pares claro/oscuro)
python tools\build_brown_skin.py --ham_ann "C:\1. HAM10000\data\ham10000\annotation_ham10000_full.json" --ham_metadata "C:\1. HAM10000\HAM10000_metadata.csv" --isic_gt "C:\1. HAM10000\data\ham10000\ISIC2018_Task3_Test_GroundTruth.csv" --image_root "C:\1. HAM10000\data\ham10000\images" --bs_dirs "100 malignas" "100 benignas" --out_dir "C:\1. HAM10000\data\brown_skin_v2"

REM 3. Experimentos (primero con --dry_run para ver los comandos)
python scripts\run_experiments.py ham10000    --image_dir "C:\1. HAM10000\data\ham10000\images" --ann_dir "C:\1. HAM10000\data\brown_skin_v2"
python scripts\run_experiments.py ham10000_bs --image_dir "C:\1. HAM10000\data\ham10000\images" --ann_dir "C:\1. HAM10000\data\brown_skin_v2"
python scripts\run_experiments.py iu_xray     --image_dir "C:\1. HAM10000\data\iu_xray\images" --iu_ann "C:\1. HAM10000\data\iu_xray\annotation.json"
```

Las tablas quedan en `results\<experimento>\tables` (prueba de ISIC 2018) y en `results\<experimento>\tables_bs_pairs` (comparación de piel clara contra oscura).

## Diseño experimental

**Brown Skin.** Son 182 imágenes de las carpetas "100 malignas" y "100 benignas". Cada una es una lesión de HAM10000 o de la prueba de ISIC 2018 recolorada a piel oscura, así que cada imagen tiene su original en piel clara. El script `build_brown_skin.py` las reparte así:

- 83 lesiones cuyo original está en la prueba de ISIC 2018 forman el **par de prueba**: la misma lesión en piel clara y en piel oscura. Ninguno de los modelos las ve al entrenar.
- 89 imágenes cuyo original está en el entrenamiento de HAM10000 solo se usan en el experimento `ham10000_bs`, como datos extra de entrenamiento.
- Se excluyen 10 imágenes sin original ni diagnóstico (ISIC_0094098_49, etc.).

| Experimento | Entrena con | Se evalúa en | Pregunta que responde |
| --- | --- | --- | --- |
| ham10000 | HAM10000 (partición por lesión) | prueba de ISIC 2018 (1,511) y par claro/oscuro (83 + 83) | ¿Cuánto vale el modelo? ¿Pierde calidad en piel oscura? |
| ham10000_bs | HAM10000 + 89 imágenes oscuras de Brown Skin | lo mismo | ¿Agregar piel oscura reduce esa diferencia? |
| iu_xray | IU X-Ray | IU X-Ray | ¿La arquitectura sigue funcionando en radiología? |

Configuraciones de `ham10000`, cada una con 3 semillas:

| Extractor | BASE | +RM | +RM+MCLN |
| --- | --- | --- | --- |
| ResNet-101 (R2Gen original) | ✓ | ✓ | ✓ |
| ViT-L/16, ImageNet-21k (propuesto) | ✓ | ✓ | ✓ |
| ViT-B/16, ImageNet-21k | | | ✓ |

A eso se suman las líneas base de recuperación y de clasificador + plantilla.

`tools/paired_tone.py` compara cada lesión consigo misma en los dos tonos. Calcula la exactitud del diagnóstico en piel clara y en piel oscura, aplica la prueba exacta de McNemar y obtiene el BLEU-4 por oración.

## Notas para el artículo

- La referencia con la que se calculan las métricas es el reporte anotado completo, normalizado igual que en el entrenamiento. No se usa la copia decodificada por el modelo, que viene truncada y con `<unk>`. Por eso los números pueden quedar un poco por debajo de los que imprime R2Gen, y así deben reportarse tanto el modelo como las líneas base.
- Si el modelo no supera con claridad a "clasificador + plantilla", el artículo debe decirlo e interpretarlo: en ese caso la tarea consiste sobre todo en clasificar.
- Los ejemplos cualitativos (como las Figs. 6 y 7) deben salir de `predictions.csv` del modelo propuesto, con su id.

## Control de tono contra calidad de imagen (`4_controles_tono.bat`)

Las imágenes de Brown Skin son más oscuras y además más borrosas que sus originales. Para separar los dos efectos, `tools/tone_controls.py` crea dos versiones más de cada original claro del par de prueba:

- **light_blur:** el original claro desenfocado hasta tener la misma nitidez que su versión oscura (varianza del laplaciano a 224×224). Conserva el color y pierde detalle.
- **light_color:** el original claro con la estadística de color de su versión oscura (transferencia de Reinhard en CIELAB). Conserva el detalle y cambia el tono.

`scripts/run_tone_controls.py` evalúa los modelos ya entrenados en las cuatro condiciones, sin reentrenar, y escribe `results/<experimento>/tables_tone_controls/tone_controls_summary.csv`.

Cómo leer el resultado:
- Si **light_color** cae como **dark**, la caída se debe al tono.
- Si cae **light_blur**, se debe a la pérdida de detalle.

El script solo usa corridas terminadas, es decir, las que ya tienen `test/metrics.json`.

## Experimentos A y B (`5_experimentos_AB.bat`)

- **A, línea base fuerte (`ham10000_clf`):** se ajusta el ViT-L completo como clasificador con tres salidas (diagnóstico, tipo de confirmación y localización) y se llena la plantilla del reporte con las tres predicciones (`tools/finetune_classifier.py`). Sirve para responder si el generador hace algo más que clasificar.
- **B, aumento de tono balanceado (`ham10000_toneaug`):** a cada imagen de entrenamiento, de las 7 clases por igual y con probabilidad 0.5, se le da el color de una imagen de Brown Skin de entrenamiento (transferencia de Reinhard, `modules/tone_aug.py`). Las referencias de color son solo las 89 imágenes de Brown Skin cuyos originales están en entrenamiento (`tools/build_tone_refs.py`), así que ninguna lesión de prueba se usa. Se entrenan R2Gen, el modelo propuesto y el clasificador de A.

Las dos se evalúan en la prueba de ISIC 2018, en los 83 pares de piel clara y oscura y en el control de tono.

## Reportes descriptivos (`6_reportes_descriptivos.bat`)

`tools/build_descriptive_reports.py` sustituye la plantilla corta de metadatos por un reporte descriptivo de varias frases. Usa la descripción de cada tipo de lesión de `data/lesion_descriptions_en.json`, los criterios ABC (asimetría, bordes y color), si es benigna o maligna y el estudio recomendado, junto con el diagnóstico y la localización reales de cada imagen. Del JSON no se usan el diámetro, la evolución, la localización de ejemplo ni el fototipo, porque no se pueden obtener de la imagen. Tampoco se usan las frases sobre crecimiento o cambios en el tiempo. El experimento `ham10000_desc` entrena R2Gen, ViT-L BASE y el modelo propuesto con 3 semillas y longitud máxima de 80 tokens. Se evalúa en la prueba de ISIC 2018 y en los pares de Brown Skin.
