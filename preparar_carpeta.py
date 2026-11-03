"""Junta en UNA carpeta todo lo necesario para entrenar en otra computadora.

Uso (en la computadora donde están los datos):
    python preparar_carpeta.py
    python preparar_carpeta.py --destino D:\\proyecto_bspc      (otra ubicación)

Crea:
    proyecto_bspc\\
        r2gen-derm\\                     código
        data\\ham10000\\images\\           HAM10000_images_all, ISIC2018_Task3_Test_Images,
                                         "100 malignas", "100 benignas"
        data\\ham10000\\annotation_ham10000_full.json
        data\\ham10000\\ISIC2018_Task3_Test_GroundTruth.csv
        data\\ham10000\\HAM10000_metadata.csv
        data\\iu_xray\\                    imágenes y annotation.json
        1_preparar_datos.bat  2_prueba_rapida.bat  3_experimentos.bat
Luego se copia la carpeta proyecto_bspc completa a la otra computadora.
"""
import argparse
import os
import shutil
import sys

HOME = os.path.expanduser('~')
R2GEN2 = os.path.join(HOME, 'Documents', 'Emmanuel Proyecto-Delfín', 'R2Gen2', 'data')
BUSCAR_EN = [R2GEN2, r'C:\1. HAM10000', os.path.join(HOME, 'Downloads'), os.path.join(HOME, 'Desktop'),
             os.path.join(HOME, 'Documents'), r'C:\CASEIB2026_proyecto']
CODIGO = os.path.dirname(os.path.abspath(__file__))


def buscar(nombre, es_carpeta, minimo=0):
    """Primera coincidencia de un archivo/carpeta (con al menos `minimo` archivos si es carpeta)."""
    for raiz in BUSCAR_EN:
        if not os.path.isdir(raiz):
            continue
        for r, dirs, files in os.walk(raiz):
            if '$Recycle.Bin' in r or '__MACOSX' in r or 'proyecto_bspc' in r:
                continue
            lista = dirs if es_carpeta else files
            if nombre in lista:
                ruta = os.path.join(r, nombre)
                if not es_carpeta or len(os.listdir(ruta)) >= minimo:
                    return ruta
    return None


def copiar(origen, destino):
    if origen is None:
        return False
    if os.path.isdir(origen):
        if os.path.exists(destino):
            print('   ya existe, se omite:', destino)
            return True
        shutil.copytree(origen, destino)
    else:
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        shutil.copy2(origen, destino)
    return True


BAT1 = r'''@echo off
cd /d "%~dp0r2gen-derm"
python tools\build_brown_skin.py --ham_ann "..\data\ham10000\annotation_ham10000_full.json" --ham_metadata "..\data\ham10000\HAM10000_metadata.csv" --isic_gt "..\data\ham10000\ISIC2018_Task3_Test_GroundTruth.csv" --image_root "..\data\ham10000\images" --bs_dirs "100 malignas" "100 benignas" --out_dir "..\data\brown_skin_v2"
pause
'''
BAT2 = r'''@echo off
REM Prueba corta: 1 epoca, 1 semilla. Solo para confirmar que todo corre. Borra results\prueba al terminar.
cd /d "%~dp0r2gen-derm"
python main_train.py --image_dir "..\data\ham10000\images" --ann_path "..\data\brown_skin_v2\ham10000_lesion.json" --dataset_name ham10000 --visual_extractor vit_large_patch16_224.augreg_in21k --variant rm_mcln --epochs 1 --batch_size 16 --save_dir results\prueba --record_dir results\prueba --seed 0
python main_test.py --image_dir "..\data\ham10000\images" --ann_path "..\data\brown_skin_v2\bs_paired_test.json" --dataset_name ham10000 --visual_extractor vit_large_patch16_224.augreg_in21k --variant rm_mcln --batch_size 16 --load results\prueba\model_best.pth --save_dir results\prueba\test_bs_pairs
pause
'''
BAT3 = r'''@echo off
REM Experimentos completos. Tardan dias; se pueden interrumpir y volver a lanzar:
REM los modelos ya entrenados (model_best.pth) no se repiten.
cd /d "%~dp0r2gen-derm"
python scripts\run_experiments.py ham10000    --image_dir "..\data\ham10000\images" --ann_dir "..\data\brown_skin_v2"
python scripts\run_experiments.py ham10000_bs --image_dir "..\data\ham10000\images" --ann_dir "..\data\brown_skin_v2"
python scripts\run_experiments.py iu_xray     --image_dir "..\data\iu_xray\images" --iu_ann "..\data\iu_xray\annotation.json"
pause
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--destino', default=r'C:\proyecto_bspc')
    args = ap.parse_args()
    D = args.destino
    print('Buscando archivos (puede tardar un par de minutos)...')

    piezas = [
        ('HAM10000_images_all', True, 10000, os.path.join(D, 'data', 'ham10000', 'images', 'HAM10000_images_all')),
        ('ISIC2018_Task3_Test_Images', True, 1500, os.path.join(D, 'data', 'ham10000', 'images', 'ISIC2018_Task3_Test_Images')),
        ('100 malignas', True, 50, os.path.join(D, 'data', 'ham10000', 'images', '100 malignas')),
        ('100 benignas', True, 50, os.path.join(D, 'data', 'ham10000', 'images', '100 benignas')),
        ('annotation_ham10000_full.json', False, 0, os.path.join(D, 'data', 'ham10000', 'annotation_ham10000_full.json')),
        ('ISIC2018_Task3_Test_GroundTruth.csv', False, 0, os.path.join(D, 'data', 'ham10000', 'ISIC2018_Task3_Test_GroundTruth.csv')),
        ('HAM10000_metadata.csv', False, 0, os.path.join(D, 'data', 'ham10000', 'HAM10000_metadata.csv')),
    ]
    faltan = []
    for nombre, carpeta, minimo, destino in piezas:
        origen = buscar(nombre, carpeta, minimo)
        print(' {:38s} <- {}'.format(nombre, origen or 'NO ENCONTRADO'))
        if not copiar(origen, destino):
            faltan.append(nombre)

    # IU X-Ray: carpeta iu_xray con annotation.json e images
    iu = None
    for raiz in BUSCAR_EN:
        cand = os.path.join(raiz, 'iu_xray')
        if os.path.isfile(os.path.join(cand, 'annotation.json')) and os.path.isdir(os.path.join(cand, 'images')):
            iu = cand
            break
    print(' {:38s} <- {}'.format('iu_xray (images + annotation.json)', iu or 'NO ENCONTRADO'))
    if not copiar(iu, os.path.join(D, 'data', 'iu_xray')):
        faltan.append('iu_xray')

    # código
    copiar(CODIGO, os.path.join(D, 'r2gen-derm'))
    for nombre, texto in (('1_preparar_datos.bat', BAT1), ('2_prueba_rapida.bat', BAT2), ('3_experimentos.bat', BAT3)):
        with open(os.path.join(D, nombre), 'w', encoding='utf-8') as f:
            f.write(texto)

    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(D) for f in fs) / 1e9
    print('\nCarpeta lista: {} ({:.1f} GB)'.format(D, total))
    if faltan:
        print('FALTAN: {}. Cópialos a mano en la ruta indicada arriba.'.format(', '.join(faltan)))
        sys.exit(1)
    print('Copia esa carpeta completa a la otra computadora y ejecuta, en orden,')
    print('1_preparar_datos.bat, 2_prueba_rapida.bat y 3_experimentos.bat.')


if __name__ == '__main__':
    main()
