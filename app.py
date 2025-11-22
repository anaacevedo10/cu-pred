from flask import Flask, render_template, request, send_file
import os
from parse_out_files import get_descriptors
import pandas as pd
from datetime import datetime
import joblib
import xgboost as xgb

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Ruta de archivos de entrada .log
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Ruta del Excel generado
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
app.config['LAST_EXCEL_PATH'] = None
app.config['PREDICTIONS_EXCEL_PATH'] = None
# Ruta del modelo
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'modelo_redox_xgb.json')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_results = None
    model_status = "Modelo cargado correctamente"
    download_available = False
    download_predictions_available = False

    if request.method == 'POST':
        print("request.files keys:", list(request.files.keys()))
        files = []

        if 'files' in request.files:
            files = request.files.getlist('files')
        elif 'file' in request.files:
            files = [request.files['file']]

        if not files:
            model_status = "No se encontraron archivos"
            print("No se encontraron archivos en request.files")
        else:
            saved_paths = []
            all_results = []

            for f in files:
                if f and f.filename != '':
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
                    print("Guardando en:", filepath)
                    f.save(filepath)
                    saved_paths.append(filepath)
                else:
                    print("Archivo sin nombre")

            if not saved_paths:
                model_status = "No se guardó ningún archivo válido"
                prediction_results = None
                print("No se guardó ningún archivo")
            else:
                # Entrada
                water_charge = ['O-w', -1.03457]
                ligands = {}
                errors_descriptors = []
                for fl in saved_paths:
                    ligands, errors_descriptors = get_descriptors(fl,water_charge,ligands,errors_descriptors)

                descriptores_ligandos = (
                    pd.DataFrame.from_dict(ligands, orient='index')
                    .rename_axis('LIGANDO')
                    .reset_index()
                )
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_path = os.path.join(app.config['OUTPUT_FOLDER'], current_datetime+'_descriptores.xlsx')
                descriptores_ligandos.to_excel(excel_path, index=False)
                app.config['LAST_EXCEL_PATH'] = excel_path
                download_available = True
                # Predicción - Modelo XGBoost cargado
                ligandos = descriptores_ligandos['LIGANDO'].astype(str).tolist()
                smiles_codes = descriptores_ligandos['SMILES_CODE'].astype(str).tolist()
                FILTRO = ['CHI3n','CHI1n','SOFTNESS','CHI2n','w+ ELECTRON ACCEPTOR','CHI0n','RADIO CENTRO MASA','CHEMICAL_POTENTIAL','CHI4n','w- ELECTRON DONATOR','ELECTROPHILICITY w']
                datos_filtrados = descriptores_ligandos.copy().drop(['LIGANDO','SMILES_CODE','HOMO', 'LUMO', 'GAP','ELECTROPHILICITY NET','ELECTRONEGATIVITY',"ÁTOMOS LIGANTES"]+FILTRO, axis=1, inplace=False)
                X = scaler.transform(datos_filtrados)
                y_pred = model.predict(X)
                prediction_results = [
                    f"SRP del complejo coordinado a {ligandos[i].split('/')[-1]}: {round(y_pred[i],2):.2f} V"
                    for i in range(len(y_pred))
                ]
                #
                model_status = f"Predicción completada para {len(y_pred)} registro(s)"
                #
                df_with_pred = descriptores_ligandos.copy()
                df_with_pred['prediccion_modelo (V)'] = y_pred
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                pred_excel_path = os.path.join(app.config['OUTPUT_FOLDER'], current_datetime+'_prediccion.xlsx')
                df_with_pred.to_excel(pred_excel_path, index=False)
                #
                app.config['PREDICTIONS_EXCEL_PATH'] = pred_excel_path
                download_predictions_available = True

    if app.config['LAST_EXCEL_PATH'] is not None:
        download_available = True

    return render_template(
        'index.html',
        model_status=model_status,
        prediction_results=prediction_results,
        download_available=download_available,
        download_predictions_available=download_predictions_available
    )

@app.route('/download_excel')
def download_excel():
    excel_path = app.config.get('LAST_EXCEL_PATH', None)
    if excel_path is None or not os.path.exists(excel_path):
        return "No hay ningún archivo Excel generado todavía.", 404
    
    return send_file(
        excel_path,
        as_attachment=True,
        download_name='descriptores.xlsx'
    )

@app.route('/download_predictions_excel')
def download_predictions_excel():
    pred_path = app.config.get('PREDICTIONS_EXCEL_PATH', None)
    if pred_path is None or not os.path.exists(pred_path):
        return "No hay ningún archivo con predicciones generado todavía.", 404

    return send_file(
        pred_path,
        as_attachment=True,
        download_name='descriptores_con_predicciones.xlsx'
    )

if __name__ == '__main__':
    app.run(debug=True)
