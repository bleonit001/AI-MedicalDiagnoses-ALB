from flask import Flask, request, render_template, jsonify  
from rapidfuzz import process# type: ignore # Import jsonify
# from fuzzywuzzy import process# type: ignore # Import jsonify
import numpy as np
import pandas as pd
import pickle


# flask app
app = Flask(__name__)



# load databasedataset===================================
# Assuming the files are in the 'datasets' folder in your project directory
sym_des = pd.read_csv("/Users/bleonitshillova/Desktop/Medical-Diagnoses SHQIP/dataset/symtoms_df-sq.csv", on_bad_lines='skip')
precautions = pd.read_csv("/Users/bleonitshillova/Desktop/Medical-Diagnoses SHQIP/dataset/precautions_df-sq.csv",on_bad_lines='skip')
workout = pd.read_csv("/Users/bleonitshillova/Desktop/Medical-Diagnoses SHQIP/dataset/workout_df-sq.csv",on_bad_lines='skip')
description = pd.read_csv("/Users/bleonitshillova/Desktop/Medical-Diagnoses SHQIP/dataset/description-sq.csv",on_bad_lines='skip')
medications = pd.read_csv("/Users/bleonitshillova/Desktop/Medical-Diagnoses SHQIP/dataset/medications-sq.csv",on_bad_lines='skip')
diets = pd.read_csv("/Users/bleonitshillova/Desktop/Medical-Diagnoses SHQIP/dataset/diets-sq.csv",on_bad_lines='skip')



# load model===========================================
svc = pickle.load(open('/Users/bleonitshillova/Desktop/Medical-Diagnoses SHQIP/svc.pkl','rb'))


#============================================================
# custome and helping functions
#==========================helper funtions================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    # Normalize and fetch description
    desc = description[description['Sëmundje'].str.strip().str.lower() == dis.strip().lower()]['përshkrimi']
    desc = " ".join(desc.tolist()) if not desc.empty else "Asnjë përshkrim i disponueshëm"

    # Normalize and fetch precautions
    pre = precautions[precautions['Sëmundje'].str.strip().str.lower() == dis.strip().lower()][
        ['Masaparaprake_1', 'Masaparaprake_2', 'Masaparaprake_3', 'Masaparaprake_4']
    ]
    pre = [x for x in pre.values[0] if pd.notna(x)] if not pre.empty else ["Asnjë Masaparaprake e disponueshëm"]

    # Fetch medications
    med = medications[medications['Sëmundje'].str.strip().str.lower() == dis.strip().lower()]['mjekim']
    med = med.tolist() if not med.empty else ["Asnjë medikament i disponueshëm"]

    # Fetch diet recommendations
    die = diets[diets['Sëmundje'].str.strip().str.lower() == dis.strip().lower()]['Diet']
    die = die.tolist() if not die.empty else ["Asnjë diet rekomanduese e disponueshëm"]

    # Fetch workout suggestions
    wrkout = workout[workout['Sëmundje'].str.strip().str.lower() == dis.strip().lower()]['stërvitje']
    wrkout = wrkout.tolist() if not wrkout.empty else ["Asnjë terapi rekomanduese e disponueshëm"]

    return desc, pre, med, die, wrkout





symptoms_dict = {
    'kruarje': 0, 'Skuqja_e_lëkurës': 1, 'shpërthimet_e_lëkurës_nyjore': 2, 
    'teshtitje_e_vazhdueshme': 3, 'dridhje': 4, 'të_dridhura': 5, 'dhimbje_nyjesh': 6, 
    'dhimbje_barku': 7, 'aciditeti': 8, 'ulçera_në_gjuhë': 9, 'humbje_muskujsh': 10, 
    'të_vjella': 11, 'djegie_miktruimi': 12, 'njolla_urinimi': 13, 'lodhje': 14, 
    'shtim_peshe': 15, 'ankthi': 16, 'duart_dhe_këmbët_e_ftohta': 17, 'ndryshimet_e_humorit': 18, 
    'humbje_peshe': 19, 'shqetësim': 20, 'letargji': 21, 'arna_në_fyt': 22, 
    'Niveli_i_parregullt_i_sheqerit': 23, 'kollë': 24, 'temperaturë_e_lartë': 25, 
    'sytë_mbytur': 26, 'gulçim': 27, 'djersitje': 28, 'dehidratim': 29, 
    'dispepsi': 30, 'dhimbje_koke': 31, 'lëkura_e_kuqe': 32, 'urina_e_errët': 33, 
    'nauze': 34, 'humbja_e_oreksit': 35, 'dhimbje_pas_syve': 36, 'Dhimbja_e_shpinës': 37, 
    'kapsllëk': 38, 'diarre': 39, 'ethe_të_lehta': 40, 'urina_e_verdhë': 41, 
    'zverdhja_e_syve': 42, 'dështimi_akut_i_mëlçisë': 43, 'lëngu_mbingarkues': 44, 
    'ënjtja_e_stomakut': 45, 'nyjet_limfatikët_e_fryrë': 46, 'Sëmundje': 47, 
    'vizion_turbullt_dhe_i_shtrembëruar': 48, 'gëlbazë': 49, 'acarim_fyti': 50, 
    'skuqje_e_syve': 51, 'presioni_i_sinusit': 52, 'rrjedhje_e_hundës': 53, 
    'mbingarkesë': 54, 'dhimbje_gjoksi': 55, 'dobësi_në_gjymtyrë': 56, 'rrahje_të_shpejta_të_zemrës': 57, 
    'dhimbje_gjatë_lëvizjeve_të_zorrëve': 58, 'dhimbje_në_rajonin_anal': 59, 'jashtëqitje_e_përgjakshme': 60, 
    'acarim_në_anus': 61, 'Dhimbja_e_qafës': 62, 'marramendje': 63, 'ngërçe': 64, 
    'mavijosje': 65, 'obeziteti': 66, 'këmbë_të_fryra': 67, 'enë_të_fryra_të_gjakut': 68, 
    'Fytyra_dhe_sy_të_fryrë': 69, 'tiroidë_e_zmadhuar': 70, 'thonjtë_e_brishtë': 71, 
    'ekstremitetet_e_fryra': 72, 'urinë_e_tepruar': 73, 'kontakte_extra_martesore': 74, 
    'tharje_dhe_dridhje_buzesh': 75, 'të_folurit_të_paqartë': 76, 'dhimbje_gjuri': 77, 
    'Dhimbja_e_nyjeve_të_kofshës': 78, 'dobësi_muskulare': 79, 'qafa_e_ngurtë': 80, 
    'ënjtje_nyjesh': 81, 'Lëvizja_ngurtësi': 82, 'rrotullime_lëvizjesh': 83, 'humbja_e_ekuilibrit': 84, 
    'paqëndrueshmëri': 85, 'dobësi_e_një_anës_trupore': 86, 'humbja_e_eres': 87, 
    'shqetësimi_i_fshikëzës': 88, 'Furinë_me_erë_të_keqe': 89, 'Ndjenja_e_vazhdueshme_e_urinës': 90, 
    'kalimi_i_gazeve': 91, 'kruarje_e_brendshme': 92, 'look_toksik_(tifos)': 93, 'depresioni': 94, 
    'nervozizëm': 95, 'dhimbje_muskulore': 96, 'altered_sensorium': 97, 'njollat_e_kuqe_mbi_trup': 98, 
    'dhimbje_barku': 99, 'menstruacione_jo_normale': 100, 'arna_diskromatike': 101, 
    'lotim_nga_sytë': 102, 'oreksi_i_shtuar': 103, 'poliuria': 104, 'historia_familjare': 105, 
    'mukoide_sputum': 106, 'pështymë_e_ndryshkur': 107, 'mungesë_përqendrimi': 108, 
    'shqetësimet_vizuale': 109, 'marrja_e_transfuzionit_të_gjakut': 110, 'marrja_e_injeksioneve_josterile': 111, 
    'koma': 112, 'gjakderdhje_në_stomak': 113, 'zgjerimi_i_barkut': 114, 'historia_e_konsumimit_të_alkoolit': 115, 
    'lëngu_mbingarkues': 116, 'gjak_në_sputum': 117, 'venat_e_shqara_në_viç': 118, 'palpitacione': 119, 
    'ecje_dhe_dhimbje': 120, 'puçrrat_e_mbushura_me_qelb': 121, 'pika_te_zeza': 122, 
    'lëkundje': 123, 'lëkurë_lëkurë': 124, 'pluhuri_si_argjendi': 125, 'dhëmbëzat_e_vogla_në_thonj': 126, 
    'thonjtë_inflamator': 127, 'flluskë': 128, 'plagë_kuqe_rreth_hundës': 129, 'kore_e_verdhë': 130, 
    'prognoza': 131
}
diseases_list = {
  15:"Infeksion mykotik",
  4:"Alergji",
  16:"GERD",
  9:"Kolestaza kronike",
  14:"Reagim i drogës",
  33:"Sëmundja e ulçerës peptike",
  1:"SIDA",
  12:"Diabeti",
  17:"Gastroenteriti",
  6:"Astma bronkiale",
  23:"Hipertensioni",
  30:"Migrena",
  7:"Spondiloza e qafës së mitrës",
  32:"Paraliza (hemorragjia e trurit)",
  28:"Verdhëza",
  29:"Malaria",
  8:"Lija e dhenve",
  11:"Dengue",
  37:"Tifoja",
  40:"Hepatiti A",
  19:"Hepatiti B",
  20:"Hepatiti C",
  21:"Hepatiti D",
  22:"Hepatiti E",
  3:"Hepatiti alkoolik",
  36:"Tuberkulozi",
  10:"Ftohja e zakonshme",
  34:"Pneumonia",
  13:"Hemorroidet dimorfike (grumbullat)",
  18:"Sulmi në zemër",
  39:"Venat me variçe",
  26:"Hipotireoza",
  24:"Hipertiroidizmi",
  25:"Hipoglicemia",
  31:"Osteoartrozë",
  5:"Artriti",
  0:"(vertigo) Vertigo Pozicionale Paroymsal",
  2:"Aknet",
  38: "Infeksioni i traktit urinar",
  35:"Psoriasis",
  27:"Impetigo"
}

def match_symptom(user_input, symptoms_list):
    """
    Matches user input symptom with the closest one in the symptoms list.
    
    Args:
        user_input (str): The symptom entered by the user.
        symptoms_list (list): List of predefined symptoms.

    Returns:
        str: The best match from symptoms_list.
    """
    match, score = process.extractOne(user_input, symptoms_list)
    if score > 80:  # Only return if similarity score is above 80%
        return match
    else:
        return None



# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(132)
    normalized_symptoms = [s.lower().strip() for s in patient_symptoms]

    symptom_found = False
    for item in normalized_symptoms:
        # Perform approximate matching using rapidfuzz
        best_match = process.extractOne(item, symptoms_dict.keys(), score_cutoff=80)
        if best_match:
            matched_symptom = best_match[0]
            input_vector[symptoms_dict[matched_symptom]] = 1
            symptom_found = True
        else:
            print(f"Paralajmërim: Simptoma '{item}' nuk u gjet në fjalorin e simptomave")

    if not symptom_found:
        return "Nuk janë futur simptoma të vlefshme."

    # Debugging: Print the constructed input vector
    print("Constructed input vector:", input_vector)

    # Predict the disease index
    try:
        predicted_index = svc.predict([input_vector])[0]
    except Exception as e:
        print("Error during prediction:", str(e))
        return "Prediction error."

    return diseases_list.get(predicted_index, "Sëmundja nuk u gjet.")

# creating routes========================================




@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    try:
        if request.method == 'POST':
            # Get symptoms from the form and validate
            symptoms = request.form.get('symptoms', '').strip()
            if not symptoms or symptoms.lower() == "symptoms":
                message = "Ju lutem, futni simptoma të vlefshme, të ndara me presje."
                return render_template('index.html', message=message)

            # Process symptoms
            user_symptoms = [symptom.strip() for symptom in symptoms.split(',')]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            # Prepare the response data
            response_data = {
                "predicted_disease": predicted_disease,
                "description": dis_des,
                "precautions": precautions,
                "medications": medications,
                "my_diet": rec_diet,
                "workout": workout
            }

            # Check if the request wants JSON response
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify(response_data)

            # If not JSON, render the result in the HTML template
            return render_template('index.html', **response_data)

   
        return render_template('index.html')

    except Exception as e:
   
        app.logger.error(f"Error in /predict route: {e}")
        
       
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500




@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/symptoms')
def contact():
    return render_template("symptoms.html")


@app.route('/developer')
def developer():
    return render_template("developer.html")




if __name__ == '__main__':

    app.run(debug=True)