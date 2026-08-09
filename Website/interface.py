from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import googlemaps as gm
import os
import sys
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0,parent_dir)

from Model import RunModel as rm
from Data.Code.NearestFeature import getNearestFeature

FEATURES = ["downtown","transit", "parks", "schools"]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    incomes = pd.read_csv("..\\Data\\CSVs\\Sorted\\incomes.csv")
    community = incomes['Community Name'].str.capitalize()
    communities = community.tolist()  #add all the communities from the incomes
    
    if request.method == 'POST':
        try:
            x = []

            initial_assessment = float(request.form.get('initial_assessment'))
            x.append(initial_assessment)

            nbhood = request.form.get('nbhood').upper()
            income = int(incomes.loc[incomes['Community Name'] == nbhood, 'Median Household Income Before Tax'].item())
            x.append(income)

            lat = request.form.get("lat")
            lon = request.form.get("lon")

            if lat == '':
                return render_template('index.html', community_list = communities, python_result="Error: address not found! Please try again.")

            # gmaps = gm.Client(key='AIzaSyC_71noWB0pJGsKNuZOA49pu3GAgl4hsA0')
            coords = [float(lat), float(lon)]

            feature_dists = []
            for feature in FEATURES:
                dist = getNearestFeature(coords, feature)
                feature_dists.append(dist)

            x.append(feature_dists[0])
            x.append(feature_dists[1])

            units = int(request.form.get('units'))
            cpu = initial_assessment / units
            x.append(cpu)

            x.append(feature_dists[2])
            x.append(feature_dists[3])

            model = request.form.get("model")

            X = np.array(x)
            
            result = rm.runModel(model,X)

            return render_template('index.html', community_list = communities, python_result=result)
        except Exception as e:
            return render_template('index.html', community_list = communities, python_result="Error: one or more input(s) are empty!")
        
    return render_template('index.html', community_list = communities, python_result=result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)

