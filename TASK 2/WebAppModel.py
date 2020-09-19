# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 01:51:09 2020

@author: Simran Agrawal
"""

# Deploy our Model Using Flask

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

WebApp = Flask(__name__)
LinearModel = pickle.load(open('LinearModel.pkl', 'rb'))


#Routing our on /Model with the html template.
@WebApp.route('/')
def Htemplate():
    return render_template('WebAppModel.html')

@WebApp.route('/Model',methods=['POST'])
def Model():   
    try:
        #get number of hours of values from form
        get_input_values = [float(x) for x in request.form.values()] 
        get_input_array  = [np.array(get_input_values)]
        predicted_values = LinearModel.predict(get_input_array)
        
    except:
        
        return render_template('WebAppModel.html', generated_text=
                               "Input Hours should in the range of 0 to 24. Hours Cannot be an Alphabet or any other Special Character")
                                   
    else:
        result_values = round(predicted_values[0], 2)
    
    
        
        if get_input_values[0] <= 0 or get_input_values[0] > 24:
            return render_template('WebAppModel.html' , generated_text="Input Hours should in the range of 0 to 24.")
                
        if result_values > 100:    
            return render_template('WebAppModel.html', generated_text="When student studies {} hours in a day, the score should be 100%".format(str(get_input_values[0])))  
       
        else:
            return render_template('WebAppModel.html', generated_text="When student studies {} hours in a day, the score should be {} %".format(str(get_input_values[0]),result_values))                              
                                   

@WebApp.route('/Model_api',methods=['POST'])

def Model_api():
    data = request.get_json(force=True)
    predicted_values = LinearModel.predict([np.array(list(data.values()))])
    result_values =  predicted_values[0]
    return jsonify(result_values)

if __name__ == "__main__":
    WebApp.run(debug=True)
    