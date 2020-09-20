# Deployment of the ML model via Flask

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

Tree = Flask(__name__)
Treemodel = pickle.load(open('DecisionTree.pkl', 'rb'))

@Tree.route('/')
def htemplate():
    return render_template('WebAppModel.html')

@Tree.route('/tree',methods=['POST'])
def tree():
   
    input_values = [float(x) for x in request.form.values()]
    final_values = [np.array(input_values)]
    predicted_values = Treemodel.predict(final_values)

    result_values = predicted_values[0]

    return render_template('WebAppModel.html', generated_text = 
                           'Expected species is {}'.format(result_values ))


@Tree.route('/tree_api',methods=['POST'])

def tree_api():
    '''
    For direct API calls trought request
    '''
    data_values = request.get_json(force=True)
    predicted_values = Treemodel.predict([np.array(list(data_values.values()))])

    result_values  = predicted_values[0]
    return jsonify(result_values )

if __name__ == "__main__":
    Tree.run(debug=True)