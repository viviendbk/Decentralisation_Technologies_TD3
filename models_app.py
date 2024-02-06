from flask import Flask, request, jsonify
import numpy as np
from joblib import load

app = Flask(__name__)

# We load the models using joblib 
models = {
    'linear_regressor': load('linear_regressor.joblib'),
    'decision_tree': load('decision_tree.joblib'),
    'random_forest': load('random_forest.joblib'),
    'xgboost': load('xgboost.joblib')
}

# Next we define the route for the prediction which will be a GET request called /predict
@app.route('/predict/<model_name>', methods=['GET'])
def predict(model_name):
    # We must first check that the model exists
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    # Then we retrieve the model
    model = models[model_name]
    
    # This retrieves the features from the query string
    features_string = request.args.get('features')
    
    # Then we convert the features to a numpy array so we can give it to the model
    features = np.array([float(x) for x in features_string.split(',')]).reshape(1, -1)
    
    # Finally we make the prediction using the model
    prediction = np.exp(model.predict(features))
    
    # In the end we return the prediction as a JSON object
    return jsonify({
        'operation status': 'success',
        'prediction': prediction.tolist(),  # Convert numpy array to list for JSON serialization
        'message': 'Prediction completed successfully'
    })

if __name__ == '__main__':
    app.run(debug=True)
