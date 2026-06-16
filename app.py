import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# -------------------------------------------------------------------------
# Load the DecisionTreeRegressor Model
# -------------------------------------------------------------------------
# The model uses these specific input features in order:
# ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission']
MODEL_PATH = "decision_pkl.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

# -------------------------------------------------------------------------
# Function to convert raw continuous numeric target value into an attractive category
# -------------------------------------------------------------------------
def categorize_prediction(value):
    """
    Adjust the threshold values below based on the nature of your target values 
    (e.g., Car Price ranges, Efficiency rankings, Score brackets).
    """
    if value < 10000:
        return {"category": "Low / Entry Level", "color": "#6c757d", "badge": "bg-secondary"}
    elif value < 25000:
        return {"category": "Budget / Economical", "color": "#17a2b8", "badge": "bg-info"}
    elif value < 50000:
        return {"category": "Mid-Range Comfort", "color": "#0d6efd", "badge": "bg-primary"}
    elif value < 90000:
        return {"category": "Premium Tier", "color": "#198754", "badge": "bg-success"}
    else:
        return {"category": "Luxury Elite", "color": "#ffc107", "badge": "bg-warning text-dark"}

# -------------------------------------------------------------------------
# Beautiful UI Template (Includes Structure, Layout and Responsive CSS)
# -------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Core Performance & Valuation Model</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-bg: #111827;
            --card-bg: #1f2937;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        body {
            background-color: var(--primary-bg);
            color: var(--text-main);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
            padding-bottom: 50px;
        }

        .hero-section {
            background: linear-gradient(135deg, #1e1b4b 0%, #111827 100%);
            border-bottom: 1px solid #374151;
            padding: 40px 0;
            margin-bottom: 40px;
            text-align: center;
        }

        .hero-section h1 {
            font-weight: 700;
            background: linear-gradient(to right, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .main-card {
            background-color: var(--card-bg);
            border: 1px solid #374151;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            padding: 30px;
        }

        .form-label {
            font-weight: 500;
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-bottom: 8px;
        }

        .form-control, .form-select {
            background-color: #111827;
            border: 1px solid #4b5563;
            color: var(--text-main);
            border-radius: 8px;
            padding: 10px 14px;
            transition: all 0.3s ease;
        }

        .form-control:focus, .form-select:focus {
            background-color: #111827;
            border-color: var(--accent-blue);
            color: var(--text-main);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
        }

        .btn-predict {
            background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
            border: none;
            color: white;
            font-weight: 600;
            padding: 12px;
            border-radius: 8px;
            width: 100%;
            margin-top: 15px;
            transition: transform 0.2s ease, opacity 0.2s ease;
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            opacity: 0.95;
            color: white;
        }

        .result-container {
            background-color: #111827;
            border-left: 5px solid var(--accent-blue);
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
        }

        .output-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #34d399;
        }

        .category-badge {
            font-size: 1rem;
            padding: 6px 14px;
            border-radius: 50px;
            font-weight: 600;
            display: inline-block;
            margin-top: 5px;
        }

        .error-banner {
            background-color: #7f1d1d;
            border: 1px solid #f87171;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            color: #fca5a5;
        }
    </style>
</head>
<body>

    <div class="hero-section">
        <div class="container">
            <h1>Predictive Diagnostics Dashboard</h1>
            <p class="text-muted mb-0">Powered by Machine Learning Decision Tree Architecture</p>
        </div>
    </div>

    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                
                {% if model_status == 'missing' %}
                <div class="error-banner text-center">
                    <h5>⚠️ Model Resource File Warning</h5>
                    <p class="mb-0">Could not locate <strong>decision_pkl.pkl</strong> in your working directory. Please ensure it is uploaded correctly alongside your application scripts.</p>
                </div>
                {% endif %}

                <div class="main-card">
                    <form method="POST" action="/">
                        <h4 class="mb-4 text-center" style="color: #a78bfa;">Input Target Variables</h4>
                        
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label">Make Code</label>
                                <input type="number" step="any" name="make" class="form-control" value="{{ inputs.get('make', '1') }}" required>
                            </div>
                            
                            <div class="col-md-6">
                                <label class="form-label">Model Code</label>
                                <input type="number" step="any" name="model" class="form-control" value="{{ inputs.get('model', '1') }}" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Manufacture Year</label>
                                <input type="number" step="any" name="year" class="form-control" value="{{ inputs.get('year', '2020') }}" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Engine Size (Liters)</label>
                                <input type="number" step="any" name="engine_size" class="form-control" value="{{ inputs.get('engine_size', '2.0') }}" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Current Mileage</label>
                                <input type="number" step="any" name="mileage" class="form-control" value="{{ inputs.get('mileage', '35000') }}" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Fuel Type Index</label>
                                <input type="number" step="any" name="fuel_type" class="form-control" value="{{ inputs.get('fuel_type', '1') }}" required>
                            </div>

                            <div class="col-md-12">
                                <label class="form-label">Transmission Setup Index</label>
                                <input type="number" step="any" name="transmission" class="form-control" value="{{ inputs.get('transmission', '1') }}" required>
                            </div>
                        </div>

                        <button type="submit" class="btn btn-predict" {% if model_status == 'missing' %}disabled{% endif %}>Execute Machine Learning Matrix</button>
                    </form>

                    {% if raw_prediction is not none %}
                    <div class="result-container text-center animate__animated animate__fadeIn">
                        <span class="text-muted d-block uppercase tracking-wider mb-1" style="font-size: 0.85rem; text-transform: uppercase;">Estimated Value Output</span>
                        <div class="output-value mb-2">{{ "{:,.2f}".format(raw_prediction) }}</div>
                        
                        <div class="mt-2">
                            <span class="text-muted d-block mb-1" style="font-size: 0.85rem;">Calculated Category Class:</span>
                            <span class="category-badge {{ category_data.badge }}">
                                {{ category_data.category }}
                            </span>
                        </div>
                    </div>
                    {% endif %}

                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# -------------------------------------------------------------------------
# Routing Logics
# -------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    model_status = 'active' if model is not None else 'missing'
    raw_prediction = None
    category_data = None
    inputs = {}

    if request.method == "POST":
        # Extract individual entries from the UI matrix
        inputs['make'] = request.form.get('make')
        inputs['model'] = request.form.get('model')
        inputs['year'] = request.form.get('year')
        inputs['engine_size'] = request.form.get('engine_size')
        inputs['mileage'] = request.form.get('mileage')
        inputs['fuel_type'] = request.form.get('fuel_type')
        inputs['transmission'] = request.form.get('transmission')

        if model is not None:
            try:
                # Format variables into structured array input matching pipeline configurations
                features = np.array([[
                    float(inputs['make']),
                    float(inputs['model']),
                    float(inputs['year']),
                    float(inputs['engine_size']),
                    float(inputs['mileage']),
                    float(inputs['fuel_type']),
                    float(inputs['transmission'])
                ]])
                
                # Run prediction
                prediction_output = model.predict(features)
                raw_prediction = float(prediction_output[0])
                
                # Transform continuous regression points into formatted visual category metrics
                category_data = categorize_prediction(raw_prediction)
                
            except Exception as e:
                print(f"Prediction matrix error: {e}")

    return render_template_string(
        HTML_TEMPLATE,
        model_status=model_status,
        raw_prediction=raw_prediction,
        category_data=category_data,
        inputs=inputs
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
