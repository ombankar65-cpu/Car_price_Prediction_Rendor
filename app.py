import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# -------------------------------------------------------------------------
# Load the DecisionTreeRegressor Model
# -------------------------------------------------------------------------
MODEL_PATH = "decision_pkl.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

# -------------------------------------------------------------------------
# Categorization Logic for Continuous Numeric Target Values
# -------------------------------------------------------------------------
def categorize_prediction(value):
    """
    Converts the raw numeric regression value into an attractive category.
    Adjust these thresholds based on your actual dataset ranges.
    """
    if value < 10000:
        return {"category": "Economy / Entry", "class": "badge-economy"}
    elif value < 25000:
        return {"category": "Standard / Value", "class": "badge-standard"}
    elif value < 50000:
        return {"category": "Mid-Range / Comfort", "class": "badge-midrange"}
    elif value < 90000:
        return {"category": "Premium / Executive", "class": "badge-premium"}
    else:
        return {"category": "Luxury / Flagship", "class": "badge-luxury"}

# -------------------------------------------------------------------------
# Sleek, Cool Tech UI Template (Combined Structure and Internal Style)
# -------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Analytical Engine</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg-gradient-start: #0f172a; /* Cool slate black */
            --bg-gradient-end: #1e293b;   /* Slate blue */
            --panel-bg: rgba(30, 41, 59, 0.7); 
            --accent-ice: #38bdf8;        /* Vivid ice blue */
            --accent-mint: #34d399;       /* Fresh mint green */
            --accent-cool: #818cf8;       /* Lavender slate blue */
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --input-border: #334155;
        }

        body {
            background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            padding: 40px 0;
        }

        .container {
            max-width: 1100px;
        }

        .dashboard-wrapper {
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            overflow: hidden;
        }

        /* Form Column Section */
        .form-side {
            padding: 45px;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        .brand-title {
            font-weight: 700;
            font-size: 1.75rem;
            letter-spacing: -0.025em;
            background: linear-gradient(to right, #ffffff, var(--accent-ice));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }

        .brand-subtitle {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 35px;
        }

        .form-label {
            font-weight: 500;
            font-size: 0.825rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }

        .form-control {
            background-color: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--input-border);
            color: var(--text-primary);
            border-radius: 12px;
            padding: 12px 16px;
            font-size: 0.95rem;
            transition: all 0.2s ease-in-out;
        }

        .form-control:focus {
            background-color: rgba(15, 23, 42, 0.8);
            border-color: var(--accent-ice);
            box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.15);
            color: var(--text-primary);
        }

        .btn-submit {
            background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
            border: none;
            color: white;
            font-weight: 600;
            font-size: 1rem;
            padding: 14px;
            border-radius: 12px;
            width: 100%;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
            transition: all 0.2s ease;
        }

        .btn-submit:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
            opacity: 0.95;
        }

        /* Results / Showcase Side Column */
        .display-side {
            padding: 45px;
            background: rgba(15, 23, 42, 0.4);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }

        .placeholder-text {
            color: var(--text-secondary);
            font-style: italic;
            font-size: 0.95rem;
            max-width: 260px;
        }

        .metric-card {
            width: 100%;
        }

        .metric-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--accent-ice);
            font-weight: 600;
            margin-bottom: 10px;
        }

        .metric-value {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: var(--text-primary);
            line-height: 1.1;
            margin-bottom: 25px;
        }

        .category-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-bottom: 8px;
        }

        /* Dynamic Cool Category Badges */
        .custom-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 100px;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
        }

        .badge-economy { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }
        .badge-standard { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
        .badge-midrange { background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }
        .badge-premium { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
        .badge-luxury { background: rgba(251, 113, 133, 0.15); color: #fb7185; border: 1px solid rgba(251, 113, 133, 0.3); }

        .error-banner {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #f87171;
            border-radius: 12px;
            padding: 12px 20px;
            margin-bottom: 25px;
            font-size: 0.9rem;
        }

        @media (max-width: 991px) {
            .form-side { border-right: none; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
            body { padding: 20px 0; }
        }
    </style>
</head>
<body>

    <div class="container">
        <div class="dashboard-wrapper">
            <div class="row g-0">
                
                <div class="col-lg-7 form-side">
                    <div class="brand-title">Analytical Core Engine</div>
                    <div class="brand-subtitle">Decision Tree Regression Model Interface</div>

                    {% if model_status == 'missing' %}
                    <div class="error-banner">
                        <strong>Missing Resource:</strong> The deployment file <code>decision_pkl.pkl</code> was not detected in the root directory. Please upload the file to run inferences.
                    </div>
                    {% endif %}

                    <form method="POST" action="/">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label">Make Identification</label>
                                <input type="number" step="any" name="make" class="form-control" value="{{ inputs.get('make', '1') }}" required>
                            </div>
                            
                            <div class="col-md-6">
                                <label class="form-label">Model Designation</label>
                                <input type="number" step="any" name="model" class="form-control" value="{{ inputs.get('model', '1') }}" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Production Year</label>
                                <input type="number" step="any" name="year" class="form-control" value="{{ inputs.get('year', '2022') }}" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Engine Capacity (L)</label>
                                <input type="number" step="any" name="engine_size" class="form-control" value="{{ inputs.get('engine_size', '2.5') }}" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Odometer (Mileage)</label>
                                <input type="number" step="any" name="mileage" class="form-control" value="{{ inputs.get('mileage', '18000') }}" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label">Fuel Architecture</label>
                                <input type="number" step="any" name="fuel_type" class="form-control" value="{{ inputs.get('fuel_type', '2') }}" required>
                            </div>

                            <div class="col-md-12">
                                <label class="form-label">Transmission Vector</label>
                                <input type="number" step="any" name="transmission" class="form-control" value="{{ inputs.get('transmission', '1') }}" required>
                            </div>
                        </div>

                        <button type="submit" class="btn-submit" {% if model_status == 'missing' %}disabled{% endif %}>
                            Evaluate Matrix Parameters
                        </button>
                    </form>
                </div>

                <div class="col-lg-5 display-side">
                    {% if raw_prediction is not none %}
                    <div class="metric-card">
                        <div class="metric-title">Predicted Metric Yield</div>
                        <div class="metric-value">{{ "{:,.2f}".format(raw_prediction) }}</div>
                        
                        <div class="category-label">Identified Class Bandwidth</div>
                        <div class="custom-badge {{ category_data.class }}">
                            {{ category_data.category }}
                        </div>
                    </div>
                    {% else %}
                    <div class="placeholder-text">
                        Fill out the parameters on the left and submit to view prediction metrics and classified bands.
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
# Application Operational Control Logics
# -------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    model_status = 'active' if model is not None else 'missing'
    raw_prediction = None
    category_data = None
    inputs = {}

    if request.method == "POST":
        inputs['make'] = request.form.get('make')
        inputs['model'] = request.form.get('model')
        inputs['year'] = request.form.get('year')
        inputs['engine_size'] = request.form.get('engine_size')
        inputs['mileage'] = request.form.get('mileage')
        inputs['fuel_type'] = request.form.get('fuel_type')
        inputs['transmission'] = request.form.get('transmission')

        if model is not None:
            try:
                # Array pipeline packaging sequence matching feature assignments
                features = np.array([[
                    float(inputs['make']),
                    float(inputs['model']),
                    float(inputs['year']),
                    float(inputs['engine_size']),
                    float(inputs['mileage']),
                    float(inputs['fuel_type']),
                    float(inputs['transmission'])
                ]])
                
                prediction_output = model.predict(features)
                raw_prediction = float(prediction_output[0])
                category_data = categorize_prediction(raw_prediction)
                
            except Exception as e:
                print(f"Operational execution evaluation failure: {e}")

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
