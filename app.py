from flask import Flask, render_template, request # type: ignore
import pandas as pd # type: ignore
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("saved_models/flood_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

FEATURES = [
    {"name": "MonsoonIntensity", "label": "🌧️ Monsoon Intensity"},
    {"name": "TopographyDrainage", "label": "🏞️ Topography Drainage"},
    {"name": "RiverManagement", "label": "🌊 River Management"},
    {"name": "Deforestation", "label": "🌳 Deforestation"},
    {"name": "Urbanization", "label": "🏙️ Urbanization"},
    {"name": "ClimateChange", "label": "🌍 Climate Change"},
    {"name": "DamsQuality", "label": "🏗️ Dams Quality"},
    {"name": "Siltation", "label": "🪨 Siltation"},
    {"name": "AgriculturalPractices", "label": "🌾 Agricultural Practices"},
    {"name": "Encroachments", "label": "🚧 Encroachments"},
    {"name": "IneffectiveDisasterPreparedness", "label": "⚠️ Disaster Preparedness"},
    {"name": "DrainageSystems", "label": "🚰 Drainage Systems"},
    {"name": "CoastalVulnerability", "label": "🌊 Coastal Vulnerability"},
    {"name": "Landslides", "label": "⛰️ Landslides"},
    {"name": "Watersheds", "label": "💧 Watersheds"},
    {"name": "DeterioratingInfrastructure", "label": "🏚️ Infrastructure Condition"},
    {"name": "PopulationScore", "label": "👥 Population Density"},
    {"name": "WetlandLoss", "label": "🌿 Wetland Loss"},
    {"name": "InadequatePlanning", "label": "📋 Urban Planning"},
    {"name": "PoliticalFactors", "label": "🏛️ Political Factors"}
]


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    values = []

    for feature in FEATURES:
        values.append(float(request.form[feature["name"]]))

    df = pd.DataFrame(
    [values],
    columns=[feature["name"] for feature in FEATURES]
)

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]

    if prediction == 1:
        result = "⚠️ Flood Likely"
    else:
        result = "✅ No Flood Expected"

    return render_template(
        "result.html",
        prediction=result
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)