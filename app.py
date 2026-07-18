from flask import Flask, render_template, request # type: ignore
import pandas as pd # type: ignore
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("saved_models/flood_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

FEATURES = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors"
]


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    values = []

    for feature in FEATURES:
        values.append(float(request.form[feature]))

    df = pd.DataFrame([values], columns=FEATURES)

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