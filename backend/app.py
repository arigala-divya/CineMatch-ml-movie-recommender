from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:5174"])

ROOT = Path(__file__).parent.parent
age_group_recs = pd.read_csv(ROOT / 'data' / 'processed' / 'age_group_recommendations.csv')


def classify_age(age):
    if age <= 12:
        return "Kids (5-12)"
    elif age <= 17:
        return "Teens (13-17)"
    elif age <= 35:
        return "Young Adults (18-35)"
    else:
        return "Adults (35+)"


@app.route("/api/result", methods=["POST"])
def result():
    data = request.get_json()
    name = data.get("name", "").strip()
    age  = data.get("age")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if age is None:
        return jsonify({"error": "age is required"}), 400
    try:
        age = int(age)
    except (ValueError, TypeError):
        return jsonify({"error": "age must be a number"}), 400
    if age < 1 or age > 120:
        return jsonify({"error": "age must be between 1 and 120"}), 400

    age_group = classify_age(age)

    recs = (
        age_group_recs[age_group_recs["recommended_for_age_group"] == age_group]
        .sort_values("hybrid_score", ascending=False)
        .head(10)["title"]
        .tolist()
    )

    return jsonify({
        "segment":         age_group,
        "profile":         f"{name}, Age {age} ({age_group})",
        "recommendations": recs,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
