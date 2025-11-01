from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime
from dotenv import load_dotenv
from server import get_nasa_csv

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/geocode', methods=['POST'])
def geocode():
    payload = request.get_json(force=True) or {}
    query = payload.get("query")  # <-- expect "query" from client
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    r = requests.get(url, params=params, headers={"User-Agent": "durhack2025"})
    if not r.ok or not r.json():
        return jsonify({"error": "Location not found"}), 404

    res = r.json()[0]
    return jsonify({
        "query": query,
        "lat": res["lat"],
        "lon": res["lon"],
        "displayName": res["display_name"]
    })

@app.route('/api/submitRequest', methods=['POST'])
def submit_request():
    payload = request.get_json(force=True) or {}
    # Expect a flat payload: { lat, lon, crop }
    try:
        lat = float(payload["lat"])
        lon = float(payload["lon"])
        crop = payload["crop"]
    except (KeyError, ValueError):
        return jsonify({"error": "Expected JSON: { lat, lon, crop }"}), 400

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = "20200101"

    # TODO: implement this
    data_csv = get_nasa_csv(lat, lon, start_date, end_date)
    # data_csv = "date,temp,ndvi\n2020-01-01,12.3,0.41\n2020-01-08,11.7,0.44"
    print(data_csv)

    # Return a clear object
    return jsonify({
        "ok": True,
        "loc": {"lat": lat, "lon": lon},
        "crop": crop,
        "csv": data_csv,
        "message": f"Fetched NASA data for {crop} at ({lat:.5f}, {lon:.5f})"
    })

if __name__ == '__main__':
    app.run(debug=True)
