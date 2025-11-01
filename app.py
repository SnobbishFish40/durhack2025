from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime, timedelta
from server.py import get_nasa_csv

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the front page."""
    return render_template('index.html')

@app.route('/api/echo', methods=['POST'])
def echo():
    """Simple demo API route."""
    data = request.get_json()
    text = data.get('text', '')
    return jsonify({"reply": f"You said: {text}"})

@app.route('/api/geocode', methods=['POST'])
def geocode():
    """Example geocoding endpoint (uses Nominatim)."""
    query = request.get_json().get("query")
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
        "display_name": res["display_name"]
    })

@app.route('/api/submitRequest', methods['POST'])
def submitRequest():
    longitude = request.form['lon']
    latitude = request.form['lat']
    cropType = request.form['crop']
    
    end_date = datetime.now()
    start_date = '20200101' 

    data_csv = get_nasa_csv(latitude, longitude, start_date, end_date)

    # Here pass data to main analysis function which will return a processed csv
    # Then pass to request_llm_analysis and finally render the result


    print(data_csv)

if __name__ == '__main__':
    app.run(debug=True)
