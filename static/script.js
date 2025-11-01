const form = document.getElementById('forecastForm');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const button = form.querySelector('button');
  button.disabled = true;

  try {
    const postcode = document.getElementById('postcode').value.trim();
    const crop = document.getElementById('crop').value.trim();

    // 1) Geocode
    let res = await fetch('/api/geocode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ query: postcode }) // <-- send {query}
    });
    if (!res.ok) throw new Error('Geocoding failed');
    const geo = await res.json();

    const lat = geo.lat;
    const lon = geo.lon;
    const loc = geo.displayName;

    // 2) Submit request with flat payload {lat, lon, crop}
    res = await fetch('/api/submitRequest', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ lat, lon, crop })
    });
    if (!res.ok) throw new Error('Submit failed');
    const report = await res.json();

    const answer =
      `The data for ${loc}\n` +
      `Latitude: ${lat}, Longitude: ${lon}\n\n` +
      `${report.message}\n\n` +
      `${report.csv}`;

    document.getElementById('output').textContent = answer;

  } catch (err) {
    document.getElementById('output').textContent = 'Error: ' + err.message;
  } finally {
    button.disabled = false;
  }
});
