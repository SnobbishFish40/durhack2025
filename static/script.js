const form = document.getElementById('forecastForm');
const Out = document.querySelector("#output");
const querier = document.getElementById("querier");
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const button = form.querySelector('button');
  button.disabled = true;

  try {
    const postcode = document.getElementById('postcode').value.trim();
    const crop = document.getElementById('crop').value.trim();
    const prediction = document.getElementById('prediction').value.trim();

    // 1) Geocode
    let res = await fetch('/api/geocode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ query: postcode })
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
      body: JSON.stringify({ lat, lon, crop, prediction })
    });

    if (!res.ok) throw new Error('Submit failed');
    const llm_resp = await res.json();
    const llm = llm_resp.text
    console.log("LLM:\n", llm)

    const md = llm.content?.[0]?.text ?? String(llm);

    const html = DOMPurify.sanitize(marked.parse(md));

    console.log("HTML\n", html)

    const child = document.createElement('div');
    child.className = "LLM";
    child.innerHTML = html;
    Out.insertBefore(child, querier);

    querier.classList.remove("hidden");
    Out.classList.remove("hidden");
  } catch (err) {
    Output('Error: ' + err.message, "Err")
  } finally {
    button.disabled = false;
  }
});


function Output(msg, typ) {
  const child = document.createElement('div');
  child.textContent = msg;
  child.className = typ;
  Out.insertBefore(child, querier);
}
