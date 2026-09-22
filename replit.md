# Running this project on Replit

This is a Flask and NumPy solar power predictor. The Flask entry point is `app.py`.

- Start the web app with `python app.py` (the **Start application** workflow runs this command).
- The app listens on port 5000 and is available in the Replit webview.
- Dependencies are tracked in `pyproject.toml` / `uv.lock`.
- Flask templates live in `templates/`, CSS in `static/`, and the trained model parameters in `results/`.
- No external API or additional secret is needed to run the predictor.