# Smart Attendance

A browser-based attendance application built with Flask. The web app uses a PostgreSQL database, is deployed to Render with the root `render.yaml` Blueprint, and does not include or require a desktop application.

## Project layout

```text
webapp/       Flask application, templates, and static assets
migrations/   Database schema migrations
tests/        Web application tests
web_app.py    WSGI and Flask CLI entry point
render.yaml   Render web-service Blueprint
```

## Deploy to Render

1. Push the repository to GitHub and create a Render Blueprint from the repository's `render.yaml`.
2. Set `WEB_ADMIN_USERNAME` and a unique, strong `WEB_ADMIN_PASSWORD` when prompted. Render generates `SECRET_KEY`; provide `DATABASE_URL` from Supabase.
3. Deploy. Render installs `requirements.txt`, applies migrations with `flask --app web_app:app db upgrade`, and starts Gunicorn on Render's `$PORT`.
4. Open the service URL and sign in. Camera access requires HTTPS, which Render provides.

### Connect Supabase

1. In Supabase, open **Connect** and copy the **Session pooler** connection string. Render services use IPv4, while Supabase's direct database endpoint may require IPv6; the session pooler supports IPv4. Use the session pooler on port `5432`, not the transaction pooler on port `6543`.
2. Replace `[YOUR-PASSWORD]` with your Supabase database password. Percent-encode reserved characters in the password, and ensure the URL uses `sslmode=require`.
3. Add the complete string as `DATABASE_URL` in the Render service's environment settings. Do not commit it to Git. The local `.env` uses the same variable.

The application already uses the PostgreSQL `psycopg` driver and converts `postgresql://` Supabase URLs to SQLAlchemy's `postgresql+psycopg://` form. No Supabase SDK or other database code changes are required; Render's Blueprint now prompts for the external `DATABASE_URL` instead of creating a Render PostgreSQL database.

The Blueprint still selects Render's free web-service plan for evaluation. Free web services can spin down when idle; choose an appropriate paid web-service plan for a live service. Configure Supabase backups and access controls appropriate to the sensitivity of attendance and biometric descriptor data.

The `/health` endpoint checks database connectivity. Schema changes are tracked in `migrations/` and applied before the web worker starts.

## Run locally

Use Python 3.11 or newer. Create a private `.env` from `.env.example`, generate a new `SECRET_KEY`, and set your Supabase session pooler URL and administrator credentials. `.env` is ignored by Git. In PowerShell, from the project folder:

```powershell
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
notepad .env
python -m pip install -r requirements.txt
flask --app web_app:app db upgrade
python web_app.py
```

Paste the generated value into `SECRET_KEY` in `.env`, then replace the example `DATABASE_URL` with your real Supabase Session pooler URL and set unique administrator credentials. The `flask ... db upgrade` command creates/updates the schema in the database configured by `DATABASE_URL`. `python web_app.py` starts Flask's local development server; use `localhost` for camera access. Do not expose the development server publicly; Render runs the application with Gunicorn.

## Privacy and recognition limitations

- Camera use is opt-in. Enrollment requires an explicit permission checkbox. Camera frames are sent to the server for recognition/enrollment over HTTPS, decoded and processed in memory, and not written to disk or stored in the database. The database stores face descriptors, names, student IDs, and attendance records.
- The browser requests camera access only on the live attendance page or after opening enrollment. Stop the camera or leave the page to release the device.
- Face descriptors are sensitive biometric data. This sample does not provide application-level encryption for descriptors, a consent/retention workflow, multi-user roles, or production-grade authentication. Use only with informed consent and appropriate safeguards.
- The base install uses a simple OpenCV pixel descriptor fallback. It is for demonstrating the workflow, not reliable biometric identity verification. The app does not enable liveness/anti-spoofing. Do not use its recognition result as a security decision.
