# Smart Attendance

The project now includes a browser-based attendance application for Render. The existing PyQt application remains available as a separate local desktop program; it does not share its SQLite database or biometric descriptors with the hosted service.

## Deploy to Render

1. Push this project to a Git repository and connect it in Render using **New + → Blueprint**.
2. Review `render.yaml`. It creates a Python web service and a PostgreSQL database. The blueprint currently selects Render's free database plan; free databases are for evaluation and expire, so upgrade the database plan before relying on it for long-term records.
3. During Blueprint setup, provide `WEB_ADMIN_USERNAME` and a unique, strong `WEB_ADMIN_PASSWORD`. Render generates `SECRET_KEY` and connects `DATABASE_URL` to the database.
4. Deploy. Render installs `requirements.txt`; the start command applies database migrations and starts Gunicorn bound to Render's dynamic `$PORT`.
5. Open the service URL, sign in, enroll a person, then use **Live attendance**. Camera access requires HTTPS, which Render provides for deployed services.

The `/health` route checks the database connection. Database changes are tracked under `migrations/` and applied before the web worker starts.

## Run locally

Use Python 3.11 or newer and PostgreSQL for the closest match to production. Create a private `.env` from `.env.example`, generate a new `SECRET_KEY`, and set the local database URL and unique web administrator credentials. `.env` is ignored by Git.

```powershell
python -m pip install -r requirements.txt
flask --app web_app:app db upgrade
flask --app web_app:app run
```

For local browser camera access, use `localhost`. Do not expose Flask's development server publicly; Render uses Gunicorn.

## Privacy and recognition limitations

- Camera use is opt-in. Enrollment requires an explicit permission checkbox. Camera frames are sent to the server for recognition/enrollment over HTTPS, decoded and processed in memory, and not written to disk or stored in the database. The database stores face descriptors, names, student IDs, and attendance records.
- The browser requests camera access only on the live attendance page or after opening enrollment. Stop the camera or leave the page to release the device.
- Face descriptors are sensitive biometric data. This sample does not provide application-level encryption for descriptors, a consent/retention workflow, multi-user roles, or production-grade authentication. Use only with informed consent and appropriate safeguards.
- The base install uses a simple OpenCV pixel descriptor fallback. It is for demonstrating the workflow, not reliable biometric identity verification. The app does not enable liveness/anti-spoofing. Do not use its recognition result as a security decision.
- The web app has a separate Render PostgreSQL database. It does not import `attendance_system.db` or `attendance_system_with_blink.py`'s local SQLite records. There is no automatic or implicit biometric-data migration.

## Legacy desktop app

The local PyQt application is available with the launcher from the project folder:

```powershell
python run_attendance.py
```

The launcher runs `attendance_system_with_blink.py` in the Conda environment named `attendance` and sets the project folder as its working directory. Before the first run, create a private `.env` from `.env.example` and set `ATTENDANCE_ADMIN_USERNAME` and `ATTENDANCE_ADMIN_PASSWORD`. Keep `.env` out of source control.

On a clean desktop environment, install the separate desktop dependencies first:

```powershell
conda activate attendance
python -m pip install -r requirements-desktop.txt
```

The launcher verifies and, if needed, downloads the official OpenCV Zoo YuNet and SFace CPU model files into the ignored `models/` folder. The files are checked against pinned SHA-256 digests. The one-time SFace download is about 37 MB. To download them manually, run `python download_face_models.py`.

In the desktop app, use **Add new student** to enter the student's name, roll number, and class. Either capture the student with the camera or click **Upload 4 photos** and select exactly four clear face images from different views (for example: front, left, right, and a slight up/down angle). Saving extracts and stores face descriptors; the selected source photos are not copied into the project or saved in the database. For existing students, click **Edit**, then upload four photos and save to replace the old face template.

Use **Take attendance** to open live recognition. If one face is visible but not recognized or not yet confirmed by the blink check, the app counts at most one unsuccessful attendance check per second. After three checks, **Manual attendance by roll number** becomes available. Enter the exact roll number of an already enrolled student; the database rejects unknown IDs. Recognition and manual attendance both update `Attendance_records/DD-MM-YYYY.xlsx` (for example, `20-10-2026.xlsx`), with one sheet per class and the columns Name, Roll Number, Class, and Attendance Time. Each student has one row per day using their first attendance time. Keep the workbook closed in Excel while attendance is being recorded.

Existing students remain in the database, but their old face templates are marked as legacy and cannot be compared to SFace. Upload four new photos or capture a new face enrollment before expecting automatic recognition for them. They can still be recorded through the manual roll-number fallback.

The OpenCV YuNet detector and SFace recognizer improve detection and matching across moderate pose changes using CPU inference. They cannot guarantee identification at every angle, under poor lighting/occlusion, or for extreme profile views. Test with consenting students and use manual attendance when recognition is uncertain; do not treat face matching as security-grade identity verification.

The desktop app continues to use its local SQLite database, separate from the Render-hosted web application's PostgreSQL data.
