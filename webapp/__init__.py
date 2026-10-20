"""Flask web application for browser-based smart attendance."""

import os
from pathlib import Path
from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, session, url_for
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()
migrate = Migrate()
csrf = CSRFProtect()


def create_app(test_config=None):
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    database_url = os.environ.get("DATABASE_URL")
    if database_url and database_url.startswith(("postgres://", "postgresql://")):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY"),
        SQLALCHEMY_DATABASE_URI=database_url or "sqlite:///attendance.sqlite3",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        MAX_CONTENT_LENGTH=4 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")),
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        WTF_CSRF_TIME_LIMIT=3600,
        APP_TITLE=os.environ.get("APP_TITLE", "Smart Attendance"),
        WEB_ADMIN_USERNAME=os.environ.get("WEB_ADMIN_USERNAME", ""),
        WEB_ADMIN_PASSWORD=os.environ.get("WEB_ADMIN_PASSWORD", ""),
    )
    if test_config:
        app.config.update(test_config)

    if not app.config["SECRET_KEY"]:
        raise RuntimeError("SECRET_KEY must be set to a strong, random value.")
    if not app.config["WEB_ADMIN_USERNAME"] or not app.config["WEB_ADMIN_PASSWORD"]:
        raise RuntimeError("WEB_ADMIN_USERNAME and WEB_ADMIN_PASSWORD must be configured.")

    os.makedirs(app.instance_path, exist_ok=True)
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)

    from . import models  # noqa: F401
    from .routes import api, auth, main

    app.register_blueprint(auth)
    app.register_blueprint(main)
    app.register_blueprint(api, url_prefix="/api")

    @app.before_request
    def require_login():
        endpoint = request.endpoint or ""
        if endpoint in {"auth.login", "main.health", "static"} or endpoint.startswith("static"):
            return None
        if not session.get("authenticated"):
            if request.path.startswith("/api/"):
                return jsonify(error="Please sign in again."), 401
            return redirect(url_for("auth.login", next=request.path))
        return None

    @app.errorhandler(413)
    def upload_too_large(_error):
        if request.path.startswith("/api/"):
            return jsonify(error="The uploaded camera data is too large. Try again with the camera preview."), 413
        return "Request too large.", 413

    @app.after_request
    def set_security_headers(response):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; media-src 'self' blob:; object-src 'none'; "
            "base-uri 'self'; frame-ancestors 'none'; form-action 'self'",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=()")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if request.endpoint != "static":
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.cli.command("init-db")
    def init_db_command():
        """Create the schema for a new local development database."""
        with app.app_context():
            db.create_all()
        print("Database tables created.")

    return app
