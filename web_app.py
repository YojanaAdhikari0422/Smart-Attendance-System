"""WSGI entry point for the Render-hosted attendance application."""

from webapp import create_app


app = create_app()
