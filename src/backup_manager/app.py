from flask import Flask, jsonify


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    return app
