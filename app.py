from flask import Flask
from flask_bootstrap import Bootstrap5
from frontend import frontend
from app_utils import load_dependencies

def create_app(configfile=None):
    # We are using the "Application Factory"-pattern here, which is described
    # in detail inside the Flask docs:
    # http://flask.pocoo.org/docs/patterns/appfactories/

    app = Flask(__name__)

    # We use Flask-Appconfig here, but this is not a requirement
    # AppConfig(app)

    # Install our Bootstrap extension
    bootstrap = Bootstrap5(app)

    # Our application uses blueprints as well; these go well with the
    # application factory. We already imported the blueprint, now we just need
    # to register it:
    app.register_blueprint(frontend)

    app.config['SECRET_KEY'] = 'devkey'

    app.gpt2_model = load_dependencies()
    
    return app

# # create an app instance
app = create_app()
