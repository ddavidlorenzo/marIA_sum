# This contains our frontend; since it is a bit messy to use the @app.route
# decorator style when using application factories, all of our routes are
# inside blueprints. This is the front-facing blueprint.
#
# You can find out more about blueprints at
# http://flask.pocoo.org/docs/blueprints/

from flask import Blueprint, render_template, flash, redirect, url_for, request, current_app, abort
import pandas as pd
import json
from app_utils import load_dependencies, get_model_paths

frontend = Blueprint('frontend', __name__)

@frontend.route('/')
def index():
    return render_template('index.html')


@frontend.route("/summarize/")
def summarize():
    models_data = get_model_paths()
    model_ids = list(models_data.keys())
    return render_template("summarization.html", models=model_ids)


@frontend.route("/compute_summary/", methods=['POST'])
@frontend.route("/compute_summary", methods=['POST'])
def compute_summary():
    # try:
    summary_length = int(request.form['summary_length'])
    article = request.form['article']
    # Get proper model id
    # model_id = '/'.join(request.form['selectedModel'].split('_')[:2])
    model_id = request.form['selectedModel'].strip()
    if hasattr(current_app, "gpt2_model"):
        if current_app.gpt2_model.attrs_to_str() == model_id:
            generated_summary = current_app.gpt2_model.generate_sample(article, max_length=summary_length)
            # generated_summary = current_app.gpt2_model.generate_summary(article, max_length=summary_length)
        else:
            generated_summary = load_dependencies(checkpoint=model_id).generate_sample(article, max_length=summary_length)
    else:
        generated_summary = "dummy summary"
    return generated_summary
    # except Exception:
    #     print('An exception has ocurred')
    #     abort(500)

@frontend.route('/models')
@frontend.route('/models/<model_id>')
def models(model_id=None):
    models_data = get_model_paths()
    model_ids = list(models_data.keys())

    if model_id:
        return render_template('models.html', models=model_ids, model_id=model_id)
    else:
        return render_template('models.html', models=model_ids)



@frontend.route('/model_card/<model_id>')
def model_card(model_id, models_data=None):
    TO_HTML_TABLE_CLASSES = "table table-bordered table-striped table-hover"
    models_data = models_data or get_model_paths()
    model = models_data[model_id]
    rouge = pd.read_csv(model['rouge'], index_col=0).to_html(classes=[TO_HTML_TABLE_CLASSES], header="true")
    train_stats_df = pd.read_csv(model['train_stats'])
    # Fix encoding issues
    # train_stats_df['Perplexity'] = train_stats_df['Perplexity'].map(lambda x: x.replace('tensor(','').replace(')', '')).astype(float)
    train_stats = train_stats_df.set_index('epoch').to_html(classes=[TO_HTML_TABLE_CLASSES], header="true")
    with open(model['config'], 'r') as f:
        config = json.load(f)
    return dict(
        train_stats=train_stats, 
        rouge=rouge,
        config=config,
        plot_data = train_stats_df.to_dict(orient="list")
        )
