from flask import Flask, render_template, request
from ner import nlp, length_component_function

app = Flask(__name__, template_folder='templates')

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True


# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Load home screen
@app.route("/")
def home():
    return render_template("index.html")


# Display the processed text and length
@app.route("/", methods=['POST'])
def process():
    if request.method == 'POST':
        form_data = request.form.get("input_text")
        length = length_component_function(doc=form_data)
        entities = nlp(text=form_data)
        return render_template("processed.html", form_data=form_data, length=length, entities=entities)


if __name__ == "__main__":
    app.run(debug=True)
