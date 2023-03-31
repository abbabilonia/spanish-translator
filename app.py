from flask import Flask, render_template, request, jsonify
from translate import Translator
import asyncio

app = Flask(__name__)

translator = Translator()
asyncio.run(translator.get_model())

@app.route('/')
def index():
    return render_template('Test.html')

@app.route('/', methods=['POST'])
def translate():
    req = request.get_json()['from_lang']
    response = ""

    if req == "undefined" or req == "":
        return jsonify(response)

    translate = asyncio.run(translator.translate(req))
    response = { 'translation' : translate.translate(req) }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)