from flask import Flask, render_template, request, jsonify
from chat import chat 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat(message): # Adjust this line to use your chatbot logic
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True)
