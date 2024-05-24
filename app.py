from flask import Flask, render_template, request, jsonify
import chat

# Initialize the Flask application
app = Flask(__name__)

# Define the route for the main page
@app.route('/')
def index():
    # Render the chat.html template
    return render_template('chat.html')

# Define the route to handle chat messages
@app.route('/chat', methods=['POST'])
def chat_route():
    # Get the user's message from the request
    user_message = request.json.get('message')
    # Use the chat module to get a response to the user's message
    bot_response = chat.chat(user_message)
    # Return the response as a JSON object
    return jsonify(response=bot_response)

# Run the application in debug mode
if __name__ == '__main__':
    app.run(debug=True)
