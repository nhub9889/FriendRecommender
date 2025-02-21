from flask import Flask, jsonify
from flask_cors import CORS
from Training_Testing import pipeline, generate_rec

app = Flask(__name__)
CORS(app)


@app.route('/get_recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    h = pipeline(model_name='GCN', hidden_size=16)
    recommendations = generate_rec(h, user_id)
    result = [{"id": user["id"], "score": user["score"]} for user in recommendations]

    return jsonify({"recommendations": result})


@app.route('/add_friend/<int:user_id>/<int:friend_id>', methods=['POST'])
def add_friend(user_id, friend_id):
    try:
        file_edges = f'facebook/0.edges'
        with open(file_edges, 'a') as f:
            f.write(f"{user_id} {friend_id}\n")

        return jsonify({"message": "Friend added successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

