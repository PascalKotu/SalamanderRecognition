from flask import Flask, request

app = Flask(__name__)

# POST - just get the image and metadata
@app.route('/sendsalamander', methods=['POST'])
def post():
    request_data = request.form['metadata']
    print(request_data)
    imagefile = request.files.get('imagefile', '')
    imagefile.save('test.jpg')
    return "OK", 200

app.run(host='0.0.0.0')