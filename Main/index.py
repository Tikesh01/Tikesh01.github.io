from flask import Flask, render_template, request
import os#os handles the interaction eith operating system 
app = Flask(__name__)

@app.route('/')
def interface():
    return render_template('index.html')

#Creation of a folder to store the files for clustering
os.makedirs("userFiles", exist_ok=True)#Exist_ok will check for file is created or not and prevent to create multiple time

@app.route('/uploader', methods = ["POST"])
def getFile( ):
    file = request.files["file"] #this is the method to store a file into a variable
    filePath = os.path.join("userFiles", file.filename)
    file.save(filePath)
    return "uploaded successfully"
if __name__ == "__main__":
    app.run(debug = True)