from flask import Flask, render_template, request
import os  #os handles the interaction eith operating system 
import pandas as pd
import numpy as np
import csv
from io import StringIO
from openpyxl import Workbook
app = Flask(__name__)

@app.route('/')
def interface():
    return render_template('index.html')

#Creation of a folder to store the files for clustering
os.makedirs("userFiles", exist_ok=True)#Exist_ok will check for file is created or not and prevent to create multiple time
@app.route('/uploader', methods = ["POST"])
def getFile( ):
    listOfExtension  = ["csv","sql","exl","json","xml","txt"] #list of possibel extantion
    file = request.files["file"] #this is the method to store a file into a variable
    filePath = os.path.join("userFiles", file.filename)
    nameOfFile = file.filename.split('.')#array of file name (n-1)th ele is extantion
    uploadedFileExtension = nameOfFile[len(nameOfFile)-1]
    if uploadedFileExtension in listOfExtension:
        file.save(filePath)
        with open(filePath,"r",encoding= "utf-8") as file: # mode "r" and encoding helps to access a file as a string
            fileContent = file.read()# Read is a method to get the content of the file
            return printer(fileContent,uploadedFileExtension,filePath)     
        # return render_template("index.html", massage=f"File uploaded succesfully")
    else:
        return render_template("index.html", error=f"only {listOfExtension} files are allowed")

def printer(fileC,type,thefile):
    #if The uploaded file is CSV
    if type == "csv":
        # rows = np.array(fileC.split("\n"))# file ka array sirf row me 1d array
        # noOfRows = rows.__len__()#toatal rows 
        # cols = np.array(rows[0].split(','))#rows me colums 
        # noOfCols = cols.__len__()# total colums
        # array = np.array(fileC.split(',')).reshape(noOfRows,noOfCols)
        # return render_template("index.html",data = f"{array}\n")
        df = pd.read_csv(thefile)
        df = df.dropna(how="all")  # Remove empty rows
        return render_template("index.html", table=df.to_html(classes="UploadedData", border=0))
        # return render_template("index.html", table = df.to_html(classes="",col_space="2em"))
    
if __name__ == "__main__":
    app.run(debug = True)