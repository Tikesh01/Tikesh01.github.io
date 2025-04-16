from flask import Flask, render_template, request
import os  #os handles the interaction with operating system 
import pandas as pd
import numpy as np
import csv
from io import StringIO
from jinja2 import Environment
app = Flask(__name__)
env = Environment(autoescape=True)
MainDF = pd.DataFrame()#This is created to share the files content one to another function
@app.route('/')
def interface():
    return render_template('index.html')

#Creation of a folder to store the files for clustering
os.makedirs("userFiles", exist_ok=True)#Exist_ok will check for file is created or not and prevent to create multiple time
@app.route('/uploader', methods = ["POST"])
def getFile():
    listOfExtension = ["csv","xsl","sql","sqlquery","xlsx","json","txt"] #list of possibel extantion
    file = request.files["file"] #this is the method to store a file into a variable
    filePath = os.path.join("userFiles", file.filename)
    nameOfFile = file.filename.split('.')#array of file name (n-1)th ele is extantion
    uploadedFileExtension = nameOfFile[len(nameOfFile)-1]
    if uploadedFileExtension in listOfExtension:
        file.save(filePath)
        # with open(filePath,"r",encoding= "utf-8") as file: # mode "r" and encoding helps to access a file as a string
        #     fileContent = file.read()# Read is a method to get the content of the file
        return printTable(uploadedFileExtension,filePath)
    else:
        return render_template("index.html", error=f"only {listOfExtension} files are allowed")

def printTable(type,filePath):
     #if The uploaded file is CSV
    if type == "csv":
        df = pd.read_csv(filePath)
        df = df.dropna(how="all")  # Remove empty rows
        MainDF = df
        return render_template("index.html", table=df.to_html(classes="UploadedData", border=0))
    if type == "xls" or "xlsx":
        pass
    if type == "sql":
        pass
    if type == "json":
        pass
    if type == "txt":
        pass
    
# Now time to cluster the file(data)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
@app.route("/clusteringStart", methods=["POST"])
def dataClustering():#FileToCluster is DF which is created to print through html table
    #Process of creating appropriate DF to apply KMeans
    noOfClusters = int(request.form.get("noOfCluster"))
    fileToClusterNumaric = MainDF.select_dtypes(include=["float64", "int64"])# to Change the string type columns to int or float because KMeans() needs int or float type colum to mesure distance
    scaler = StandardScaler()
    scaled_data = (scaler.fit_transform((fileToClusterNumaric)))
    
    # Process of applying kMeans
    Kmeans = KMeans(n_clusters = noOfClusters, random_state=42)
    Kmeans_lables = Kmeans.fit(scaled_data)
    # hence means returns a array So need to change into Dataframe
    clusteredDF = pd.DataFrame(Kmeans_lables)

    return render_template("index.html", clusteredData = clusteredDF.to_html(classes="UploadedclusteredData", border=0))
 
def downLoadFile():
    pass
    
if __name__ == "__main__":
    app.run(debug = True)