from flask import Flask, render_template, request
import os  #os handles the interaction with operating system 
import pandas as pd
import numpy as np
from jinja2 import Environment
from qiskit import QuantumCircuit

app = Flask(__name__)
env = Environment(autoescape=True)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

@app.route('/')
def interface():
    return render_template('index.html')

#Creation of a folder to store the files for clustering
os.makedirs("userFiles", exist_ok=True)#Exist_ok will check for file is created or not and prevent to create multiple time
@app.route('/uploader', methods = ["POST"])
def getFile():
    print("AJAX detected:", request.headers.get("X-Requested-With"))
    listOfExtension = ["csv","xsl","sql","sqlquery","xlsx","json","txt"] #list of possibel extantion
    file = request.files["file"] #this is the method to store a file into a variable
    filePath = os.path.join("userFiles", file.filename)
    app.config['path'] = filePath
    nameOfFile = file.filename.split('.')#array of file name (n-1)th ele is extantion
    uploadedFileExtension = nameOfFile[len(nameOfFile)-1]
    if uploadedFileExtension in listOfExtension:
        file.save(filePath)
        return detectType(uploadedFileExtension,filePath,file.filename)#this fuction will detect the perticular type of file than detect the column contianing date,datetime or year 
    else:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            error_html = render_template("partials/type_error.html", error=f"only {listOfExtension} files are allowed")
            return {"error": error_html}
        else:
            return render_template("index.html", error=f"only {listOfExtension} files are allowed")


def detectType(type,filePath,fileName):
     #if The uploaded file is CSV
    if type == "csv":
        df = pd.read_csv(filePath)
        os.remove(filePath)
        app.config['df'] =  df.dropna(how="all")  # Remove empty rows
        columns = df.columns
        isDateTimeCol = detect_datetime_columns(df)
        arr = isDateTimeCol.values()
        if 1 in  arr or 2 in arr or 3 in arr:#if has any datetime column it will render the form to ask cluster according to date time
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return {
                        "table": render_template("partials/data_table.html", table=df.head(50).to_html(classes="UploadedData", border=0), noOfRows=len(df.index), noOfColumns=len(columns), fileName=fileName),
                        "note": render_template("partials/date_question.html", note="dateTime", noOfRows=len(df.index), noOfColumns=len(columns)),
                        "quote": ""
                    }
        else:#otherWise ask for preferemce only
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return {
                        "table": render_template("partials/data_table.html",table=df.head(50).to_html(classes="UploadedData", border=0), noOfRows=len(df.index), noOfColumns=len(columns), fileName =fileName),
                        "note": "",
                        "quote": render_template("partials/priority_form.html", quote="NodateTimeOnlyPriority", columns=columns)
                }
            else:
                return render_template("index.html",table=df.head(50).to_html(classes="UploadedData", border=0), noOfRows=len(df.index), noOfColumns=len(columns))
            
    if type == "xls" or "xlsx":
        pass
    if type == "sql":
        pass
    if type == "json":
        pass
    if type == "txt":
        pass
#================================================/////===========================================
#================================================/////===========================================
@app.route("/askUserClustAccDateTime", methods=["POST"])
def check():
    value = int(request.form.get("useDateTime"))
    df = app.config['df']
    columns = df.columns
    app.config['useDateTime'] = value
    if value in [0,1,2]: #Return yes or no to cluster 
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return {#It will render the form to ask preference
                "quote": render_template("partials/priority_form.html", quote="Do you want Heirarchial clustering", columns=columns),
                "table": render_template("partials/data_table.html", table=(df.head(100)).to_html(classes="UploadedData", border=0), noOfColumns=len(columns), noOfRows=len(df.index)),
                "note": ""
            }
        else:
            return render_template("index.html",table=df.head(100).to_html(classes="UploadedData", border=0), noOfColumns=len(columns), noOfRows=len(df.index), quote="Do you want Heirarchial clustering", columns=columns)

#================================================/////===========================================
app.config['useDateTime'] = 0#if No date time column detected

from pandas.api.types import is_datetime64_any_dtype
@app.route('/start', methods=["POST"])
def uploadFileToClust():
    mainDf = app.config['df']#mianDf is the data frame containing the whole content of file
    columns = np.array(mainDf.columns)#to know the number of colums in dataframe(csv) file
    noOfColumns = len(columns)
    noOfRows = len(mainDf.index)
    print(f"Total columns = {noOfColumns}")
    print(f"Total rows = {noOfRows}")
    value = app.config['useDateTime']
    priorities = {i: None for i in columns}
    for i in columns:
        priorities[i] = request.form.get(f'{i}')
    priorities = {key: value for key,value in priorities.items() if value}
    priorities = dict(sorted(priorities.items()))
    print(f"priorities = {priorities}")
    if value != 0:#IF IT is being clustered according o date time 
        sortAccordingToDateTime = detect_datetime_columns(mainDf)
        print(sortAccordingToDateTime)
        for col in mainDf.columns:
            if sortAccordingToDateTime[col] in [1,2,3]:
                if value == 1:
                    mainDf = clusterDateTimeCol(mainDf,col,sortAccordingToDateTime[col],True) 
                if value == 2:
                    mainDf = clusterDateTimeCol(mainDf,col,sortAccordingToDateTime[col],False) 
            else:
                continue
    else: #if there i no column of datetime or selected not to clust acc to dt
        mainDf=pd.DataFrame(data = np.arange(0,100).reshape(20,5))
    
    arrOfGroupNames = mainDf.index.get_level_values(0).unique()# this variable iS containing the group names
# ------------------------------/////------------------
    if len(arrOfGroupNames) !=0:
        smallDf = mainDf.groupby(level=0).head(20)
    else:
        smallDf = mainDf.head(50)
            
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return {
                "table": render_template("partials/data_table.html", table=app.config['df'].head(50).to_html(classes="UploadedData", border=0), noOfColumns=noOfColumns, noOfRows=noOfRows),
                "clustered": render_template("partials/clustered_data.html", clusteredData=smallDf.to_html(classes="UploadedClusteredData", border=0), noOfColumns=noOfColumns, noOfRows=noOfRows),
                "quote": render_template("partials/priority_form.html", quote="Do you want Heirarchial clustering", columns=columns)
            }     
    else:
        return render_template("index.html", table=app.config['df'].head(50).to_html(classes="UploadedData", border=0), clusteredData= smallDf.to_html(classes="UploadedClusteredData", border=0), quote="Do you want Heirarchial clustering", noOfRows=noOfRows, noOfColumns=noOfColumns, columns=columns)        

#================================================///////=============================================
import re
def detect_datetime_columns(df):
    result = {}

    for col in df.columns:
        col_data = df[col]

        # Handle if column is numeric and looks like a year
        if pd.api.types.is_integer_dtype(col_data) or pd.api.types.is_float_dtype(col_data):
            if col_data.dropna().empty == False:
            # Check if float values have only 2 decimal places
                if pd.api.types.is_float_dtype(col_data):
                    if col_data.dropna().apply(lambda x: round(x, 2) == x).all():
                        if col_data.dropna().between(1800, 2100).all():
                            result[col] = 1  # Only year
                            continue
                # For integer values
                elif col_data.dropna().between(1800, 2100).all():
                    result[col] = 1  # Only year
                    continue

        # Convert to string for flexible matching
        col_str = df[col].astype(str).str.strip()

        # Handle year-like strings
        if col_str.str.match(r'^\d{4}$').all():
            print("date is str")
            if col_str.astype(int).between(1800, 2100).all():
                result[col] = 1  # Only year
                continue

        # Check if values have full date (yyyy-mm-dd, dd-mm-yyyy, etc.)
        if col_str.str.match(r'^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$').all():
            result[col] = 2  # Only date
            continue

        # Check if values have date AND time
        if col_str.str.contains(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}.*\d{1,2}:\d{2}').all():
            result[col] = 3  # Date + Time
            continue

        # Else not a recognized datetime pattern
        result[col] = 0

    return result
# ===============================================///////==============================================
def clusterDateTimeCol(fContent, col,no,ascending):
    if no ==1:
        # Try to detect and sort if the column is just year values
        fContent = fContent.sort_values(by=col,ignore_index=True, ascending=ascending)
        fContent = fContent.reset_index(drop=True)
        fContent = multiIndex(fContent,col, True, asc=ascending)
    elif no == 2 or 3:
        # Now, try to detect proper date/datetime columns
        try:
            # Avoid processing numeric-only or zero-filled columns
            sample_vals = fContent[col].astype(str).str.strip().replace('0', np.nan).dropna()
            if len(sample_vals) == 0:
                return fContent# All values are zero or empty-like
            
            # Try parsing
            try:
                parsed_col = pd.to_datetime(fContent[col], errors='raise')
            except:
                parsed_col = pd.to_datetime(fContent[col], dayfirst=True, errors='raise')
               
            if all(parsed_col.dt.time == pd.to_datetime('00:00:00').time()) and no == 2:  # Only date
                fContent[col] = parsed_col
                fContent = clean_and_sort_date_column(fContent, col, ascending)
                fContent = fContent.reset_index(drop=True)
                fContent = multiIndex(fContent,col,yearOnly=False,asc=ascending)
                # Replace original column with parsed datetime values
            elif no == 3:  # Date + time
                fContent[col] = parsed_col
                fContent = handle_datetime_column(fContent, col, ascending)
                fContent = fContent.reset_index(drop=True)
                fContent = multiIndex(fContent,col,yearOnly=False,asc=ascending)
        except Exception as e:
            return fContent # Not a datetime column
        
   
    return fContent 

# ===============================================///==========================================
#fuction which will sort the dates if the df has date containing columns
def clean_and_sort_date_column(dff, column_name, ascending=True):
        try:
            
            # Step 2: Drop NaT (invalid formats)
            dff = dff.dropna(subset=[column_name])
            
            # Step 3: Sort the DataFrame by that column
            dff = dff.sort_values(by=column_name, ascending=ascending)

            # Optional: Format to clean date string (YYYY-MM-DD)
            dff[column_name] = dff[column_name].dt.strftime('%Y-%m-%d')
            print(f"{column_name} date called ")
            return dff
        
        except Exception as e:
            print(f"⚠️ Error while processing date column: {e}")
            return dff
# ===============================================///////==============================================
# if df has column containing the date and time both
def handle_datetime_column(df, column_name, ascending):
    print(f"{column_name} dateTime")
    # Check if most values in column are datetime with time
    values = df[column_name].dropna().astype(str).head(20)
    count_datetime = sum([is_datetime(v) for v in values])

    if count_datetime >= len(values) // 2:  # At least half must be datetime-like
        # Convert full column to datetime
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=[column_name])
        # Sort by that column
        df = df.sort_values(by=column_name,ascending=ascending).reset_index(drop=True)
        print(f"[INFO] '{column_name}' successfully recognized and sorted as datetime.")
    else:
        print(f"[INFO] '{column_name}' does not contain proper datetime with time.")

    return df
# ===============================================///////==============================================
#Group(MultiIndexing) accordind to the condition 
def multiIndex(dataFrame, colToCheck, yearOnly = False, asc=True):
    # Step 1: Get last indices of each year
    if yearOnly == True:
        yearsWithLastIndex = get_last_indices_of_each_year(dataFrame[colToCheck], True, asc)
    else:
        yearsWithLastIndex = get_last_indices_of_each_year(pd.to_datetime(dataFrame[colToCheck]),False, asc)
    if asc==False:
        yearsWithLastIndex = dict(reversed(list(yearsWithLastIndex.items())))
        
    print(yearsWithLastIndex)
    nameOfGroups = list(yearsWithLastIndex.keys())
    last_indices = list(yearsWithLastIndex.values())

    # Step 2: Compute counts from last indices
    group_sizes = []
    prev = -1
    for idx in last_indices:
        group_sizes.append(idx - prev)
        prev = idx
    print(group_sizes)
    # Step 3: Create array per group
    objOfGroups = {
        f'key{i}': np.array([f'Group of year {year}'] * group_sizes[i])
        for i, year in enumerate(nameOfGroups)
    }
    
    # Step 4: Combine into one array
    outside = np.concatenate(list(objOfGroups.values()))
    
    # Step 5: Create inside index
    inside = np.arange(len(outside))
    
    multi_index = pd.MultiIndex.from_arrays([outside, inside], names=["Group", "Sr No."])
    
    dataFrame = dataFrame.reindex(range(len(multi_index)))
    dataFrame.set_index(multi_index,inplace=True,)
    print("Multicalled")
    return dataFrame

# ===============================================///////==============================================
def get_last_indices_of_each_year(date_series, YearOnly=False, acs=True):
    
    # data_series = data_series.apply(pd.to_numeric, errors='coerce').astype('Int64')
    if YearOnly == True:
        print("yearOnly work")
        df = pd.DataFrame({'year': date_series}, index=np.arange(len(date_series)))
    
    if YearOnly == False:
        # Extract year
        years = date_series.dt.year
        
        # Create a DataFrame with index
        df = pd.DataFrame({'year': years}, index=np.arange(len(date_series)))
        
    # Get last index of each year group
    last_indices = df.groupby('year').apply(lambda x: x.index[-1]).to_dict()
        
    print("last index called")
    return last_indices

# ===============================================///////==============================================

def downLoadFile():
    pass
    
if __name__ == "__main__":
    app.run(debug = True)
    
