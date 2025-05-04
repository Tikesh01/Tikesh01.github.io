from flask import Flask, render_template, request, send_file, make_response
import os
import pandas as pd
import numpy as np
from jinja2 import Environment
import io

app = Flask(__name__)
env = Environment(autoescape=True)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

@app.route('/')
def interface():
    return render_template('index.html')

os.makedirs("userFiles", exist_ok=True)
@app.route('/uploader', methods = ["POST"])
def getFile():
    print("AJAX detected:", request.headers.get("X-Requested-With"))
    listOfExtension = ["csv","xsl","sql","sqlquery","xlsx","json","txt"]
    file = request.files["file"]
    filePath = os.path.join("userFiles", file.filename)
    app.config['path'] = filePath
    nameOfFile = file.filename.split('.')
    uploadedFileExtension = nameOfFile[len(nameOfFile)-1]
    if uploadedFileExtension in listOfExtension:
        file.save(filePath)
        return detectType(uploadedFileExtension,filePath,file.filename)
    else:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            error_html = render_template("partials/type_error.html", error=f"only {listOfExtension} files are allowed")
            return {"error": error_html}
        else:
            return render_template("index.html", error=f"only {listOfExtension} files are allowed")

def detectType(type,filePath,fileName):
    if type == "csv":
        df = pd.read_csv(filePath)
        os.remove(filePath)
        app.config['df'] =  df.dropna(how="all")
        columns = df.columns
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return {
                "table": render_template("partials/data_table.html",table=df.head(50).to_html(classes="UploadedData", border=0), noOfRows=len(df.index), noOfColumns=len(columns), fileName=fileName),
                "quote": render_template("partials/priority_form.html", quote="Select priorities for clustering", columns=columns)
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

@app.route('/start', methods=["POST"])
def uploadFileToClust():
    mainDf = app.config['df']
    columns = np.array(mainDf.columns)
    noOfColumns = len(columns)
    noOfRows = len(mainDf.index)
    print(f"Total columns = {noOfColumns}")
    print(f"Total rows = {noOfRows}")

    # Get priorities from form
    priorities = {}
    for i in columns:
        priorities[i] = request.form.get(f'{i}')
    decending = request.form.get('dec')
    if decending == 1:
        decending == False
    priorities = {key: value for key,value in priorities.items() if value}
    priorities = dict(sorted(priorities.items(),key=lambda item: item[1]))
    print(f"priorities = {priorities}")
    
    if all(pd.api.types.is_numeric_dtype(dtype)  for dtype in mainDf.dtypes):
        # Process each priority column 
        for col in priorities.keys():
            pass
            
    else:
        r = detectColumns(mainDf, priorities.keys())
        print(r)
        for i in r.keys():
            if r[i] !=0:
                if r[i] == 'rollNo':
                    pass
                if r[i] == 'yearOnly':
                    mainDf = clusterDateTimeCol(mainDf, i,1)
                if r[i] == 'dateOnly':
                    mainDf = clusterDateTimeCol(mainDf,i,2)
                if r[i] == 'dateAndTime':
                    mainDf = clusterDateTimeCol(mainDf, i, 3)
                    
    app.config['mainDf'] = mainDf
    
    arrOfGroupNames = mainDf.index.get_level_values(0).unique() if isinstance(mainDf.index, pd.MultiIndex) else []

    if len(arrOfGroupNames) != 0:
        smallDf = mainDf.groupby(level=0).head(20)
    else:
        smallDf = mainDf.head(50)
            
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return {
            "table": render_template("partials/data_table.html", table=app.config['df'].head(50).to_html(classes="UploadedData", border=0), noOfColumns=noOfColumns, noOfRows=noOfRows),
            "clustered": render_template("partials/clustered_data.html", clusteredData=smallDf.to_html(classes="UploadedClusteredData", border=0), noOfColumns=noOfColumns, noOfRows=noOfRows)
        }     
    else:
        return render_template("index.html", table=app.config['df'].head(50).to_html(classes="UploadedData", border=0), clusteredData=smallDf.to_html(classes="UploadedClusteredData", border=0), noOfRows=noOfRows, noOfColumns=noOfColumns)

import re
def detectColumns(df, prioColumns):
    result = {}    
    for col in prioColumns:
        # Initialize result as 0 (unrecognized type)
        result[col] = 0
        
        col_data = df[col]
        col_str = df[col].astype(str).str.strip()
        
        # 1. Check for Roll Numbers (type 4)
        if pd.api.types.is_integer_dtype(col_data):
            if check_roll_number(col_data):
                result[col] = 'rollNo'
                continue
        
        # 2. Check for Year values (type 1)
        if check_year_values(col_data, col_str):
            result[col] = 'yearOnly'
            continue
            
        # 3. Check for Date values (type 2)
        if check_date_format(col_str):
            result[col] = 'dateOnly'
            continue
            
        # 4. Check for DateTime values (type 3)
        if check_datetime_format(col_str):
            result[col] = 'dateAndTime'
            continue
    
    return result

def check_roll_number(col_data):
    try:
        # Convert numbers to strings for checking patterns
        sample_start = [str(i) for i in col_data.head(10)]
        sample_middle = [str(i) for i in col_data.iloc[int(len(col_data)/2)-5:int(len(col_data)/2)+5]]
        sample_end = [str(i) for i in col_data.iloc[-10:]]
        
        # Combine samples
        samples = sample_start + sample_middle + sample_end
        
        # Check if all numbers have the same length and >= 5 digits
        if len(set(len(str(x)) for x in samples)) == 1:
            length = len(str(samples[0]))
            if length >= 5:
                # Check if all numbers start with the same digit
                first_digit = str(samples[0])[0]
                return all(str(x).startswith(first_digit) for x in samples)
    except:
        pass
    return False

def check_year_values(col_data, col_str):
    # Handle if column is numeric and looks like a year
    if pd.api.types.is_integer_dtype(col_data) or pd.api.types.is_float_dtype(col_data):
        if col_data.dropna().empty == False:
            if pd.api.types.is_float_dtype(col_data):
            # Check if float values have only 2 decimal places
                if col_data.dropna().apply(lambda x: round(x, 2) == x).all():
                    if col_data.dropna().between(1800, 2100).all():
                        return True # Only year
            # For integer values
            elif col_data.dropna().between(1800, 2100).all():
                True # Only year
    return False

def check_date_format(col_str):
    # Check for common date formats (yyyy-mm-dd, dd-mm-yyyy, etc.)
    date_pattern = r'^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$'
    return col_str.str.match(date_pattern).all()

def check_datetime_format(col_str):
    # Check for datetime format (date + time)
    datetime_pattern = r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}.*\d{1,2}:\d{2}'
    return col_str.str.contains(datetime_pattern).all()

def clusterDateTimeCol(fContent, col,no,ascending=True):
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

def handle_datetime_column(df, column_name, ascending):
    print(f"{column_name} dateTime")
    # Check if most values in column are datetime with time
    values = df[column_name].dropna().astype(str).head(20)
    count_datetime = sum([pd.api.is_datetime(v) for v in values])

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

@app.route('/Download', methods=["POST"])
def downLoadFile():
    file = request.form.get('fileTypes')
    df = app.config.get('mainDf')
    
    if df is None:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return {"error": "No data available to download"}
        return render_template("index.html", error="No data available to download")
    
    if file == 'csv':
        # Create a string buffer
        buffer = io.StringIO()
        # Write the dataframe to the buffer
        df.to_csv(buffer, index=False)
        # Get the value of the buffer
        buffer.seek(0)
        
        # Create the response
        output = make_response(buffer.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=exported_data.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    elif file == 'excel':
        # Create a bytes buffer for Excel file
        buffer = io.BytesIO()
        # Write the dataframe to the buffer
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='exported_data.xlsx'
        )
    elif file == 'json':
        # Create a string buffer
        buffer = io.StringIO()
        # Write the dataframe to JSON
        df.to_json(buffer, orient='records')
        buffer.seek(0)
        
        # Create the response
        output = make_response(buffer.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=exported_data.json"
        output.headers["Content-type"] = "application/json"
        return output
        
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return {"error": "Invalid file type selected"}
    return render_template("index.html", error="Invalid file type selected")

if __name__ == "__main__":
    app.run(debug = True)

