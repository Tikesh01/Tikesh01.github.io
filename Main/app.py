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
        print("error")
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
        if all(pd.api.types.is_numeric_dtype(dtype)  for dtype in df.dtypes):
            print('AllNumaric')
            return fReturn(table=df.head(50),quote='yes',noOfCluster='yes', columns=columns)
        else:
            print('simple')
            return fReturn(table=df.head(50),quote='yes', columns=columns)
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
    priorities = {key: value for key,value in priorities.items() if value}
    priorities = dict(sorted(priorities.items(),key=lambda item: item[1]))
    #get order fro form
    order = request.form.get('dec')
    if order == 'decO':
        order == False
    
    #No of clusters from from
    noOfClusters = request.form.get('nCluster')
        
    print(f"priorities = {priorities}")
    print(f'order : {order}')
    print(noOfClusters)
    
    if all(pd.api.types.is_numeric_dtype(dtype)  for dtype in mainDf.dtypes):
        # Process each priority column 
        mainDf = cluster_based_quantum_sort(mainDf, priorities.keys(), noOfClusters if noOfClusters else None,order=order)
            
    else:
        colTypes = detectColumns(mainDf, priorities.keys())
        print(colTypes)
        a = 0
        yearOnly = False
        for i in priorities.keys():
            if colTypes[i] != None:
                if colTypes[i] == 'yearOnly':
                    yearOnly = True
                    mainDf = clusterDateTimeCol(mainDf, i,1, ascending=order)
                elif colTypes[i] == 'dateOnly':
                    mainDf = clusterDateTimeCol(mainDf,i,2,ascending=order)
                elif colTypes[i] == 'dateAndTime':
                    mainDf = clusterDateTimeCol(mainDf, i, 3, ascending=order)
                elif colTypes[i] == 'rollNo':
                    pass
                elif colTypes[i] == 'id':
                    pass
                elif colTypes[i] == 'oneOr2Digit':
                    pass
                elif colTypes[i] == 'numeric' or colTypes[i] == 'allInt':
                    pass
                elif colTypes[i] == 'str':
                    pass
                if a== 0:#Multiindex only for first time
                    mainDf = multiIndex(mainDf,i,colTypes[i], yearOnly=yearOnly, asc=True if order==None else False)
                a+=1
    app.config['mainDf'] = mainDf
    arrOfGroupNames = mainDf.index.get_level_values(0).unique() if isinstance(mainDf.index, pd.MultiIndex) else []

    if len(arrOfGroupNames) != 0:
        smallDf = mainDf.groupby(level=0).head(10)
    else:
        smallDf = mainDf.head(50)
    
    print("All done")
    return fReturn(app.config['df'].head(50),clustTable=smallDf, noOfColumns=noOfColumns,noOfRows=noOfRows)
    
    
def fReturn(table,clustTable=None, noOfCluster=None, quote=None, columns=None, noOfColumns=None,noOfRows =None):
    try:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            print("succes")
            return {
                "table": render_template("partials/data_table.html", table=table.to_html(classes="UploadedData", border=0), noOfColumns=len(app.config['df'].columns), noOfRows=len(app.config['df'].index)),
                "clustered" : render_template("partials/clustered_data.html", clusteredData=clustTable.to_html(classes="UploadedClusteredData", border=0), noOfColumns=noOfColumns, noOfRows= noOfRows)if clustTable !=None else '',
                "quote": render_template("partials/priority_form.html", quote=quote, columns=columns, noOfCluster=noOfCluster) if quote!=None else ''
            }     
    
    except Exception as e:
        print(f'error : {e}')
    return render_template("index.html", table=app.config['df'].head(50).to_html(classes="UploadedData", border=0), clusteredData=clustTable.to_html(classes="UploadedClusteredData", border=0), noOfRows=noOfRows, noOfColumns=noOfColumns)


import re
def detectColumns(df, prioColumns):
    result = {}    
    for col in prioColumns:
        # Initialize result as 0 (unrecognized type)
        result[col] = None
        
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
        
        if detectIdTypeCol(col_data):
            result[col] = 'id'
            continue
        if OneOr2digitDetection(col_data):
            result[col] = 'oneOr2Digit'
            continue
        else:
            result[col] = detectSimpleDtypes(col_data)
            
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

def detectIdTypeCol(col_data):
    if pd.api.types.is_string_dtype(col_data):
        pattern = r'\b[A-Z0-9]{1,4}[-_./]?[A-Z0-9]{2,6}[-_./]?[A-Z0-9]{0,5}\b'
        return all(re.fullmatch(pattern, item) for item in col_data)
    
def OneOr2digitDetection(col_data):
    try:
        if all(len(str(i))<=2 for i in col_data):
            return True
    except:
        pass
    return False

def detectSimpleDtypes(col_data):
    if pd.api.types.is_integer_dtype(col_data):
        return 'allInt'
    if pd.api.types.is_float_dtype(col_data):
        if col_data.isna().all()== False:
            return 'numaric'
    if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
        return 'str'
    return None

def clusterDateTimeCol(fContent, col,no,ascending=True):
    if no ==1:
        # Try to detect and sort if the column is just year values
        fContent = fContent.sort_values(by=col,ignore_index=True, ascending=ascending)
        fContent = fContent.reset_index(drop=True)
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
                # Replace original column with parsed datetime values
            elif no == 3:  # Date + time
                fContent[col] = parsed_col
                fContent = handle_datetime_column(fContent, col, ascending)
                fContent = fContent.reset_index(drop=True)
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

def multiIndex(dataFrame, colToCheck, colType, yearOnly = False, asc=True):
    # Step 1: Get last indices of each year
    if yearOnly == True:
        yearsWithLastIndex = get_last_indices_of_each_year(dataFrame[colToCheck], True, asc)
    else: 
        if colType=="dateOnly" or colType == 'dateAndTime':
            yearsWithLastIndex = get_last_indices_of_each_year(pd.to_datetime(dataFrame[colToCheck]),False, asc)
        else:
            yearsWithLastIndex = get_last_indices_of_each_year(dataFrame[colToCheck], False, asc)
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
        f'key{i}': np.array([f'Group of {year}'] * group_sizes[i]) for i, year in enumerate(nameOfGroups)
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
        try:
            years = date_series.dt.year
            # Create a DataFrame with index
            df = pd.DataFrame({'year': years}, index=np.arange(len(date_series)))
        except:
             #Create a DataFrame with index
            df = pd.DataFrame({'year': date_series}, index=np.arange(len(date_series)))
        
    # Get last index of each year group
    last_indices = df.groupby('year').apply(lambda x: x.index[-1]).to_dict()
        
    print("last index called")
    return last_indices

from sklearn.cluster import KMeans
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
import time

# Cache for storing quantum circuits to avoid recreating them
circuit_cache = {}

def create_oracle(values, target_idx, num_qubits):
    st = time.time()
    cache_key = f'oracle_{target_idx}_{num_qubits}'
    if cache_key in circuit_cache:
        return circuit_cache[cache_key]

    oracle = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if (target_idx >> i) & 1:
            oracle.x(i)
    
    if num_qubits == 1:
        oracle.h(0)
        oracle.z(0)
        oracle.h(0)
    elif num_qubits > 3:
        mid = num_qubits // 2
        oracle.h(num_qubits - 1)
        oracle.mcx(list(range(mid)), mid)
        oracle.mcx(list(range(mid, num_qubits - 1)), num_qubits - 1)
        oracle.h(num_qubits - 1)
    else:
        oracle.h(num_qubits - 1)
        if num_qubits == 2:
            oracle.cx(0, 1)
        else:
            oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    
    for i in range(num_qubits):
        if (target_idx >> i) & 1:
            oracle.x(i)
    circuit_cache[cache_key] = oracle
    et = time.time()
    print(f"create_oracle time = {et-st}")
    return oracle

def create_diffusion(num_qubits):
    st = time.time()
    cache_key = f'diffusion_{num_qubits}'
    if cache_key in circuit_cache:
        return circuit_cache[cache_key]

    diffusion = QuantumCircuit(num_qubits + 1)
    for qubit in range(num_qubits):
        diffusion.h(qubit)
    for qubit in range(num_qubits):
        diffusion.x(qubit)
    chunk_size = 3
    for i in range(0, num_qubits - 1, chunk_size):
        control_qubits = list(range(i, min(i + chunk_size, num_qubits - 1)))
        if len(control_qubits) > 0:
            diffusion.h(num_qubits)
            diffusion.mcx(control_qubits, num_qubits)
            diffusion.h(num_qubits)
    for qubit in range(num_qubits):
        diffusion.x(qubit)
    for qubit in range(num_qubits):
        diffusion.h(qubit)
    circuit_cache[cache_key] = diffusion
    et = time.time()
    print(f"create_diffusion = {et-st}")
    return diffusion

def grover_find_min_index(values):
    st = time.time()
    n = len(values)
    num_bits = max(1, int(np.ceil(np.log2(n))))
    min_idx = np.argmin(values)
    
    qr = QuantumRegister(num_bits + 1, 'q')
    cr = ClassicalRegister(num_bits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    for i in range(num_bits):
        circuit.h(qr[i])
    
    iterations = int(np.pi/4 * np.sqrt(2**num_bits))
    oracle = create_oracle(values, min_idx, num_bits + 1)
    diffusion = create_diffusion(num_bits)
    
    for _ in range(iterations):
        circuit = circuit.compose(oracle)
        circuit = circuit.compose(diffusion)
    
    for i in range(num_bits):
        circuit.measure(qr[i], cr[i])
    
    backend = Aer.get_backend('aer_simulator')
    result = backend.run(circuit, shots=1000).result()
    counts = result.get_counts()
    max_count_result = max(counts.items(), key=lambda x: x[1])[0]
    et = time.time()
    print(f" grover find min index time = {et -st}")

    return int(max_count_result, 2) % n

def quantum_sort_cluster(cluster_df, sort_column):
    st=time.time()
    if len(cluster_df) == 0:
        return cluster_df
    
    df = cluster_df.copy()
    sorted_indices = []
    values = df[sort_column].tolist()
    remaining_indices = list(range(len(values)))
    
    while remaining_indices:
        remaining_values = [values[i] for i in remaining_indices]
        min_idx = grover_find_min_index(remaining_values)
        actual_idx = remaining_indices[min_idx]
        sorted_indices.append(actual_idx)
        remaining_indices.remove(actual_idx)
    
    et=time.time()
    print(f"quantum_sort_cluster time = {et -st}")
    return df.iloc[sorted_indices].reset_index(drop=True)

def cluster_based_quantum_sort(df, Pcols, n_clusters=None, i=0, order =True):
    if i >= len(Pcols):
        return df

    sort_column = Pcols[i]
    if sort_column not in df.columns:
        print(f"Column '{sort_column}' not found.")
        return df

    print(f'\nLevel {i}: Sorting by column "{sort_column}"')
    start_time = time.time()

    # Perform clustering on the current sort_column
    clustering_data = df[[sort_column]]
    if n_clusters is None:
        n_clusters = int(np.ceil(len(df[sort_column]) / 60))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(clustering_data)

    unique_clusters = sorted(df['cluster'].unique())
    all_sorted = []

    for cluster_id in unique_clusters:
        cluster_df = df[df['cluster'] == cluster_id].drop(columns=['cluster'])
        print(f"  → Cluster {cluster_id} (size {len(cluster_df)})")

        # Sort this cluster
        sorted_cluster = quantum_sort_cluster(cluster_df, sort_column)

        # Recursively sort the next column (if available)
        next_sorted_cluster = cluster_based_quantum_sort(pd.DataFrame(sorted_cluster), Pcols, i=i+1)
        all_sorted.append(next_sorted_cluster)

        print(f"  ✓ Completed Cluster {cluster_id}")

    # Merge and sort by current column
    merged_df = pd.concat(all_sorted, ignore_index=True)
    final_sorted_df = merged_df.sort_values(by=sort_column,ascending=order).reset_index(drop=True)

    end_time = time.time() 
    # print(f"✔ Level {i} sorting by '{sort_column}' completed in {end_time - start_time:.2f} seconds")

    return final_sorted_df
    
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

