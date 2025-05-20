import os
import io
import time
import numpy as np
import pandas as pd
from jinja2 import Environment
from sklearn.cluster import KMeans
from qiskit_aer import AerSimulator, Aer
from qiskit import QuantumCircuit, transpile
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from flask import Flask, render_template, request, send_file, make_response


app = Flask(__name__)
env = Environment(autoescape=True)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

@app.route('/')
def interface():
    return render_template('index.html')

os.makedirs("QubitSort_User_Files", exist_ok=True)
@app.route('/uploader', methods = ["POST"])
def getFile():
    print("AJAX detected:", request.headers.get("X-Requested-With"))
    listOfExtension = ["csv","xsl","sql","sqlquery","xlsx","json","txt"]
    file = request.files["file"]
    filePath = os.path.join("userFiles", file.filename)
    app.config['path'] = filePath
    nameOfFile = file.filename.split('.')
    uploadedFileExtension = nameOfFile[len(nameOfFile)-1].lower()
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
    elif type == "xls" or "xlsx":
        df = pd.read_excel(filePath)
    elif type == "sql":
        df = pd.read_sql_table(filePath)
    elif type == "json":
        df = pd.read_json(filePath)
        
    os.remove(filePath)
    app.config['df'] =  df.dropna(how="all")
    columns = df.columns
    if all(pd.api.types.is_numeric_dtype(dtype)  for dtype in df.dtypes):
            print('AllNumaric')
            return fReturn(table=df.head(50),quote='yes',noOfCluster='yes', columns=columns)
    else:
        print('simple')
        return fReturn(table=df.head(50),quote='yes',columns=columns)

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
    asc = True
    order = request.form.get('dec')
    if order == 'decO':
        asc = False
    #No of clusters from from
    noOfClusters = request.form.get('nCluster')
        
    print(f"priorities = {priorities}")
    print(f'ascending : {asc}')
    print(noOfClusters)
    
    try:
        if all(pd.api.types.is_numeric_dtype(dtype)  for dtype in mainDf.dtypes):
            # Process each priority column 
            mainDf = cluster_based_quantum_sort(mainDf, priorities.keys(), noOfClusters if noOfClusters else None,order=order)
                
        else:   
            colTypes = detectColumns(mainDf, priorities.keys())
            mainDf = dataClean(mainDf,colTypes)
            print(colTypes)
            mainDf = mainRecursiveSort(mainDf, priorityCols=list(priorities.keys()),colTypes=colTypes)
            mainDf = multiIndex(dataFrame=mainDf,colToCheck=list(priorities.keys()[0]),colType= {next(iter(colTypes)): next(iter(colTypes.values()))})
    except Exception as e:
        fReturn(table=app.config['df'].head[50],error=e)
       
    arrOfGroupNames = mainDf.index.get_level_values(0).unique() if isinstance(mainDf.index, pd.MultiIndex) else []
    print(arrOfGroupNames)
    app.config['mainDf'] = mainDf
    if len(arrOfGroupNames) != 0:
        app.config['smallDf'] = mainDf.groupby(level=0).head(10)
    else:
        app.config['smallDf']= mainDf.head(50)
    
    noOfColumns = len(mainDf.columns)
    noOfRows = len(mainDf.index)
    print("All done")
    return fReturn(app.config['df'].head(50),clustTable=app.config['smallDf'], noOfColumns=noOfColumns,noOfRows=noOfRows,quote='yes',columns=columns)

def mainRecursiveSort(df, priorityCols, colTypes, level=0, asc=True, multIndex=True, yearOnly=False):
    if level >= len(priorityCols):
        return df

    col = priorityCols[level]

    if colTypes[col]['0'] != 100:  # column has type
        if colTypes[col]['0'] == 60 or colTypes[col]['1'] == 40:
            yearOnly = True
            df = clusterDateTimeCol(df, col, 1, ascending=asc)
        elif colTypes[col]['0'] == 30 or colTypes[col]['1'] == 70:
            df = clusterDateTimeCol(df, col, 2, ascending=asc)
        elif colTypes[col]['0'] == 20 or colTypes[col]['1'] == 80:
            df = clusterDateTimeCol(df, col, 3, ascending=asc)
        elif colTypes[col]['0'] == 70 or colTypes[col]['1'] == 30:
            df = sortRollCol(df, col, asc)
        elif colTypes[col]['1'] == 50 or colTypes[col]['0'] == 50:
            df = sortRollCol(df, col, asc)
        elif colTypes[col]['0'] == 90 or colTypes[col]['1'] == 10:
            df = digitSorting(df, col, asc)
        elif colTypes[col]['0'] == 80 or colTypes[col]['1'] == 20:
            df = df.sort_values(by=col, ignore_index=True, ascending=asc)
            multIndex = False
        elif colTypes[col]['0'] == 40 or colTypes[col]['1'] == 60:
            df = cluster_text_column(df=df, column_name=col)

        # Apply MultiIndex if needed
        if level == 0 and multIndex:
            df = multiIndex(df, col, colTypes[col], yearOnly=yearOnly, asc=asc)

        # Recurse: sort within each group of this column
        if isinstance(df.index, pd.MultiIndex):
            grouped = [df.loc[group].copy() for group in df.index.get_level_values(0).unique()]
            sorted_groups = [mainRecursiveSort(group, priorityCols, colTypes, level+1, asc, multIndex, yearOnly) for group in grouped]
            df = pd.concat(sorted_groups).reset_index(drop=True)
        else:
            df = mainRecursiveSort(df, priorityCols, colTypes, level+1, asc, multIndex, yearOnly)

    return df

def fReturn(table,clustTable=None,error=None, noOfCluster=None, quote=None, columns=None, noOfColumns=None,noOfRows =None):
    try:
        if error == None:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                print("succes")
                return {
                    "table": render_template("partials/data_table.html", table=table.to_html(classes="UploadedData", border=0), noOfColumns=len(app.config['df'].columns), noOfRows=len(app.config['df'].index), columns=columns),
                    "clustered" : render_template("partials/clustered_data.html", clusteredData=clustTable.to_html(classes="UploadedClusteredData", border=0), noOfColumns=noOfColumns, noOfRows= noOfRows, columns=columns)if clustTable !=None else '',
                    "quote": render_template("partials/priority_form.html", quote=quote, columns=columns, noOfCluster=noOfCluster) if quote!=None else ''
                }     
    
    except Exception as e:
        print(f'error : {e}, {error}')
    return render_template("index.html", table=app.config['df'].head(50).to_html(classes="UploadedData", border=0), clusteredData=clustTable.to_html(classes="UploadedClusteredData", border=0), noOfRows=noOfRows, noOfColumns=noOfColumns,columns=columns)

def dataClean(df,colTypes):
    for i in df.index[(df.isna().sum(axis=1)==len(df.columns))]:
        df= df.drop(index=i, inplace=True)
        
    for i in colTypes.keys():
        if df[i].isna().sum() >=len(df[i])-2:
            df = df.drop(columns=[i],inplace=True)
            print(i,' droped')
            continue
        
        if df[i].isna().sum() != 0:#IF column has no type
                print(i)
                if (colTypes[i]['0'] == 60 or colTypes[i]['1'] == 40): # if column is type of year only
                    df[i] = df[i].fillna(2099)
                elif colTypes[i]['0'] == 30 or  colTypes[i]['1'] == 70:#if column is type of date only 
                    df[i] = df[i].fillna(pd.to_datetime('2030-01-01'))
                elif colTypes[i]['0'] == 20 or  colTypes[i]['1'] == 80 :#if column is type of date and time 
                    df[i]= df[i].fillna('2030-01-01')
                elif colTypes[i]['0'] == 70 or colTypes[i]['1']==30: #Roll no type
                    pass
                elif colTypes[i]['1']==50 or colTypes[i]['0']==50:#if it is of type id 
                    df[i]=df[i].fillna("NAN-123-00")
                elif colTypes[i]['0']==90 or colTypes[i]['1'] ==10:#if it of type oneor2digit
                    df[i] = df[i].fillna(0)
                elif colTypes[i]['0'] == 80 or  colTypes[i]['1'] == 20 : #if it type of nemric
                    df[i]=df[i].fillna(0)
                elif colTypes[i]['0'] == 40 or  colTypes[i]['1'] == 60 : #string or object
                    df[i] = df[i].fillna('NULL-NAN')
            
        elif len(set(df[i]))==1:
            for k in range(2):
                df[i].at[k]="Changed For better Clustering"
        continue
    return df

import re
import math
def detectColumns(df, prioColumns):
    result = {}    
    for col in prioColumns:
        qc =  QuantumCircuit(1,1)
        col_data = df[col]
        col_str = df[col].astype(str).str.strip()
        # 3. Check for Date values (type 2)
        if check_date_format(col_str):
            p = 0.70000
                
        # 4. Check for DateTime values (type 3)
        elif check_datetime_format(col_str):
            p = 0.800000
            
        elif OneOr2digitDetection(col_data) and detectIdTypeCol(col_data)==False:
            p = 0.100000
        # 1. Check for Roll Numbers (type 4)
        elif pd.api.types.is_numeric_dtype(col_data):
            p = 0.2000
            # 2. Check for Year values (type 1)
            if check_year_values(col_data):
                p = 0.40000
            elif check_roll_number(col_data):
                p = 0.30000

        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            p = 0.600000
            if detectIdTypeCol(col_data):
                p = 0.5000
        
        angle = 2 * math.asin(math.sqrt(p))
        qc.ry(angle, 0)
        # Initialize result as 0 (unrecognized type)
        result[col] = measurCir(qc,0)
        
    return result

def measurCir(i,j):
    i.measure(j,j)
    simulator = AerSimulator()
    # Transpile & run
    compiled = transpile(i, simulator)
    r = simulator.run(compiled, shots=100000).result()
    counts = r.get_counts()
    for k,v in counts.items():
        counts[k] = int(v/1000)
    return counts

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

def check_year_values(col_data):
    if pd.api.types.is_string_dtype(col_data):
        try:
            col_data = pd.to_numeric(col_data)
        except:
            pass
    # Handle if column is numeric and looks like a year
    if pd.api.types.is_numeric_dtype(col_data) or  pd.api.types.is_float_dtype(col_data) or  pd.api.types.is_integer_dtype(col_data):
        if col_data.dropna().empty == False:
            if pd.api.types.is_float_dtype(col_data):
            # Check if float values have only 2 decimal places
                if col_data.dropna().apply(lambda x: round(x, 2) == x).all():
                    if col_data.dropna().between(1800, 2050).all():
                        return True # Only year
            # For integer values
            elif col_data.dropna().between(1800, 2100).all():
                return True # Only year
                
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
    non2Digit=0
    for v  in col_data:
        if len(str(v)) > 2 and str(v)!='nan':
            non2Digit=o=non2Digit+1
    
    if len(col_data)/2.2 > non2Digit:
        return True
    return False

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
                fContent = clean_and_sort_date_column(fContent, col, ascending=ascending)
                fContent = fContent.reset_index(drop=True)
                # Replace original column with parsed datetime values
            elif no == 3:  # Date + time
                fContent[col] = parsed_col
                fContent = handle_datetime_column(fContent, col, ascending=ascending)
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

def digitSorting(df,col,asc):
    df[col] = df[col].astype(str)
    df = df.sort_values(by=col,ascending=asc, ignore_index=True)
    return df

def sortRollCol(df, col,asc):
    digits_df = df[col].astype(str).apply(lambda x: pd.Series(list(x)))
    
    digits_df['original_index'] = df.index
    
    sorted_digits_df = recursiveSort(digits_df)

    sorted_indices = sorted_digits_df['original_index'].values
    if asc == False:
        sorted_indices = reversed(sorted_indices)
        
    sorted_df = df.loc[sorted_indices].reset_index(drop=True)

    return sorted_df

def recursiveSort(df_digits, col=0):
    if col >= int(len(df_digits.columns) - 1):  # exclude 'original_index' column
        return df_digits

    # Sort by the current digit column
    df_digits = df_digits.sort_values(by=col, kind='stable', ignore_index=True)

    # Group by current digit and recursively sort each group
    result = []
    for value, group in df_digits.groupby(col, sort=False):
        sorted_group = recursiveSort(group.reset_index(drop=True), col + 1)
        result.append(sorted_group)
    try:
        return pd.concat(result, ignore_index=True)
    except:
        return pd.DataFrame(data=result,index=np.arange(len(result)))

from sentence_transformers import SentenceTransformer 
from sklearn.metrics import silhouette_score
def cluster_text_column(df: pd.DataFrame, column_name: str, k_min: int = 2, k_max: int = 40, plot: bool = True) -> pd.DataFrame:
    comments = df[column_name].dropna().astype(str).tolist()

    # Sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X = model.encode(comments)

    # Find the best number of clusters using silhouette scores
    k_range1 = list(range(k_min, 11,2))
    k_range2 = list(range(12,25,3))
    k_range3 = list(range(28,40,4))
    k_range= k_range1+k_range1+k_range2+k_range3
    scores = silhouetteScores(k_range,X)
    print(scores)
    best_k_range =dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2])
    print(best_k_range)
    
    k_scores = {}
    for i in best_k_range.keys():
        best_k = silhouetteScores(list(range(int(i) - 1, int(i) + 2)), X)
        max_k = max(best_k, key=best_k.get)
        k_scores[int(max_k)] = best_k[max_k]
        if best_k[max_k] == 1.0:
            break

    # Get the k with the highest silhouette score
    final_k = max(k_scores, key=k_scores.get)
    best_score = k_scores[final_k]

    print(f"\nBest K: {final_k}, Highest Silhouette Score: {best_score:.4f}")

    final_model = KMeans(n_clusters=final_k, random_state=42, n_init=20, max_iter=550)
    final_labels = final_model.fit_predict(X)

    # Create a copy of the original DataFrame and assign cluster labels
    df_clustered = df.copy()
    df_clustered["cluster_label"] = -1
    df_clustered.loc[df[column_name].notna(), "cluster_label"] = final_labels
    df_clustered = df_clustered.sort_values(by='cluster_label',ignore_index=True)
    
    return df_clustered

def silhouetteScores(k_range, X):
    if 1 in k_range:
        k_range.remove(1)
    scores = {}
    a = 0
    index = 0
    while index < len(k_range):
        k = k_range[index]

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"K={k}, Silhouette Score={score:.4f}")

        # Check conditions and expand k_range dynamically
        if len(scores.values()) >= 14 and all(x < 0.7 for x in scores.values()):
            print("nye")
            k_range = list(range(k + 6, 170, 6))
            continue
        
        if len(scores.values()) >= 10 and all(x < 0.48 for x in scores.values()):
            print("kye")
            k_range = list(range(k + 6, 112, 5))
            continue
        
        if len(scores.values()) >= 7 and all(x < 0.32 for x in scores.values()):
            print("bye")
            k_range = list(range(k + 5, 90, 4))
            continue
        
        if len(scores.values()) >= 2:
            if all(x < 0.25 for x in scores.values()):
                print("hii")
                k_range = list(range(k + 4, 60, 3))
                continue
            values = list(scores.values())
            if values[a] > values[a + 1] or len(set(values[-2:])) == 1:
                return scores
            
            a += 1
        index += 1
    
    return scores

def multiIndex(dataFrame, colToCheck, colType, yearOnly = False, asc=True):
    # Step 1: Get last indices of each years
    skip= False
    if yearOnly == True:
        yearsWithLastIndex = get_last_indices_of_each_year(dataFrame[colToCheck], True)
    else: 
        if (colType['0'] == 30 or  colType['1'] == 70) or (colType['0'] == 20 or  colType['1'] == 80):
            yearsWithLastIndex = get_last_indices_of_each_year(pd.to_datetime(dataFrame[colToCheck]),False)
        elif(colType['1']==30 or colType['0']==70) or (colType['1'] ==50 or colType['0']==50):
            yearsWithLastIndex = get_last_indices_of_each_year(dataFrame[colToCheck],YearOnly=False,rol=True,asc=asc)
            asc=True
        elif(colType['1']==60 or colType['0']==40):
            skip = True
        else:   
            yearsWithLastIndex = get_last_indices_of_each_year(dataFrame[colToCheck], False)
    if asc==False and skip !=True:
        yearsWithLastIndex = dict(reversed(list(yearsWithLastIndex.items())))
    if skip ==False:   
        if yearsWithLastIndex == None or len(yearsWithLastIndex.items()) == 0:
            return dataFrame
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
    else:
        outside = np.array(dataFrame['cluster_label'])
    
    inside = np.arange(len(outside))

    print(outside)
    # Step 5: Create inside index
    
    multi_index = pd.MultiIndex.from_arrays([outside, inside], names=["Group", "Sr No."])
    
    dataFrame = dataFrame.reindex(range(len(multi_index)))
    dataFrame.set_index(multi_index,inplace=True,)
    print("Multicalled")
    return dataFrame
    
def get_last_indices_of_each_year(date_series, YearOnly=False,rol=False, a=0.6,asc=True):
    
    # data_series = data_series.apply(pd.to_numeric, errors='coerce').astype('Int64')
    if rol==True and findDuplicate(date_series) <=10:
            df = pd.DataFrame({'year':date_series},index=np.arange(len(date_series))).astype(str)
            df['half'] = df['year'].apply(lambda x: x[:round(len(str(x))*a)])
            print(a)
            last_indices =  df.groupby('half',).apply(lambda x: x.index[-1]).to_dict()
            if asc ==False:
                last_indices = dict(reversed(list(last_indices.items())))
                
            group_sizes = []
            prev = -1
            k=0
            for i,idx in enumerate(last_indices.values()):
                group_sizes.append(idx - prev)
                prev = idx
                if group_sizes[i]<= 15:
                    k=k+1
            print()
            if round(len(last_indices)*0.45)<=k and a>=0.20:
                last_indices = get_last_indices_of_each_year(date_series,False,True,a=round(a-0.10,ndigits=3), asc=asc)
            return last_indices
    if YearOnly == True:
        print("yearOnly work")
        df = pd.DataFrame({'year': date_series}, index=np.arange(len(date_series)))
    
    else:
        # Extract year
        try:
            years = date_series.dt.year
            # Create a DataFrame with index
            df = pd.DataFrame({'year': years}, index=np.arange(len(date_series)))
            print("Year Extracted")
        except:
            #Create a DataFrame with index
            df = pd.DataFrame({'year': date_series}, index=np.arange(len(date_series)))
            print("normal")
            
    last_indices = {}        
    if df['year'].nunique() != len(df['year']):
    # Get last index of each year group
        last_indices = df.groupby('year').apply(lambda x: x.index[-1]).to_dict()
        
    print("last index called")
    return last_indices

def findDuplicate(series):
    series = series[series.duplicated(keep=False)]
    half = set(series)
    print(len(half))
    if len(half)>=len( series)*0.30:
        return 9
    return 11


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
    return diffusion

def grover_find_min_index(values):
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
    return int(max_count_result, 2) % n

def quantum_sort_cluster(cluster_df, sort_column):
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
    
    return df.iloc[sorted_indices].reset_index(drop=True)

def cluster_based_quantum_sort(df, Pcols, n_clusters=None, i=0, order =True):
    if i >= len(Pcols):
        return df

    sort_column = Pcols[i]
    if sort_column not in df.columns:
        print(f"Column '{sort_column}' not found.")
        return df

    print(f'\nLevel {i}: Sorting by column "{sort_column}"')

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

        # Sort this cluster
        sorted_cluster = quantum_sort_cluster(cluster_df, sort_column)

        # Recursively sort the next column (if available)
        next_sorted_cluster = cluster_based_quantum_sort(pd.DataFrame(sorted_cluster), Pcols, i=i+1)
        all_sorted.append(next_sorted_cluster)

        print(f"  ✓ Completed Cluster {cluster_id}")

    # Merge and sort by current column
    merged_df = pd.concat(all_sorted, ignore_index=True)
    final_sorted_df = merged_df.sort_values(by=sort_column,ascending=order).reset_index(drop=True)


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
    app.run(debug = False)

