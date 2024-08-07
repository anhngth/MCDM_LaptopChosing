import numpy as np
import pandas as pd

#Pairwise Comparison Matrix
criteriaDict = {
            1: "Durability",
            2: "Performance",
            3: "Price",
            4: "Storage",
            5: "Design",
            6: "Size"}

alternativesName = ["Dell", "Mac", "HP", "ASUS", "Lenovo"]

def load(excel_file, sheet_name, reshape_shape, start_cell, end_cell):
    start_row, start_col = int(start_cell[1:]), ord(start_cell[0]) - ord('A')
    end_row, end_col = int(end_cell[1:]), ord(end_cell[0]) - ord('A')
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=start_row - 1, nrows=end_row - start_row + 1, usecols=range(start_col, end_col + 1), engine='openpyxl')
    data = np.array(df).reshape(reshape_shape)
    return data

excel_file = "D:\PYTHON CODE\MCDM\FAHP.xlsx"
sheet_name = 'Trang tính2'
reshape_shape = (5, 5, 3)

crxcr   = load(excel_file, sheet_name, reshape_shape=(6,6,3), start_cell="B3", end_cell="S8")
alt_cr1 = load(excel_file, sheet_name, reshape_shape, start_cell="B14", end_cell="P18")
alt_cr2 = load(excel_file, sheet_name, reshape_shape, start_cell="B23", end_cell="P27")
alt_cr3 = load(excel_file, sheet_name, reshape_shape, start_cell="B32", end_cell="P36")
alt_cr4 = load(excel_file, sheet_name, reshape_shape, start_cell="B41", end_cell="P45")
alt_cr5 = load(excel_file, sheet_name, reshape_shape, start_cell="B50", end_cell="P54")
alt_cr6 = load(excel_file, sheet_name, reshape_shape, start_cell="B59", end_cell="P63")

alt = np.stack((alt_cr1, alt_cr2, alt_cr3, alt_cr4,alt_cr5, alt_cr6 ))

def isConsistent(matrix, printComp=True):
    RI = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }

    mat_len = len(matrix)
    midMatrix = np.zeros((mat_len, mat_len))
    for i in range(mat_len):
        for j in range(mat_len):
            midMatrix[i][j] = matrix[i][j][1]
    if(printComp): print("Mid-value matrix: \n", midMatrix, "\n")

    eigenvalue = np.real(np.linalg.eigvals(midMatrix))
    lambdaMax = max(eigenvalue)
    if(printComp): print("Eigenvalue: ", eigenvalue)
    if(printComp): print("LambdaMax: ", lambdaMax)
    if(printComp): print("\n")

    RIValue = RI[mat_len]
    if(printComp): print("R.I. Value: ", RIValue)

    CIValue = (lambdaMax-mat_len)/(mat_len - 1)
    if(printComp): print("C.I. Value: ", CIValue)

    CRValue = CIValue/RIValue
    if(printComp): print("C.R. Value: ", CRValue)

    if(printComp): print("\n")
    if(CRValue<=0.1):
        if(printComp): print("Matrix reasonably consistent, we could continue")
        return True
    else:
        if(printComp): print("Consistency Ratio is greater than 10%, we need to revise the subjective judgment")
        return False
    
def pairwiseComp(matrix):
    matrix_len = len(matrix)

    #calculate fuzzy geometric mean value
    geoMean = np.zeros((len(matrix),3))

    for i in range(matrix_len):
        for j in range(3):
            temp = 1
            for tfn in matrix[i]:
                temp *= tfn[j]
            temp = pow(temp, 1/matrix_len)
            geoMean[i,j] = temp
    
    print("Fuzzy Geometric Mean Value: \n", geoMean, "\n")
    #calculate the sum of fuzzy geometric mean value
    geoMean_sum = np.zeros(3)
    for row in geoMean:
        geoMean_sum[0] += row[0]
        geoMean_sum[1] += row[1]
        geoMean_sum[2] += row[2]
    
    print("Fuzzy Geometric Mean Sum:", geoMean_sum, "\n")
    
    #calculate weights
    weights = np.zeros(matrix_len)
    for i in range(len(geoMean)):
        temp = 0
        for j in range(len(geoMean[0])):
            temp += geoMean[i,j]*(1/geoMean_sum[(3-1)-j])
        weights[i] = temp 
    
    print("Weights: \n", weights, "\n")
    #caculate normaized weights
    normWeights = np.zeros(matrix_len)
    weights_sum = np.sum(weights)
    for i in range(matrix_len): 
        normWeights[i] = weights[i]/weights_sum
    
    print("Normalized Weights: ", normWeights,"\n")
    return normWeights

def pairwiseComp(matrix):
    matrix_len = len(matrix)

    #calculate fuzzy geometric mean value
    geoMean = np.zeros((len(matrix),3))

    for i in range(matrix_len):
        for j in range(3):
            temp = 1
            for tfn in matrix[i]:
                temp *= tfn[j]
            temp = pow(temp, 1/matrix_len)
            geoMean[i,j] = temp
    
    print("Fuzzy Geometric Mean Value: \n", geoMean, "\n")
    #calculate the sum of fuzzy geometric mean value
    geoMean_sum = np.zeros(3)
    for row in geoMean:
        geoMean_sum[0] += row[0]
        geoMean_sum[1] += row[1]
        geoMean_sum[2] += row[2]
    
    print("Fuzzy Geometric Mean Sum:", geoMean_sum, "\n")
    #calculate weights
    weights = np.zeros(matrix_len)

    for i in range(len(geoMean)):
        temp = 0
        for j in range(len(geoMean[0])):
            temp += geoMean[i,j]*(1/geoMean_sum[(3-1)-j])
        weights[i] = temp 
    
    print("Weights: \n", weights, "\n")
    #caculate normaized weights
    normWeights = np.zeros(matrix_len)
    weights_sum = np.sum(weights)
    for i in range(matrix_len): 
        normWeights[i] = weights[i]/weights_sum
    
    print("Normalized Weights: ", normWeights,"\n")
    return normWeights


def FAHP(crxcr, alt, alternativesName):
    crxcr_cons = isConsistent(crxcr, False)
    if(crxcr_cons):
        print("Criteria X criteria comparison matrix reasonably consistent, we could continue")
    else: 
        print("Criteria X criteria comparison matrix consistency ratio is greater than 10%, we need to revise the subjective judgment")
        
    for i, alt_cr in enumerate(alt):
        isConsistent(alt_cr, False)
        if(crxcr_cons):
            print("Alternatives X alternatives comparison matrix for criteria",i+1,"is reasonably consistent, we could continue")
        else: 
            print("Alternatives X alternatives comparison matrix for criteria",i+1,"'s consistency ratio is greater than 10%, we need to revise the subjective judgment")
    
    print("\n")
    
    print("Criteria X criteria ======================================================\n")
    crxcr_weights = pairwiseComp(crxcr)
    print("Criteria X criteria weights: ", crxcr_weights)
    
    
    print("\n")
    print("Alternative x alternative ======================================================\n")
    
    alt_weights = np.zeros((len(alt),len(alt[0])))
    for i, alt_cr in enumerate(alt):
        print("Alternative x alternative for criteria", criteriaDict[(i+1)],"---------------\n")
        alt_weights[i] =  pairwiseComp(alt_cr)
        
    print("Alternative x alternative weights:")
    alt_weights = alt_weights.transpose(1, 0)
    print(alt_weights)
    
    sumProduct = np.zeros(len(alt[0]))
    for i  in range(len(alt[0])):
        sumProduct[i] = np.dot(crxcr_weights, alt_weights[i])
        
    print("\n")
    print("RANKING =====================================================================\n")
    
    output_df = pd.DataFrame(data=[alternativesName, sumProduct]).T
    output_df = output_df.rename(columns={0: "Alternatives", 1: "Sum_of_Product"})
    output_df = output_df.sort_values(by=['Sum_of_Product'],ascending = False)
    output_df.index = np.arange(1,len(output_df)+1)
    
    print(output_df)
    return output_df
  
# isConsistent(alt[3])
output_df = FAHP(crxcr, alt, alternativesName)
