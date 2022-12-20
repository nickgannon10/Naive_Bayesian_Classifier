import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
  

pdf = pd.read_csv('/Users/nicholasgannon/Desktop/AI/A5/pdf.txt', sep =',', header = None)
data = pd.read_csv('/Users/nicholasgannon/Desktop/AI/A5/data.txt', sep =',', header = None)  
#convert the pdf into something that can be input into a Bayesian classifier 
def clean_pdf(pdf):
    pdf1 = pdf.T
    pdf2 = pdf1[0].append(pdf1[1]).reset_index(drop=True)
    pdf3 = pd.DataFrame(pdf2)
    classification = []
    for count, ele in enumerate(pdf3[0]):
        if count <= 399:
            classification.append("B")
        else:
            classification.append("A")
        
    pdf3["classification"] = classification   
    Xi = []
    Xi2 = []
    for i in range(400): 
        x = i * .5
        Xi.append(x)
        Xi2.append(x)

    Xi = pd.DataFrame(Xi) 
    Xi["second_temp"] = Xi2 
    Xi = Xi[0].append(Xi["second_temp"]).reset_index(drop=True)
    Xi = pd.DataFrame(Xi)
    pdf3["X_val"] = Xi
    
    pdf3["Xi_yi"] = pdf3[0] * pdf3["X_val"]
    return pdf3

def clean_turns_functional(clean): 
    X_quant = []
    for i in clean[0]:
        if i == 0: 
            X_quant.append([0]) 
        elif i <= .005: 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
        elif i <= .010: 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
        elif i <= .015: 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
        elif i <= .020: 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
        elif i <= .025: 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
        elif i <= .030: 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
        elif i <= .035: 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
        elif i <= .040: 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
        elif i < .45: 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist()) 
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())
            X_quant.append(clean['X_val'][clean[clean[0] == i].index.tolist()].tolist())  
        else: 
            X_quant.append([0])

    Xfin = []     
    for item in X_quant:
        Xfin.append((item[0]))

    pdf_functional = pd.DataFrame(Xfin)

    clas = []
    for i in range(len(pdf_functional)): 
        if i < 694: 
            clas.append(0)
        else: 
            clas.append(1)

    pdf_functional["class"] = clas
    return pdf_functional 
# P(Y=y) for all possible y
def priors(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior
# P(X=x|Y=y) using Gaussian dist.
def gaussian_liklihood(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    return p_x_given_y
# P(X=x1|Y=y)P(X=x2|Y=y)...P(X=xn|Y=y) * P(Y=y) for all y
def naive_bayes(df, X, Y):
    # feature names
    features = list(df.columns)[:-1]

    # priors
    prior = priors(df, Y)

    Y_pred = []
    # loop over all data 
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= gaussian_liklihood(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 

def extractDigits(lst):
    return list(map(lambda el:[el], lst))

def Predicted_Results(data):
    data1 = data.fillna(0)
    data2 = data1.T
    data3 = np.asarray(data2)
    data4 = []
    for element in range(10): 
        e = np.asarray(extractDigits(data3[:, element]))
        data4.append(e)

    data5 = np.asarray(data4)
    Y_pred = []
    for element in range(10):
        all_guesses = naive_bayes(pdf_functional, X = data5[element,:], Y='class')
        Y_pred.append(all_guesses)
    Y_pred_final = np.asarray(Y_pred)

    Final_Prob_Predictions = []
    Classification = []

    for i in range(10):
        Prob = np.sum(Y_pred_final[i,:])/300 
        Final_Prob_Predictions.append(Prob) 
        if Prob > .9:
            Classification.append("Plane")
        elif Prob <= .9 and Prob > .5:
            Classification.append("likely Plane, but Probability not decisive")
        elif Prob <= .5 and Prob >= .1: 
            Classification.append("likely Bird, but Probability not decisive")
        else: 
            Classification.append("Bird")

    
    Conglomerated_Results = {
        'Probability' : Final_Prob_Predictions,
        'Classification' : Classification   }
    Conglomerated_Results = pd.DataFrame(Conglomerated_Results)
    
    return Conglomerated_Results

   
print(">>> Running ---Naïve Recursive Bayesian classifier ---")
clean = clean_pdf(pdf=pdf)
pdf_functional = clean_turns_functional(clean)
print(Predicted_Results(data))






