import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.svm import SVC
import csv
import warnings
import sys
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
from datetime import datetime

training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')

# Combine datasets
data = pd.concat([training, testing], ignore_index=True)

cols= training.columns
cols= cols[:-1]

reduced_data = training.groupby(training['prognosis']).max()

# Near the top of the code, after loading data:
x = training[cols]
y = training['prognosis']

# Create and fit label encoder
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

# Train models with encoded labels
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x, y_encoded)

model = SVC()
model.fit(x, y_encoded)

                                            ##### the chat bot #####
# Open the file at the beginning of the script
output_file_path = "prescriptions.txt"
output_file = open(output_file_path, 'a')
current_date = datetime.now().strftime("%Y-%m-%d")


def append_prescription(prescription_details):
    output_file.write(prescription_details + "\n")
    output_file.flush()  # Ensure the content is written immediately

append_prescription(f"""
--------------------------------------------------------------------
          Medical Prescription (Generated Automatically)
--------------------------------------------------------------------
Date: {current_date}""")

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("\n")
        print("\n")
        print("=============  Consultation ==========")
        type_out(f"okay {name}, You should take the consultation from doctor. ")
        readn(f"okay {name}, You should take the consultation from doctor. ")
    else:
        print("\n")
        print("\n")
        print("=============  Consultation ==========")
        type_out(f"okay {name}, It might not be that bad but you should take precautions.")
        readn(f"okay {name}, It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

# Add a global variable to track TTS preference
enable_tts = False

def type_out(text, delay=0.03):
    """Prints text one character at a time with a delay."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()  # Adds a newline after the text.

def readn(nstr):
    """Read text aloud if TTS is enabled."""
    if enable_tts:  # Check the global variable
        engine = pyttsx3.init()
        engine.setProperty('voice', "english+f5")
        engine.setProperty('rate', 130)
        engine.say(nstr)
        engine.runAndWait()
        engine.stop()

def getInfo():
    """Get user information and set TTS preference."""
    global enable_tts  # Access the global variable
    global name
    type_out("""Disclaimer:
            This chatbot is a school project designed for educational purposes. While it aims to provide general 
         health-related information, it is not a substitute for professional medical advice, diagnosis, or treatment.
                    Always consult a qualified healthcare provider for any medical concerns. 
                        The information provided by this chatbot may contain inaccuracies."""+"\n")
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    type_out("\nPlease enter your Name? :")
    name = input("")
    type_out(f"Hello, {name}!")
    append_prescription(f"Patient Name: {name}"+"\n")

    # Ask for TTS preference
    while True:
        type_out("\nWould you like the responses to be read aloud? (yes/no): ")
        tts_preference = input("").strip().lower()
        if tts_preference in ['yes', 'no']:
            enable_tts = (tts_preference == 'yes')
            break
        else:
            type_out("Please respond with 'yes' or 'no'.")

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
    
def sec_predict(symptoms_exp):
    symptoms_dict = {symptom: index for index, symptom in enumerate(x)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1
    input_vector_df = pd.DataFrame([input_vector], columns=x.columns)
    pred = model.predict(input_vector_df)
    return le.inverse_transform(pred)  # Convert back to original label

def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        type_out(str("\nEnter the symptom you are experiencing  \t    ->"))
        readn("Enter the symptom you are experiencing")
        disease_input = input("")
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            type_out("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                type_out(str(num)+")"+str(it), 0.006)
            if num!=0:
                type_out(f"Select the symptom you mean from the list (0 - {num}):  ")
                readn("Select the symptom you mean from the list")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
            # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            type_out("Enter valid symptom.")
            readn("Enter valid symptom.")
    while True:
        try:
            type_out("Okay. From how many days ? : ")
            readn("From how many days")
            num_days=int(input(""))
            break
        except:
            type_out("Enter valid input.")
            readn("Enter valid input.")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list = list(symptoms_present)
            # for hh in dis_list:
                # append_prescription(hh+"\n")
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )

            type_out("\n"+" I will ask you about some related symptoms now")
            readn("I will ask you about some related symptoms now")
            type_out("are you having:")
            readn("are you having:")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                type_out(str(syms)+"? : ")
                readn(str(syms))
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        type_out("provide proper answers i.e. (yes/no) : ")
                        readn("provide answer with yes or no ")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                type_out("\n"+"You may have "+str(present_disease[0])+"\n")
                append_prescription("You may have "+str(present_disease[0])+"\n")
                readn("You may have "+str(present_disease[0]))
                type_out(description_list[present_disease[0]]+"\n", 0.0003)
                append_prescription(description_list[present_disease[0]]+"\n")

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                type_out("\n"+"You may have "+str(present_disease[0])+"or "+str(second_prediction[0])+"\n")
                append_prescription("You may have "+str(present_disease[0])+"or "+str(second_prediction[0]))
                readn("You may have "+str(present_disease[0])+"or "+str(second_prediction[0]))
                type_out(description_list[present_disease[0]]+"\n", 0.0003)
                append_prescription(description_list[present_disease[0]]+"\n")
                type_out(description_list[second_prediction[0]]+"\n", 0.0003)

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            type_out("Take following measures : ")
            append_prescription("Take following measures : "+ "\n")
            for  i,j in enumerate(precution_list):
                type_out(str(i+1)+")"+str(j))
                append_prescription(str(i+1)+")"+str(j))


    recurse(0, 1)

getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf1,cols)
append_prescription("""
--------------------------------------------------------------------
    Stay healthy, and dont hesitate to seek medical advice!
--------------------------------------------------------------------
""")
                    
output_file.close()
print("----------------------------------------------------------------------------------------")