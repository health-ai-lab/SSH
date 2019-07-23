# Semi-Supervised Hierarchical Classification

## How to cite

M. Mirtchouk, D. McGuire, A. L. Deierlein, and S. Kleinberg. Automated Estimation of Food Type from Body-worn Audio and Motion Sensors in Free-Living Environments. In: *MLHC*, 2019

## Overview:

This code implements the semi-supervised top-down hierarchical classification technique described in the paper above. This approach takes a dataset with detailed labels and one with coarse labels and uses the strongly labeled data to provide more granular labels for the weakly labeled data. The approach further allows for hierarchical labels.
The primary application for the method is annotatation and classification of free-living dietary data at the level of intakes. Thus at test time an intake of "steak" could be classified as "steak" or as "beef" or "meat" depending on the classifier's confidence. and correct at level 2. 
For training, strong labels are at the level of intakes (e.g. an intake is steak) while weak ones are at the level of meals (e.g. meal contained steak and potato, but it unknown which bites are which). 
We further assume an ontology is provided, which contains paths such as: protein->meat->beef->steak. In our ontology format:
 protein,beef  
 beef,meat  
 meat,steak


## Using the code

### Preparation:

1. Food ontology data format: ensure food ontology data is a single CSV file with each row being a parent child relationship (such as a food) of the format: A,B where B is a child of A.

2. Lab/Free-Living alldata format: ensure that all Lab and Free-Living data (all subjects and sessions concatenated together) is in the form of a pickle file with the data in the form X,Y where X are the features and Y is the food or meal. Meals with multiple foods should be underscore separated (A\_B\_C). It should be placed in path/f/comb_allmeal.pkl or path/l/comb_allmeal.pkl accordingly.

3. Lab/Free-Living individual meal data format: ensure the Lab and Free-Living meal data is in the form of a pickle file with the data in the form X,Y where X are the features and Y is the food or meal. Meals with multiple foods should be underscore seperated (A\_B\_C). It should be placed in path/f/comb\_subj_sess\_meal.pkl or path/l/comb\_subj\_sess\_meal.pkl accordingly. If there are multiple of the same meal in the same subject and session, add a 1 or 2 at the end of the file for distinction.

4. Prepare a directory to save the two files: savedIntakes.pkl and labeledDatafn.pkl is just the path you where you would like to save the files (default is in path/fl/)

5. Choose a parameter setting for how often [every N samples] you would like to retrain your Random Forest (retrainR) and the minimum amount of intakes per class you would like to end up with (threshold). We used retrainR=50, and threshold=250, but values will depend on the amount of data you have and would like to generate


### Usage with detailed parameters:

Usage: python semisuper\_hierarchical\_classification.py (freeliving|lab) subj sess sensor\_combination meal mode path ontologyfn [savedIntakesfn labeledDatafn retrainR threshold] (the last 4 parameters are optional)

1. freeliving: to hierarchically label the Free-Living data; lab: is to  hierarchically label the Lab data. [can be shorthand to f or l]
2. subj: The subject number that you would like to test [Ex:02,102]
3. sess: The session number that you would like to test [Ex:0000,0001]
4. Sensor combinations are: AGRL or any combination of them [Ex:AGRL,AGR,AGL,...A,G,R,L]
4. meal: The meal you would like to leave out (multiple separated by underscores) [Ex:sandwich,pork\_rice\_salad]
5. mode: Leave One Meal Out (LOMO), Leave One Half Meal Out (LOHMO) [Ex:LOMO|LOHMO]
6. path: root path (make sure to add / at the end) [Ex:/data/]
7. Parameters ending in fn are for file name locations (without the beginning path) [Ex:food_ontology.csv,savedIntakesfn.pkl,labeledDatafn.pkl]. The food_ontology must be created by you, savedIntakes.pkl and labeledDatafn.pkl is just the path you where you would like to save the files
8. retrainR: How often to retrain the Random Forest, threshold is the minimum amount of samples  [Ex:50]
9. threshold: Minimum amount of intakes per class [Ex:250]

### To run the code:

python semisuper\_hierarchical\_classification.py f 102 0000 AGRL sandwich LOMO /data/ food_ontology.csv savedIntakesfn.pkl labeledDatafn.pkl 50 250

### File format examples

food\_ontology.csv example

A,B  
A,C  
A,D  
B,E  
B,F  
F,G  
C,H

Explanation: The parent is A who has 3 children B,C, and D. B has 2 children E and F. F has a child G. C has a child H. Therefore: D,E,G, and H are leaf nodes. 

path/lab/comb\_alldata.pkl and path/fl/comb\_alldata.pkl example  

(assuming 5 features)

X=[ [A,B,C,D,E],  
[A2,B2,C2,D2,E2],  
[A3,B3,C3,D3,E3], ... ]  

Y=[ "sandwich",  
"pork\_rice\_salad",  
"rice", ... ]  

Generate Pickle Example:  
alldatafn = path+fl+"/"+comb+"\_alldata.pkl"  
pickle.dump([X,Y],open(alldatafn, "wb"))  

path/lab/comb\_subj\_sess\_meal.pkl and path/fl/comb\_subj\_sess\_meal.pkl example  

(assuming 5 features)  

X=[ [A,B,C,D,E],  
[F,G,H,I,J],  
[K,L,M,N,O], ... ]   

Y=[ "sandwich",  
"sandwich",  
"sandwich", ... ]  

Generate Pickle Example:  
mealdatafn = path+fl+"/"+comb+ "\_" +subj + "\_" + sess + "\_"+meal+".pkl"  
pickle.dump([X,Y],open(mealdatafn, "wb"))

