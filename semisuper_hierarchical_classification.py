from collections import Counter
from collections import defaultdict
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier

import copy
import math
import numpy as np
import os
import pickle
import random
import sys

""" 

Does majority voting based on the weights (second item in tuple). 
Also weighs the original answer oans by oweight
Needweight is useful when you need the weight along with the maximal value

arr: array of tuples [('bread',.6),('drink',.25),('meat',.15)]
oans: prediction from classifier 'drink'
oweight: weight of predicted answer to be weighted more (could be 1 so it is the same as the rest)
needweight: boolean whether to include the weight of the maximal element

"""

def maj_vote(arr,oans,oweight=1,needweight=False):
        d={}  #dictionary to store all weights
        for item in arr:
                if item[0] in d:
                        d[item[0]]+=float(item[1])
                else:
                        d[item[0]]=float(item[1])
        if d.get(oans) is not None:  #weighing original answer by oweight to settle ties and weigh it more
                d[oans]=d[oans]*oweight
        maxval=max(d.iteritems(), key=operator.itemgetter(1))
        if needweight:  #return the type with the weight
                return maxval
        return maxval[0]

""" 

Checks if a is a decendant of b
ddepth is the depth of each node like ddepth['food']=0 and ddepth['liquid']=1,etc [will be filled in]
p is the parent of every node

"""

def isDes(a,b,p,ddepth):
     #is b a des of a (a is ansestor)
    if ddepth[b]<ddepth[a]:
    	return False
    for i in range(ddepth[b]-ddepth[a]):
    	b=p[b]
    if a==b:
    	return True
    return False

""" 

Fills the depth of each food
f: current food
d: disctionary of all foods
ddepth: the depth of each node like ddepth['food']=0 and ddepth['liquid']=1,etc [will be filled in]

"""

def fillDepth(f,d,ddepth):
	if f in d:
		for q in d[f]:
                	ddepth[q]=ddepth[f]+1
                	fillDepth(q,d,ddepth)

""" 

Imports an ontology in the form of a 2 column seperated list
first,second means that second is a decendant of first ex: meat,beef
all level 1 foods must have 'food' as a parent: food,NAMEOFFOOD ex: food,liquid 
all comments beggining with  # are ignored
ontofn: path to ontology

"""

def importOnto(ontofn):
        f=open(ontofn,"r")
        f1=f.readlines()
        d={}
        p={}
        for x in f1:
                if len(x)<=1 or x[0]==" #":
                        continue
                a=x.split(",")
		a[1]=a[1][:-1] #take out the \n
                if a[0] in d:
                        d[a[0]].append(a[1])
                else:
                        d[a[0]]=[a[1]]
                p[a[1]]=a[0]
        ddepth={}
        ddepth['food']=0
        fillDepth('food',d,ddepth)
        highestLevel=0  #init to find the longest depth
        for kdep in ddepth:
                if ddepth[kdep]>highestLevel:
                        highestLevel=ddepth[kdep]
        return d,p,ddepth,highestLevel


""" 

Saved the newly created intakes
mydict:newly created intakes
savedintakesfn: Path to savedIntakes

"""

def save_savedIntakes(mydict,savedintakesfn):
  with open(savedintakesfn,'w') as f:
    pickle.dump(mydict,f)


""" 

Load the saved pickle (if it exists)
savedintakesfn: Path to savedIntakes

"""

def load_savedIntakes(savedintakesfn):
  with open(savedintakesfn,'r') as f:
    return pickle.load(f)

""" 

Loads all the Lab data, all FL data, or each subj,sess pair
datafn: Path to the data

"""

def load_data(datafn):
  with open(labdatafn,'r') as f:
    return pickle.load(f)

""" 

Extracts the X and Y associated for currfood
XX: Fetaures
YY: Labels for the features
currfood: Food that is to be used

"""
def getfooddata(XX,YY,currfood):
	tempX=[]
	tempY=[]
	for i in range(len(YY)):
		if YY[i]==currfood:
			tempX.append(XX[i])
			tempY.append(YY[i])
	return np.array(tempX),np.array(tempY)

""" 

Using normal distributions, generate more food intakes based on threshold which is thresh

XX: Fetaures
YY: Labels for the features
ontofn: path to ontology
savedIntakes: Dictionary of savedIntakes (originally empty, but fills up as it's used)
usesaved: Boolean whether to save the newly created intakes into savedIntakes
Create a new classifier every new retrainR intakes

"""

def createMoreIntakes(XX,YY,ontofn,thresh=250,retrainR=50,savedIntakes={},usesaved=False):
	lenFeat=len(XX[0])
	lenoldXX=len(XX)  # use for later to save newly created intakes
	foodList=[]
	badFoodList=[]
	dd,p,ddepth,highestLevel=importOnto(ontofn)

	if usesaved:
		 #Make the algorithm faster by re-using some intakes
		mylist=Counter(YY)
		for food in mylist:
			if mylist[food]<thresh:
				foodList.append(food)  #add the food if its above a certain threshold
			else:
				badFoodList.append(food)  #add the food if its blow a certain threshold

		for keyfood in savedIntakes.keys():
			nextkey=False
			for food in badFoodList:
				if isDes(food,keyfood,p,ddepth) and not nextkey:
					 #This means that food is a parent of keyfood and can be used
					if thresh<len(savedIntakes[q]['X']):
						XX.append(savedIntakes[q]['X'][:thresh])
						tempYY=len(savedIntakes[q]['X'][:thresh])*[food]
						YY.append(tempYY)
						nextkey=True
					else:  #not enough and use all the saved intakes
						XX.append(savedIntakes[q]['X'])
						tempYY=len(savedIntakes[q]['X'])*[food]
						YY.append(tempYY)

	 #below we recalculate as we could have added some from savedIntakes
	mylist=Counter(YY)
	foodList=[]
	badFoodList=[]
	for food in mylist:
		if mylist[food]<thresh:
			foodList.append(food)  #add the food if its below a certain threshold (not enough)
		else:
			badFoodList.append(food)  #add the food if its above the threshold as its done
	savefoodList=copy.copy(foodList)

	lenFoods=len(foodList)
	savelenFoods=len(foodList)  # for use in the while loop
	newXX=[]
	newYY=[]
	for i in range(len(YY)):
		 #newXX and newYY are the food features that are not done and need to be generated
		if YY[i] not in badFoodList:
			newXX.append(XX[i])
			newYY.append(YY[i])

	lennewYY=len(newYY)
	lenbfl=len(YY)-lennewYY  # length of the lab food list
	myfooddict={}
	for f in foodList:
		myfooddict[f]=True  #True means it needs to be generated

	curNum=0  # Counter to retrain classifier every retrainR intakes
	while (len(XX)-lenbfl)!=(savelenFoods*thresh):
		r=np.random.randint(0,lenFoods)  #choose a food to add to the list
		food=foodList[r]
		curX=[]
		curY=[]
		for i in range(len(YY)):
			tempX=[]
			if YY[i]==food:
				 # get all data associated to the random food
				for j in range(len(XX[i])):
					tempX.append(XX[i][j])
				tempX=np.array(tempX).astype(np.float)
				curX.append(tempX)
				curY.append(YY[i])
		curY=np.array(curY)
		YYcountfood=len(curY)  #amount of current foods in list
		if YYcountfood>=thresh:
			if myfooddict[food]:
				if food in foodList:
					foodList.remove(food)
					lenFoods=len(foodList)
				myfooddict[food]=False  #False means we are done with the food
			continue  # We are done with the food: food
		newX=[]
		allmeans=np.mean(curX,axis=0,dtype=np.float64)
		allstds=np.std(curX,axis=0,dtype=np.float64)
		for i in range(lenFeat):
			if allstds[i]==0:
				allstds[i]=.01  # standard deviation 0 should be not be used
			newpoint=np.random.normal(allmeans[i], allstds[i],1)[0]
			newX.append(newpoint)
		newX=np.array([newX])  # make 2d for prediction

		if curNum%retrainR==0:  #every retrainR create a new classifier
			numcores=10  #Amount of cores to use
			T=lenFoods*2
			c=RandomForestClassifier(T,n_jobs=numcores)  #Make n_jobs more to make it faster
        	c.fit(newXX,newYY)
        	finalans=c.predict(newX)
		if finalans[0]==food:  #If the generated is equal to the RF prediction
			XX.append(newX)
			YY.append(food)
			newXX.append(newX)
			newYY.append(food)
			curNum+=1  #increment counter by 1
	if usesaved:
		for i in range(len(savefoodList)):
			if savefoodList[i] not in savedIntakes:
				tempsIX,tempsIY=getfooddata(XX[lenoldXX:],YY[lenoldXX:],savefoodList[i])
				savedIntakes[foodList[i]]['X']=tempsIX
				savedIntakes[foodList[i]]['Y']=tempsIY
			else:
				tempsIX,tempsIY=getfooddata(XX[lenoldXX:],YY[lenoldXX:],savefoodList[i])
				savedIntakes[foodList[i]]['X'].append(tempsIX)
				savedIntakes[foodList[i]]['Y'].append(tempsIY)
	return XX,YY,savedIntakes

""" 

arr: array of tuples [('bread',.6),('drink',.25),('meat',.15)]
N: Amount of standard deviations away (default is 1)

"""

def testsig(arr,N=1):
	if len(arr)<=1:
		return True
	arr.sort(reverse=True)  #sort the values so arr[0] is biggest
	if (arr[0]-arr[1])<(1/3):
		return False
	numpyarr=np.array(arr)
	meanarr=np.mean(numpyarr, axis=0)
	stdevarr=np.std(numpyarr, axis=0)
	if numpyarr[0]>meanarr+(N*stdevarr):
		return True
	return False

""" 

Returns the Y_train at level 'level'
p is the parent dictionary so p['beef']='meat'
Y_train: Lab labels to be made into a different level

"""

def make_super(Y_train,p,level=1):
	Y_train_super=[]
	for food in Y_train:
		if ddepth[food]==level:
			Y_train_super.append(food)
			continue
		foodsave=food
		while ddepth[foodsave]!=level and ddepth[foodsave]!=0:
			foodsave=p[foodsave]
		Y_train_super.append(foodsave)
	return Y_train_super

""" 

Gets the training set without the test set
X_train: Lab features that are labeled
Y_train: Lab labels
LabelX: Features to be labeled
LabelY: A set of possible foods (A_B_C)

"""

def getcleandata(X_train,Y_train,X_test,Y_test):
	X_train2=[]
	Y_train2=[]
	for i in range(len(X_train)):
		if not np.any(np.all(np.isin(X_test,X_train[i],True),axis=1)):
			X_train2.append(X_train[i])
			Y_train2.append(Y_train[i])
	X_train2=np.array(X_train2)
	Y_train2=np.array(Y_train2)
	return X_train2,Y_train2,X_test,Y_test

""" 

Labels the FL data (which is the form A_B_C with either A,B, or C or it's parents)
X_train: Lab features that are labeled
Y_train: Lab labels
LabelX: Features to be labeled
LabelY: A set of possible foods (A_B_C) which could finally be either A,B,C, it's parents, or None
ontofn: path to ontology
Create a new classifier every new retrainR intakes
savedIntakesfn path where you saved the new intakes
labeledDatafn: path where you want to save the labeled data

"""

def label_intakes(X_train,Y_train,LabelX,LabelY,ontofn,thresh=250,retrainR=50,savedIntakesfn='savedIntakesfn.pkl',labeledDatafn='labeledData.pkl'):
	dd,p,ddepth,highestLevel=importOnto(ontofn)
	myLabelIntake=[]
	LabelY_save=copy.deepcopy(LabelY)
	myLenLabelX=len(LabelX)
	labelNum = list(range(myLenLabelX))
	random.shuffle(labelNum)  #shuffles the order to label at
	curNum=0
	for indexArr in range(myLenLabelX):
		mypred=""
		mypredSig=[]  #all the significant ones
		randi=labelNum[indexArr] #we are labeling number randi in a random order
		currLevel=1 #start at top level
		bnotdone=True
		
		while bnotdone:
			if len(mypred)>=1 and mypred in LabelY_save[randi]:
        	    		X_train.append(LabelX[randi]) #we found a good food type that is not repeated
	            		Y_train.append(mypred)
				bnotdone=False
				break
			Label_split=LabelY[randi].split("_")
			Label_split=list(set(Label_split)) #make it unique
	        	if len(Label_split)==1:
        	   	 	X_train.append(LabelX[randi]) #we found a good food type that is not repeated
				if mypred=="": #if one is not yet set
					mypred=Label_split[0]
			    	Y_train.append(mypred)
				bnotdone=False
                		break
			LabelX2,LabelY2=makeHigherLevel(dd,p,ddepth,[LabelX[randi]],[LabelY[randi]],[LabelY[randi]],currLevel) #we are labeling number randi
			X_train2,Y_train2=makeHigherLevel(dd,p,ddepth,X_train,Y_train,LabelY2,currLevel)
			savedIntakes={}
			exists = os.path.isfile(savedIntakesfn)
			if exists:
				savedIntakes=load_savedIntakes(savedIntakesfn)
			X_train2,Y_train2,savedIntakes=createMoreIntakes(X_train2,Y_train2,ontofn,thresh,retrainR,savedIntakes,True)
			save_savedIntakes(savedIntakes,savedintakesfn)
			LabelX2=LabelX2[0] #make into 1D array
			LabelY2=LabelY2[0] #make into singleton
			isSig=True
			if len(X_train2)==0:
				bnotdone=False
				break  #should never happen
			if curNum%retrainR==0:
				numcores=10  #Amount of cores to use
				lenfoods=len(list(set(Y_train2)))
				T=lenFoods*2
				c=RandomForestClassifier(T,n_jobs=numcores)
        		c.fit(X_train2,Y_train2)
			a = c.predict_proba([LabelX2])[0]
			classname = c.classes_
			numMax=-1
			indexMax=-1
			aArr=[] #int array of the probability
			for indexNum in range(len(a)):
				aArr.append(int(a[indexNum]*100)/100)  #make it 2 decimal places
				if a[indexNum]>numMax:
					numMax=a[indexNum]
					indexMax=indexNum
			mypred=classname[indexMax]
			isSig=testsig(aArr)  # check if it is significant
			if isSig:
				newLabelY=""
	            		LYs=LabelY[q].split("_")
		                for f in LYs:
                		        if isDes(mypred,f,dd,p,ddepth):
                                		newLabelY+=f+"_"
                		newLabelY=newLabelY[:-1]  #delete the last _
				LabelY[randi]=newLabelY  #make the only labels the significant ones
				mypredSig.append([mypred,numMax])
			currLevel+=1
			if currLevel>highestLevel:
				if len(mypredSig)>=1 and mypred==[mypredSig[-1][0]]:  #choose the leaf node
					bnotdone=False
					break 
				elif len(mypredSig)>=1:
					mypred=max(mypredSig,key=itemgetter(1))[0]  #gets the max
					bnotdone=False
					break
				else:
					mypred=None  #Label it as nothing
					bnotdone=False
					break
		myLabelIntake.append(mypred)
	myLabelIntakeOrdered=[""]*len(LabelY_save)
	for imyNum in range(len(labelNum)):
		myNum=labelNum[imyNum]
		myLabelIntakeOrdered[myNum]=myLabelIntake[imyNum]
	X_train.append(LabelX)
	Y_train.append(myLabelIntakeOrdered)
	labeledData= {'LabelX':LabelX,'LabelY':myLabelIntakeOrdered,'X_train':X_train,'Y_train':Y_train}
	pickle.dump(labeledData, open(labeledDatafn, 'wb'))  #save the labeled data into pickle

""" 

Classifies the FL data (no looking at the answer!, took out the LabelY param [not needed])
Basically, the same as label_intakes without the LabelY variable as well as labeledDatafn which is not needed
ontofn: path to ontology
Create a new classifier every new retrainR intakes

"""

def classify_intakes(X_train,Y_train,X_test,ontofn,thresh=250,retrainR=50,savedIntakesfn='savedIntakesfn.pkl'):
	dd,p,ddepth,highestLevel=importOnto(ontofn)
	myLabelIntake=[]
	myLenX_test=len(X_test)
	curNum=0
	for indexArr in range(myLenX_test):
		mypred=""
		mypredSig=[]  #all the significant ones
		numi=labelNum[indexArr] #we are labeling number numi in a numerical order
		currLevel=1 #start at top level
		bnotdone=True
		predLabelY=None
		X_test2=X_test[numi]  #features we are trying to label
		while bnotdone:
			if len(mypred)==1:
				bnotdone=False
				break
			X_train2=copy.deepcopy(X_train)
			Y_train2=make_super(Y_train,p,currLevel)
			savedIntakes={}
			exists = os.path.isfile(savedIntakesfn)
			if exists:
				savedIntakes=load_savedIntakes(savedIntakesfn)
			X_train2,Y_train2,savedIntakes=createMoreIntakes(X_train2,Y_train2,ontofn,thresh,retrainR,savedIntakes,True)
			save_savedIntakes(savedIntakes,savedintakesfn)
			isSig=True
			if len(X_train2)==0:
				bnotdone=False
				break  #should never happen
			if curNum%retrainR==0:
				numcores=10  #Amount of cores to use
				lenfoods=len(list(set(Y_train2)))
				T=lenFoods*2
				c=RandomForestClassifier(T,n_jobs=numcores)
        		c.fit(X_train2,Y_train2)
			a = c.predict_proba([X_test2])[0]
			classname = c.classes_
			numMax=-1
			indexMax=-1
			aArr=[] #int array of the probability
			for indexNum in range(len(a)):
				aArr.append(int(a[indexNum]*100)/100)  #make it 2 decimal places
				if a[indexNum]>numMax:
					numMax=a[indexNum]
					indexMax=indexNum
			mypred=classname[indexMax]
			isSig=testsig(aArr)  # check if it is significant
			if isSig:
				mypredSig.append([mypred,numMax])
				predLabelY=mypred  #make the only labels the predicted one
			currLevel+=1 # go down one level
			if currLevel>highestLevel:
				if len(mypredSig)>=1 and mypred==[mypredSig[-1][0]]:  #choose the leaf node
					bnotdone=False
					break 
				elif len(mypredSig)>=1:
					mypred=max(mypredSig,key=itemgetter(1))[0]  #gets the max
					bnotdone=False
					break
				else:
					mypred=None  #Label it as nothing
					bnotdone=False
					break
		myLabelIntake.append(mypred)
	myLabelIntake=np.array(myLabelIntake)
	return myLabelIntake

""" 

Prints the results by level with total percent and accuracy and the overall intake accuracy

"""

def printresult(X_test,myans,ontofn):
	dd,p,ddepth,highestLevel=importOnto(ontofn)
	dCorrect={}
	dTotal={}
	for i in range(len(X_test)):
		if isDes(myans[i],X_test[i],p,ddepth):  #if b is a a des of a (a is ansestor)
			if ddepth[myans[i]] in dCorrect:
    				dCorrect[ddepth[myans[i]]]+=1
    		else:
    			dCorrect[ddepth[myans[i]]]=1
		if ddepth[myans[i]] in dTotal:
			dTotal[ddepth[myans[i]]]+=1
		else:
			dTotal[ddepth[myans[i]]]=1
	dsumTotal=sum(dTotal.itervalues())
	overalltotal=0
	print "        Percent total Level Accuracy"
	for i in range(1,max(dTotal)+1):
		overalltotal+=1.0*dCorrect[i]/dsumTotal
		print "Level " + str(i)+ " " + str(int(100.0*dTotal[i]/dsumTotal)/1.0) + " " + str(int(100.0*dCorrect[i]/dTotal[i])/1.0)  #make it 2 decimals
	print "Overall (intakes)",str(int(100.0*overalltotal)/1.0)

""" 

Converts all the Y_train to the level of currLevel and makes sure they are in currLabelY (or ancestors or currLabelY)
currLabelY: Labels in the form A_B_C for each food which have to be converted to level currLevel

"""

def makeHigherLevel(d,p,ddepth,X_train,Y_train,currLabelY,currLevel):

	 #Below is for when you are making a higher level of the intake you are labeling (length 1)
	if len(Y_train)==1 and Y_train[0]==currLabelY[0]:
		currY=Y_train[0]
		currY_split=currY.split("_")  #Split the meal into components
		newY=[]
		for cY in currY_split:
			if ddepth[cY]==currLevel:
				newY.append(cY)  #You are at the correct depth
				break
			else:
				while cY in p and ddepth[cY]>currLevel:
					cY=p[cY]  #Choose the parent and go one level up
				if currLevel>=ddepth[cY] and ddepth[cY]!=0:
					newY.append(cY)
		newY.sort()
		newY2='_'.join(newY)  #makes it into a single string
		return X_train,[newY2]
	newY_train=[]
	newX_train=[]

	for iCurr in range(len(Y_train)):
		currY=Y_train[iCurr]  #Current intake meal
		bGoodFood=True
		if currLabelY[0]!=None:
			myLabelY=currLabelY[0].split("_")
			bGoodFood=False  #make sure food is in the list of possible foods
			for mLY in myLabelY:
				if isinstance(currY, (list,)):
					currY=currY[0]  #should never happen
				if isDes(mLY,currY,d,p,ddepth):  # if its the first level, have all the food types
					bGoodFood=True  #food is in the list of possible foods!
        while bGoodFood:  #It's in the list of possible foods
            if ddepth[currY]==currLevel:
		newX_train.append(X_train[iCurr])
		newY_train.append(currY)
		bGoodFood=False
	    elif currLevel>ddepth[currY]:
		bGoodFood=False  # it doesnt go that far down
            else:
                currY=p[currY]  #Go to parent
		if isinstance(currY, (list,)):
			currY=currY[0]  #should never happen
		if currY=='food':
			bGoodFood=False
	return newX_train,newY_train

if __name__ == '__main__':

	if len(sys.argv)==9 or len(sys.argv)==13:
	    	fl=sys.argv[1].lower()[0]  #only keep f or l
	    	otherfl='l'
	    	if fl=='l':
	    		otherfl='f'
	    	subj=sys.argv[2]
	    	sess=sys.argv[3]
	    	comb=sys.argv[4]
	    	meal=sys.argv[5]
	    	mode=sys.argv[6].upper()
	    	path=sys.argv[7]
		if path[-1]!="/":
			path+="/" #make it a path with / at the end
		ontofn=path+sys.argv[8] #add the path
	    	if len(sys.argv)==13:
			savedIntakesfn=path+fl+"/"+comb+"_"+sys.argv[9] #add the path, fl, and combname
			labeledDatafn=path+fl+"/"+comb+"_"+sys.argv[10] #add the path, fl, and combname
			retrainR=int(sys.argv[11])
			thresh=int(sys.argv[12])
		else: #give them default values
			savedIntakesfn=path+fl+"/"+comb+"_"+"savedIntakes.pkl"
			labeledDatafn=path+fl+"/"+comb+"_"+"labeledData.pkl"
			retrainR=50
			thresh=250
		X_train,Y_train = load_data(path+otherfl+"/"+comb+"_alldata.pkl")
		LabelX,LabelY = load_data(path+fl+"/"+comb+"_alldata.pkl")
		X_test,Y_test = load_data(path+fl+"/"+comb+"_"+subj+"_"+sess+"_"+meal+".pkl")
		exists = os.path.isfile(labeledDatafn)
		if not exists:  #labels the data
			label_intakes(X_train,Y_train,LabelX,LabelY,ontofn,thresh=thresh,retrainR=retrainR,savedIntakesfn=savedIntakesfn,labeledDatafn=labeledDatafn)
		trainlabeldict=pickle.load(labeledDatafn)
		X_train=trainlabeldict['X_train']
		Y_train=trainlabeldict['Y_train']
		 #now that we have all the labels, we have to take out the labels corresponding to the food at hand
		if mode == "LOMO":
			X_train,Y_train,X_test,Y_test=getcleandata(X_train,Y_train,X_test,Y_test)
			myans=classify_intakes(X_train,Y_train,X_test,ontofn,thresh=thresh,retrainR=retrainR,savedIntakesfn=savedIntakesfn,labeledDatafn=labeledDatafn)
		elif mode == "LOHMO":
			X_train1,Y_train1,X_test1,Y_test1=getcleandata(X_train,Y_train,X_test[:len(X_test)/2],Y_test[:len(X_test)/2])
			myans1=classify_intakes(X_train1,Y_train1,Y_test1,ontofn,thresh=thresh,retrainR=retrainR,savedIntakesfn=savedIntakesfn,labeledDatafn=labeledDatafn)
			X_train2,Y_train2,X_test2,Y_test2=getcleandata(X_train,Y_train,X_test[len(X_test)/2:],Y_test[len(X_test)/2:])
			myans2=classify_intakes(X_train2,Y_train2,Y_test2,ontofn,thresh=thresh,retrainR=retrainR,savedIntakesfn=savedIntakesfn,labeledDatafn=labeledDatafn)
			myans=np.concatenate((myans1,myans2))
		printresult(Y_test,myans)
	else:
		print "Usage: semisuper_hierarchical_classification (freeliving|lab) subj sess sensor_combination meal mode path ontologyfn [savedIntakesfn labeledDatafn retrainR threshold] (the last 4 parameters are optional)"
		print "Example: python semisuper_hierarchical_classification f 102 0000 AGRL sandwich LOMO /data/ food_ontology.csv savedIntakesfn.pkl labeledDatafn.pkl 50 250"
