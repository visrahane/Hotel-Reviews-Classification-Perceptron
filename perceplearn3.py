import sys;
import math;
import json;
import string;
from collections import defaultdict;

# list ds - {id,class1,class2,map of wordCount}
sentenceFeatureClassList = [];
vocabSet=set();

def readFile(fileName):
    inputFileObj = open(fileName, encoding="utf8");
    return inputFileObj;


def saveToFile(class1WeightMap,biasClass1,class2WeightMap,biasClass2):
    f = open("vanillamodel.txt", "w");
    f.write(json.dumps(class1WeightMap));
    f.write("\n");
    f.write(json.dumps(biasClass1));
    f.write("\n");
    f.write(json.dumps(class2WeightMap));
    f.write("\n");
    f.write(json.dumps(biasClass2));

    f.close()


def removePunctuation(line):
    translator = str.maketrans('', '', string.punctuation);
    return line.translate(translator);


def constructTokens(inputFileObj):
    i = 0;
    for line in inputFileObj:
        # create wordCountMap
        wordCountMap = defaultdict(int);
        sentenceFeatureClassListRow=[];
        line = removePunctuation(line);
        doc = line.rstrip().split(" ");
        sentenceFeatureClassListRow.append(doc[0]);  # set id
        sentenceFeatureClassListRow.append(-1) if (doc[1] == "Fake") else sentenceFeatureClassListRow.append(1);  # set class1
        sentenceFeatureClassListRow.append(-1) if (doc[2] == "Neg") else sentenceFeatureClassListRow.append(1);   # set class2
        words = (doc[3:]);
        for word in words:
            word = word.lower();
            if (word in stopList):
                continue;
            vocabSet.add(word);
            wordCountMap[word] += 1;
        sentenceFeatureClassListRow.append(wordCountMap);
        sentenceFeatureClassList.append(sentenceFeatureClassListRow);
        i += 1;

def calculateWordProbs(classWordMap):
    print("hi");


def trainPerceptron(classNo):
    #init
    classWeightMap = defaultdict(int);
    for key in vocabSet:
        classWeightMap[key]=0
    bias=0;

    for iteration in range(21):
        for doc in sentenceFeatureClassList:
            activation=0;
            for featureX,weight in classWeightMap.items():
                activation+=weight*doc[3].get(featureX,0);
            activation+=bias;
            y=doc[classNo];
            if(y*activation<=0):
                for featureA,weightA in classWeightMap.items():
                    classWeightMap[featureA]=weightA+y*doc[3].get(featureA,0);
                bias += y;

    return classWeightMap,bias;

inputFileObj = readFile(sys.argv[1]);
stopList = [line.rstrip() for line in open("input/stop-words.txt", encoding="utf8")];
constructTokens(inputFileObj);
class1WeightMap,biasClass1=trainPerceptron(1);
class2WeightMap,biasClass2=trainPerceptron(2);
saveToFile(class1WeightMap,biasClass1,class2WeightMap,biasClass2);

# print("classCountMap:",classCountMap);
# print("priorClassProps:",priorClassProps);
# print("vocab:",vocabMap);
# print("classWordMap:",classWordMap1);
# print("classWordMap:",classWordMap2);
