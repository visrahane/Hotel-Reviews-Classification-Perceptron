import sys;
import math;
import json;
import string;
import random;
from collections import defaultdict;

# list ds - {id,class1,class2,map of wordCount}
sentenceFeatureClassList = [];
vocabSet=set();

def readFile(fileName):
    inputFileObj = open(fileName, encoding="utf8");
    return inputFileObj;


def saveToFile(fileName,class1WeightMap,biasClass1,class2WeightMap,biasClass2):
    f = open(fileName, "w");
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


def trainAvgPerceptron(classNo):
    # init
    classWeightMap = defaultdict(int);
    cachedWeightMap = defaultdict(int);

    # for key in vocabSet:
    #     classWeightMap[key]=0
    bias = 0;
    cachedBias = 0;
    c = 1;
    for iteration in range(21):
        random.shuffle(sentenceFeatureClassList);
        for doc in sentenceFeatureClassList:
            activation = 0;
            for featureX, input in doc[3].items():
                activation += input * classWeightMap.get(featureX, 0);
            activation += bias;
            y = doc[classNo];
            if (y * activation <= 0):
                for featureA, inputA in doc[3].items():
                    classWeightMap[featureA] = classWeightMap.get(featureA, 0) + y * inputA;
                    cachedWeightMap[featureA] = classWeightMap.get(featureA, 0) + y * inputA;
                bias += y;
                cachedBias += y * c;
            c += 1;

    for key in classWeightMap:
        classWeightMap[key] -= cachedWeightMap[key] / c;

    return classWeightMap, bias - cachedBias / c;

def trainVanillaPerceptron(classNo):
    # init
    classWeightMap = defaultdict(int);
    # for key in vocabSet:
    #     classWeightMap[key]=0
    bias = 0;

    for iteration in range(21):
        random.shuffle(sentenceFeatureClassList);
        for doc in sentenceFeatureClassList:
            activation = 0;
            for featureX, input in doc[3].items():
                activation += input * classWeightMap.get(featureX, 0);
            activation += bias;
            y = doc[classNo];
            if (y * activation <= 0):
                for featureA, inputA in doc[3].items():
                    classWeightMap[featureA] = classWeightMap.get(featureA, 0) + y * inputA;
                bias += y;


    return classWeightMap, bias;

inputFileObj = readFile(sys.argv[1]);
stopList = [line.rstrip() for line in open("input/stop-words.txt", encoding="utf8")];
constructTokens(inputFileObj);
class1WeightMap,biasClass1=trainVanillaPerceptron(1);
class2WeightMap,biasClass2=trainVanillaPerceptron(2);
class1AvgWeightMap,biasAvgClass1=trainAvgPerceptron(1);
class2AvgWeightMap,biasAvgClass2=trainAvgPerceptron(2);
saveToFile("averagedmodel.txt",class1AvgWeightMap,biasAvgClass1,class2AvgWeightMap,biasAvgClass2);
saveToFile("vanillamodel.txt",class1WeightMap,biasClass1,class2WeightMap,biasClass2);
# print("classCountMap:",classCountMap);
# print("priorClassProps:",priorClassProps);
# print("vocab:",vocabMap);
# print("classWordMap:",classWordMap1);
# print("classWordMap:",classWordMap2);
