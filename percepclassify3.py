import sys;
import json;
import string;
from collections import defaultdict;

def readTestFile(fileName):
    testFile=open(fileName,encoding="utf8");
    return testFile;

def loadWeightsFromFile(fileName):
    f=open(fileName,encoding="utf8");
    lines=f.readlines();
    return json.loads(lines[0]), json.loads(lines[1]),json.loads(lines[2]),json.loads(lines[3]);

def removePunctuation(line):
    translator = str.maketrans('', '', string.punctuation);
    return line.translate(translator);

def getClass(classWeightMap,bias,wordMap):
    activation=0;
    for feature,inputValue in wordMap.items():
        activation+=classWeightMap.get(feature,0)*inputValue;
    activation+=bias;
    return 1 if(activation>=0) else -1;

def labelData(inputFileObj):
    outputFileObj = open('percepoutput.txt', 'w', encoding="utf8");
    for line in inputFileObj:
        wordCountMap = defaultdict(int);
        line = removePunctuation(line);
        doc = line.rstrip().split(" ");
        output=doc[0];
        words = doc[1:];
        for word in words:
            word = word.lower();
            wordCountMap[word] += 1;
        derivedClass1=getClass(class1WeightMap,biasClass1,wordCountMap);
        derivedClass1="True" if(derivedClass1==1) else "Fake"
        derivedClass2 =getClass(class2WeightMap,biasClass2,wordCountMap);
        derivedClass2 = "Pos" if (derivedClass2 == 1) else "Neg"
        output+=" "+derivedClass1 + " "+derivedClass2;
        print(output,file=outputFileObj);

class1WeightMap,biasClass1,class2WeightMap,biasClass2=loadWeightsFromFile(sys.argv[1]);
inputFileObj=readTestFile(sys.argv[2]);
labelData(inputFileObj);
inputFileObj.close();