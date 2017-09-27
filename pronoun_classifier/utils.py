from collections import Counter, defaultdict
import re, os, random, numpy


class SourceWord:
    def __init__(self, word, pos, dep):
  
        self.word = normalize(word)
        self.pos = pos
        self.dep = dep
        self.wordvec = None
        self.posvec = None
        self.depvec = None
        self.ivec = None
        self.vec = None

    def __str__(self):
        return '|'.join([self.word, self.pos, self.dep])

class TargetWord:
    def __init__(self, lemma, pos):
  
        self.lemma = normalize(lemma)
        self.pos = pos
        self.lemmavec = None
        self.posvec = None
        self.ivec = None
        self.vec = None

    def __str__(self):
        return '|'.join([self.lemma, self.pos])

class SentenceWords:
    def  __init__(self, rawSent, source):
        self.tokens = []
        words = filter(bool, rawSent.split(' '))
        for word in words:
            parts = word.split('|')
            if source:
                self.tokens.append(SourceWord(parts[0], parts[1], parts[2]))
            else:
                if re.match( r'REPLACE', parts[0]):
                    self.tokens.append(TargetWord(parts[0], "REPL"))
                else:
                    self.tokens.append(TargetWord(parts[0], parts[1]))
                

class PronounInstance:
    def  __init__(self, pClass, tPron, sPron, sIndex, tIndex, pHeadIndex):
        self.pClass = pClass
        self.tPron = tPron
        self.sPron = sPron
        self.sIndex = int(sIndex)
        self.tIndex = int(tIndex)
        self.pHeadIndex = int(pHeadIndex)-1  #compensate by 1, because of data file format

    def  __init__(self, line):
        parts = filter(bool, line.split(' '))
        if len(parts) < 5:
            print "problem with input example: ", line
        self.pClass = parts[0]
        self.tPron = parts[1]
        self.sPron = parts[2]
        self.sIndex = int(parts[3])
        self.tIndex = int(parts[4])
        self.pHeadIndex = int(parts[5])-1 #compensate by 1, because of data file format

    def printPron(self):
        print " ".join([self.pClass, self.tPron, self.sPron, str(self.pHeadIndex)])


class Sentence:
    def  __init__(self, sourceSent, targetSent):
        self.source = sourceSent
        self.target = targetSent
        self.pronouns = []

    def addPronounInstance(self, instance):
        self.pronouns.append(instance)
    
    def numPronouns(self):
        return len(self.pronouns)
 
# collects vocab by reading from data file
# used with online sampling
def vocabFromFile(file_name, filter_rare = -1):
    wordsCount = Counter()
    posCount = Counter()
    depCount = Counter()
    lemmaCount = Counter()
    tposCount = Counter()
    charsCount = Counter()

    pronFreqs = defaultdict(int)

    sentenceCount = 0
    exampleCount = 0

    with open(file_name, 'r') as pronFP:

        sentence = None
        for line in pronFP:
            parts = filter(bool, line.strip().split('\t'))
            if len(parts) == 2:  #sentence pair
                if sentence != None:  #store previous sentence, before reading a new one
                    wordsCount.update([token.word for token in sentence.source.tokens])
                    posCount.update([token.pos for token in sentence.source.tokens])
                    depCount.update([token.dep for token in sentence.source.tokens])
                    lemmaCount.update([token.lemma for token in sentence.target.tokens])
                    tposCount.update([token.pos for token in sentence.target.tokens])
                    for token in sentence.source.tokens:
                        charsCount.update(token.word)
                    sentence = None
                    
                sentenceCount += 1
                source = SentenceWords(parts[0], True)
                target = SentenceWords(parts[1], False)
                sentence = Sentence(source, target)
                            
            elif len(parts) == 1:  # pronoun instance
                exampleCount += 1
                pInstance = PronounInstance(line)
                #sentence.addPronounInstance(pInstance)
                pronFreqs['tot'] += 1
                pronFreqs[pInstance.pClass] += 1
            else:
                print "something wrong when reading the file ", file_name, " ", line, " ", len(parts)
                   
    if sentence != None:  #store last sentence
        wordsCount.update([token.word for token in sentence.source.tokens])
        posCount.update([token.pos for token in sentence.source.tokens])
        depCount.update([token.dep for token in sentence.source.tokens])
        lemmaCount.update([token.lemma for token in sentence.target.tokens])
        tposCount.update([token.pos for token in sentence.target.tokens])
        for token in sentence.source.tokens:
            charsCount.update(token.word)
        
    if filter_rare > 0:
        wordsCount = Counter(w for w in wordsCount.elements() if wordsCount[w] > filter_rare)
        lemmaCount = Counter(w for w in lemmaCount.elements() if lemmaCount[w] > filter_rare)

    print 'sentences read ', sentenceCount, " instances: ", exampleCount
        
    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), depCount.keys(), lemmaCount, {l: i for i, l in enumerate(lemmaCount.keys())}, tposCount.keys(), charsCount.keys(), pronFreqs)
    

# collects vocab by reading from already processed data (used as default)
def vocab(sentences, filter_rare = -1):
    wordsCount = Counter()
    posCount = Counter()
    depCount = Counter()
    lemmaCount = Counter()
    tposCount = Counter()
    charsCount = Counter()

    for sentence in sentences:
        wordsCount.update([token.word for token in sentence.source.tokens])
        posCount.update([token.pos for token in sentence.source.tokens])
        depCount.update([token.dep for token in sentence.source.tokens])
        lemmaCount.update([token.lemma for token in sentence.target.tokens])
        tposCount.update([token.pos for token in sentence.target.tokens])
        for token in sentence.source.tokens:
            charsCount.update(token.word)

    if filter_rare > 0:
        wordsCount = Counter(w for w in wordsCount.elements() if wordsCount[w] > filter_rare)
        lemmaCount = Counter(w for w in lemmaCount.elements() if lemmaCount[w] > filter_rare)

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), depCount.keys(), lemmaCount, {l: i for i, l in enumerate(lemmaCount.keys())}, tposCount.keys(), charsCount.keys())


#It|PRP|SBJ|2|'s 's|VBZ|ROOT|0|root hard|JJ|PRD|2|'s to|TO|EXTR|2|'s overstate|VB|IM|4|to the|DT|NMOD|7|destruction destruction|NN|OBJ|5|overstate .|.|P|2|'s    REPLACE_0 sein|VERB schwer|ADJ ,|. die|DET Zerstoerung|NOUN zu|PRT ueberbewerten|VERB .|.
#es es|PRON It 0 0 2

def read_prons(fh):
    sentenceCount = 0
    exampleCount = 0

    sentences = []

    for line in fh:
        parts = filter(bool, line.strip().split('\t'))
        if len(parts) == 2:  #sentence pair
            sentenceCount += 1
            source = SentenceWords(parts[0], True)
            target = SentenceWords(parts[1], False)
            sentences.append(Sentence(source, target))
                            
        elif len(parts) == 1:  # pronoun instance
            exampleCount += 1
            sentences[-1].addPronounInstance(PronounInstance(line))           
        else:
            print "something wrong when reading the file ", line, len(parts)
            
    print 'sentences read ', sentenceCount, " instances: ", exampleCount
    return sentences



def getDistribution(data):
	''' 
	data is a list of objects of type Sentence with two SentenceWords objects (source and target) and a list (pronouns) filled with PronounInstance objects
	    (list of Sentence) -> (dict)
	'''

	prons_dict = {}
	indexesindata = {}

	for s in data:
		for p in s.pronouns:
			if p.pClass in prons_dict:
				prons_dict[p.pClass] += 1
				indexesindata[p.pClass].append(data.index(s))
			else:
				prons_dict[p.pClass] = 1
				indexesindata[p.pClass] = [data.index(s)]
		
	return prons_dict, indexesindata



def getDistributionPercentage(data):
	''' 
	data is a list of objects of type Sentence with two SentenceWords objects (source and target) and a list (pronouns) filled with PronounInstance objects
	    (list of Sentence) -> (dict)
	'''

	prons_dict = {}
	for s in data:
		for p in s.pronouns:
			if p.pClass in prons_dict:
				prons_dict[p.pClass] += 1
			else:
				prons_dict[p.pClass] = 1
		
	# determine distribution in dev
	percentages = {}
	total_dev = sum(prons_dict.values())
	for key in prons_dict:
		percentages[key] = float(prons_dict[key])/float(total_dev)

	return percentages


def getPronounProbabilities(pronFreqs, pronPercent, sampleProp, sampleEqual):
    total = pronFreqs['tot'] * sampleProp
    numPerClass = int(total/len(pronPercent))

    pronProbs = {}

    if sampleEqual:
        print "Expected count per pronoun: ", numPerClass

    for pron in pronPercent:
        if sampleEqual:
            if pronFreqs[pron] < numPerClass:
                pronProbs[pron] = 1
                print pron, ": ", pronFreqs[pron]
            else:
                pronProbs[pron] = 1.0*numPerClass/pronFreqs[pron]
                print pron, ": ", pronProbs[pron]
        else:
            num = int(total * pronPercent[pron])
            if pronFreqs[pron] < num:
                pronProbs[pron] = 1
                print pron, " Expected: ", num, " Actual: ", pronFreqs[pron]
            else:
                pronProbs[pron] = 1.0*num/pronFreqs[pron]
                print pron, " Expected: ", num, " Prob: ", pronProbs[pron]
            
    return pronProbs

# for offline sampling 
def sample(percentage, all_train, all_dev):
	''' Sentence object has two SentenceWords objects (source and target) and a list filled with PronounInstance objects
	(float, list of Sentence, list of Sentence) -> (list of Sentence)
	'''

	prons_dict_train, train_indexes = getDistribution(all_train)
	prons_dict_dev, dev_indexes = getDistribution(all_dev)

	# determine sample size
	total_examples = sum(prons_dict_train.values())
	sample_size = int(float(total_examples) * percentage)
	
	# determine distribution in dev
	percentages = {}
	total_dev = sum(prons_dict_dev.values())
	for key in prons_dict_dev:
		percentages[key] = float(prons_dict_dev[key])/float(total_dev)
	
	# compute necessary instances to retain from train according to sample size and distribution of dev
	necesary = {}
	for key in percentages:
		necesary[key] = int(percentages[key] * sample_size)
	
	# get instances from train
	final_sample = []
	all_indexes = []

	for key in train_indexes:
		selected = numpy.random.choice(numpy.asarray(train_indexes[key]), necesary[key], replace=True)
		indexes_in_sample = selected.tolist()
		
		all_indexes += indexes_in_sample
	
	for i in set(all_indexes):#set because some sentences can be double if more than one pronoun
		final_sample.append(all_train[i])

	return final_sample

# for online sampling
def readSample(file_name, pronProbs):

    sentenceCount = 0
    exampleCount = 0
    sentenceCountSave = 0
    exampleCountSave = 0

    sentences = []

    with open(file_name, 'r') as pronFP:

        sentence = None
        
        for line in pronFP:
            parts = filter(bool, line.strip().split('\t'))
            if len(parts) == 2:  #sentence pair
                if sentence != None:  #decide if to store previous sentence, before reading a new one
                    freq = 0
                    for pro in sentence.pronouns:
                        pfreq = pronProbs[pro.pClass]
                        if pfreq > freq:
                            freq = pfreq
                    if random.random() <= freq:
                        sentences.append(sentence)
                        
                        sentenceCountSave += 1
                        exampleCountSave += sentence.numPronouns()
                        
                    sentence = None

                sentenceCount += 1
                source = SentenceWords(parts[0], True)
                target = SentenceWords(parts[1], False)
                sentence = Sentence(source, target)
                            
            elif len(parts) == 1:  # pronoun instance
                exampleCount += 1
                sentence.addPronounInstance(PronounInstance(line))           
            else:
                print "something wrong when reading the file ", file_name, " ", line, " ", len(parts)
                   
    if sentence != None:  #possibly store last sentence
        freq = 1
        for pro in sentence.pronouns:
            pfreq = pronProbs[pro.pClass]
            if pfreq > freq:
                freq = pfreq
        if random.random() <= freq:
            sentences.append(sentence)
            
            sentenceCountSave += 1
            exampleCountSave += sentence.numPronouns()
        
    print 'sentences read ', sentenceCount, " instances: ", exampleCount
    print 'sentences stored ', sentenceCountSave, " instances: ", exampleCountSave        

    return sentences



def write_prons(fn, pron_gen, classes):
    with open(fn, 'w') as fh:
        for pronoun in pron_gen:
            fh.write(classes.i2c[pronoun] + '\n')   
 
def write_gold(fn, data):
    with open(fn, 'w') as fh:
        for sentence in data:
            for pronoun in sentence.pronouns:
                fh.write(pronoun.pClass + '\n')
 

def evaluate_pronouns(gold, pred, out, langPair):
    command = 'perl ../utils/evaluate-pronouns.perl ' +  pred + ' ' + gold + ' ' + langPair + ' > ' + out
    print "EVAL command: ", command
    os.system(command)
           
class PronClass:
    def __init__(self, langPair):
        if langPair == "en-fr":
            self.classes = ["ce", "elle", "elles", "il", "ils", "cela", "on", "OTHER"]
        elif langPair == "de-en":
            self.classes = ["he", "she", "it", "they", "you", "this", "these", "there", "OTHER"]
        elif langPair == "es-en":
            self.classes = ["he", "she", "it", "they", "there", "you", "OTHER"]
        else:  #default: en-de
            self.classes = ["er", "sie", "es", "man", "OTHER"]

        self.c2i = {clas: ind for ind, clas in enumerate(self.classes)}
        self.i2c = {ind: clas for ind, clas in enumerate(self.classes)}
        print "classes set: ", self.classes
        
    def getSize(self):
        return len(self.classes)

    def getClassIndex(self, clas):
        return self.c2i[clas]


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return '123' if numberRegex.match(word) else word.lower()
