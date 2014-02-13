atHome = True

if (atHome):
  dirarticle = r"C:/Documents and Settings/Heather/My Documents/research/AMIA 2008/SelectedOA/"
  dirfilelist = r"C:/Documents and Settings/Heather/My Documents/research/AMIA 2008/"
  dirout = r"C:/Documents and Settings/Heather/My Documents/research/AMIA 2008/OUT/"
else:
  dirarticle = r"E:/articles/SelectedOA/"
  dirfilelist = r"E:/AMIA 2008/"
  dirout = r"E:/AMIA 2008/OUT/"

import string
import re
import glob
import os
import nltk
import math
import operator
import time
import nltk.evaluate
from collections import defaultdict 
from nltk import wordnet
from pprint import pprint

def precision(reference, test):
  if len(test) == 0: 
    return None 
  else: 
    num_correct = [1 for r,t in zip(reference, test) if (t==True and r==t)]
    return float(sum(num_correct))/sum(test)

def recall(reference, test):
  if len(reference) == 0: 
    return None 
  else: 
    num_correct = [1 for r,t in zip(reference, test) if (r==True and r==t)]
    return float(sum(num_correct))/sum(reference)

# from nltk.evaluate.fmeasure
def fmeasure(reference, test, alpha=0.5):
    p = precision(reference, test) 
    r = recall(reference, test) 
    if p is None or r is None: 
        return None 
    if p == 0 or r == 0: 
        return 0 
    return 1.0/(alpha/p + (1-alpha)/r)

def loadtagger():
  from cPickle import load
  input = open('tagger.pkl', 'rb')
  tagger = load(input)
  input.close()
  return(tagger)

def loadoutput():
  from cPickle import load
  pkl = open(dirout + 'DBgrep.pkl', 'rb')
  output = load(pkl)
  pkl.close()
  return(output)

def get_filename_list(cohort):
  filename_list = []
  filenames = open(dirfilelist + cohort + ".txt", "r").read().split("\n")

  for filename in filenames:
      #filename_list.append(dirarticle + cohort + filename[1:])
      filename_list.append(filename)
  return(filename_list)

def in_test_set(pmcid):
      lasttwo = pmcid[-2:]
    # if in the test set, don't run with this one, skip to the next filename
      if (lasttwo.startswith("6") or lasttwo.startswith("7") or
        lasttwo.startswith("8") or lasttwo.startswith("9")):
        return(True)
      else:
        return(False)
      
tagger = loadtagger()

def pos(sent):
  return(tagger.tag(nltk.wordpunct_tokenize(sent)))

chunkers = dict()

grammar0 = r"""
    VX: {<TO|BE.*|DO.*|HV.*|MD|RB.*|VB.*>+<TO|BE.*|DO.*|HV.*|MD|RB.*|VB.*|JJ.*|IN|PP>*}
    """
chunkers[0] = nltk.RegexpParser(grammar0)

grammar1 = r"""
    NX: {<AT|DT|N.*|JJ.*|P.*>+}
    """
chunkers[1] = nltk.RegexpParser(grammar1)
  
cfdcohort = nltk.ConditionalFreqDist()
cfdleaves = nltk.ConditionalFreqDist()

cohorts = defaultdict()
features_list = ("grep", "accession", "accessionword", "url", "phrasesP", "phrasesR", "phrasesPR")
cohorts_list = ("website", "smd", "arrayexpress", "geo", "genbank", "pdb")

for cohort in cohorts_list:
  cohorts[cohort] = dict()
cohorts["all"]= dict()
cohorts["website"] = dict()
  
cohorts["geo"]["grep"]  = r"\bGEO\b|Omnibus"
cohorts["geo"]["accession"]  = r"\bG[A-Z]{2}.[0-9]{3,10}\b"
cohorts["arrayexpress"]["grep"] = r"ArrayExpress"
cohorts["arrayexpress"]["accession"]  = r"\b[A-Z]-[A-Z]{4}-[0-9]{1,6}\b"
cohorts["smd"]["grep"]  = r"SMD|Stanford Microarray Data"
cohorts["smd"]["accession"]  = r"\b[0-9]{5}\b"
cohorts["genbank"]["grep"]  = r"Genbank"
cohorts["genbank"]["accession"]  = r"\b[A-Z]{1,3}[0-9]{3,10}\b"
cohorts["pdb"]["grep"]  = r"\bPDB\b|Protein Datab|Protein Data b"
cohorts["pdb"]["accession"]  = r"\b[0-9][A-Za-z0-9]{3}\b"
cohorts["all"]["url"] = r"http|ftp|www"
cohorts["all"]["accessionword"] = r"accession"
cohorts["all"]["phrasesP"] = "(" + "|".join([r"(\b(we|have|is|was|were|is|are|be|have|has|been).(accessioned|added|archived|assigned|deposited|entered|imported|included|inserted|loaded|lodged|placed|posted|provided|registered|reported.to|stored|submitted|uploaded.to))",
                                                r"(\b(is|are|will.be|made).{0,20}(available|accessible))",
                                                r"(\b(be).(accessed|browsed|downloaded|found|obtained|queried|retrieved|searched|viewed))",
                                                r"(\b(through|under|as).{0,20}accession)",
                                                r"(\b((given)|new|received|assigned).{0,20}(accession))",
                                                r"(data.{0,20}availability|for public distribution|for.{0,20}release upon publication|for the.{0,20}data.{0,20}generated|from this study have.{0,20}accession|data.{0,10}from this study|access to.{0,20}data.{0,10})"
                                            ]) + ")"
cohorts["all"]["phrasesR"] = "(" + "|".join([r"(\b(accession.{0,20}(for|at).{0,100}(is|are)))",
                                              r"(\b(raw|original|our|complete|detailed).{0,20}data)",
                                              r"(\b(we|have|is|was|were|is|are|be|have|has|been).(exported|gave|given|listed|provided|reported))"
                                            ]) + ")"
cohorts["all"]["phrasesPR"] = "(" + "|".join([cohorts["all"]["phrasesP"],
                                              cohorts["all"]["phrasesR"]
                                            ]) + ")"

cohorts["website"]["grep"] = r"(microarray.{0,20}(http|ftp|www))|((http|ftp|www).{0,20}microarray)"
#cohorts["website"]["grep"] = r"(our.web)"
cohorts["website"]["accession"] = r"http|ftp|www"
cohorts["all"]["accessionword"] = r"online|web|url|site"


for cohort in cohorts_list:
  cohorts[cohort]["pad"] = "(.{0,150}(" + cohorts[cohort]["grep"] + ").{0,150})"
  for feature in cohorts[cohort].keys():
    cohorts[cohort][feature + "compile"] = re.compile(cohorts[cohort][feature], re.IGNORECASE | re.DOTALL)
    for bonusfeature in cohorts["all"].keys():
      cohorts[cohort][bonusfeature + "compile"] = re.compile(cohorts["all"][bonusfeature], re.IGNORECASE | re.DOTALL)
  

#pmcids_out = loadoutput()
pmcids = dict()
for cohort in cohorts_list:
  pmcids_list = get_filename_list(cohort)
  pmcids_set = set(pmcids_list)
  pmcids1_list = get_filename_list(cohort + "1")
  pmcids[cohort + "1"] = set(pmcids1_list)
  pmcids[cohort + "0"] = pmcids_set.difference(pmcids[cohort + "1"])

  pmcids[cohort+"1"+"test"] = set(filter(in_test_set, pmcids[cohort+"1"]))
  pmcids[cohort+"1"+"train"] = pmcids[cohort + "1"].difference(pmcids[cohort+"1"+"test"])
  pmcids[cohort+"0"+"test"] = set(filter(in_test_set, pmcids[cohort+"0"]))
  pmcids[cohort+"0"+"train"] = pmcids[cohort + "0"].difference(pmcids[cohort+"0"+"test"])

for key in sorted(pmcids.keys()):
  print key, len(pmcids[key])

# match url that is in triangle brackets
urlbracketpattern = '(<[^>]*?((http|ftp)[^<]*?)>)'

# get text, removing markup, keeping any urls that were embedded in markup
def get_text(filename):
     # "Read text from a file, normalizing whitespace and stripping HTML markup"
     try:
        textFH = open(filename)
        text = textFH.read()
        textFH.close()
     except:
        #print "couldn't open and read from file " + filename
        text = ""

     urlmatches = re.findall(urlbracketpattern, text, re.IGNORECASE | re.DOTALL)
     for amatch in urlmatches:
         #print amatch
         # replace, to get rid of the angle brackets so it won't be deleted
         text = text.replace(amatch[0], amatch[1])
     ## Strip XML markup
     text = re.sub(r'<.*?>', ' ', text)
     # replace double quotes
     text = text.replace('"', " ")
     return text

def runGetVerbs(padFind):
  verbstrings = []
  verbngram = []
  posstrings = []
  posngram = []
  vbgstrings = []
  for sent in padFind:
    possent = pos(sent)
    
    for chunki in range(0, 1):  #range(len(chunkers)):
      chunksent = chunkers[chunki].parse(possent)
      if (chunksent == None):
        continue
      
      #print "\n\n", sent
      #print possent, "\n\n"
      #print chunksent
      count = 0
      for c in chunksent.subtrees():
        count = count + 1
        if (count>1):
          #print c.leaves()
          leaves = string.join([nltk.tag.util.tuple2str(leaf) for leaf in c.leaves()])
          leaves = leaves.lower()
          #print "\n***" + leaves,
          verbstrings.append(" ".join([leaf[0] for leaf in c.leaves()]))
          posstrings.append(" ".join([leaf[1] for leaf in c.leaves()]))
          verbngram.append("o".join([leaf[0] for leaf in c.leaves()]))
          posngram.append("o".join([leaf[1] for leaf in c.leaves()]))
          vbgstrings.append(" ".join([leaf[0] for leaf in c.leaves() if leaf[1] in ["VBG", "VBD", "JJ"]]))
    
  return (" ".join(verbstrings) + " ".join(posstrings) + " ".join(verbngram) + " ".join(posngram),
          " ".join(vbgstrings))

def runWordnet():
        #if False:
          #for leaf in c.leaves():
            #if(leaf[1]=="VBN" or leaf[1]=="VBD" or leaf[1]=="JJ"):
            #if (re.match("TO|BE.*|DO.*|HV.*|MD|IN|PP", leaf[1])):  
            #  cfdcohort[cohortvalue].inc(leaf)         
            #  cfdleaves[leaf].inc(cohortvalue)
              
                    maxscore = -1
                    try:
                      verb = wordnet.morphy(leaf[0], wordnet.VERB)
                      senses = wordnet.V[verb]
                      for sense in senses:
                        ref_verbs = ["submit","obtain","enter","access","archive","retrieve","present","post","query","import","download","view","find","deposit"]
                        s = set([])
                        for ref in ref_verbs:
                          newscore = wordnet.V[ref][0].path_similarity(sense)
                          s.update(wordnet.V[ref][0].hypernym_paths()[0])
                          maxscore = max(maxscore,newscore)
                        print s,"\n\n"
                    except:
                      maxscore = -2
                    print maxscore
        #cfdcohort[cohortvalue].inc(leaves)         
        #cfdleaves[leaves].inc(cohortvalue)


def calcscore(c, scoremult=+1):
  positive = cfdleaves[c]["train1"]
  total = cfdleaves[c]["train1"] + cfdleaves[c]["train0"] + 0.0
  score = scoremult * (positive/total) * math.log(total, 2)
  return(score)

def calcfreq(c, cohort, scoremult=+1):
  score = cfdleaves[c][cohort]
  return(score)
      
def runcalcs(cohort, scoremult=+1):        
  #scores = [(c, calcscore(c, scoremult)) for c in cfdcohort[cohort].samples()]
  scores = [(c, calcfreq(c, cohort, scoremult)) for c in cfdcohort[cohort].samples()]
  scoressorted = sorted(scores, key=operator.itemgetter(1), reverse=True)  # sorts on second element of tuple.  0 for first.
  print scoressorted

tagger = loadtagger()
#runall()
#runcalcs("train1")
#runcalcs("train0", -1)



#sPMID = r'<article-id pub-id-type="pmid">[0-9]+</article-id>'
#rPMID = re.compile(sPMID, re.IGNORECASE | re.DOTALL)
  
def stats(padFind, rFeature, answer):
    ## Split into sentences
    #sentences = sentenceSplitter(text)

    foundone = False
    sentences = []
    for aPadMatch in padFind:
      #print "sent:" + sent
      sent = aPadMatch[0]
      #print sent
      sentences.append(sent)
      #print "\n",sent
      searchmatch = rFeature.search(sent)
      if (searchmatch):
          foundone = True
          #print "searchmatch:" + searchmatch.group(0)
    #if foundone:
      #print "FOUND ONE!"
      #if answer==False:
      #print "FOUND ONE!"
      #print "\n",sentences
    #else:
      #print "DIDN'T FIND ONE"
      #if answer==True:
        #print "DIDN'T FIND ONE"
        #print "\n",sentences
    return(foundone)

  
def tester(cohort, filenames, answer, label=""):
  cohortstarttime = time.time()
  outputs = defaultdict()
  outputFH = open(dirout + "ML" + label + ".csv", "w")
  outputFH.write("pmcid,cohort,Code,text")
  #outputFH.write(",text,text")  # extra for verb experiment
  i = 0
  for feature in features_list:
    outputs[feature] = []
    i = i+1
    outputFH.write(",f" + str(i))  # titles need to be of the form f1?
    
  for filename in filenames:
        #print "\n", i, filename
        print ".",
        text = None
        lasttwo = filename[-2:]
        text = get_text(dirarticle + lasttwo + "/" + filename + ".nxml")
        numPads = 0
        if text:
          padFind = cohorts[cohort]["padcompile"].findall(text)
          numPads = len(padFind)
          if (numPads>0):
            outputFH.flush()
            outputFH.write("\n" + filename + "," + cohort)
            outputFH.write("," + answer + "-Code")
            allsents = "|||".join([padFind[i][0] for i in range(len(padFind))])
            allsents = re.sub("\s", " ", allsents).replace(",", " ").replace('"', ' ').encode('utf-8')
            outputFH.write("," + allsents)
            #(verbString, vbgverbs) = runGetVerbs([m[0] for m in padFind])
            #outputFH.write("," + verbString + "," + vbgverbs)

        for feature in features_list:
          if (numPads>0):
             out = stats(padFind,
                         cohorts[cohort][feature + "compile"],
                         answer)
             outputFH.write("," + feature + str(out))
             outputFH.flush()
             print filename, feature, out
          else:
            if text:  # so just didn't find the pad
               out = False
            else:          # couldn't open the file
               out = None
          outputs[feature].append(out)
          

  outputFH.write("\n")
  outputFH.close()
  elapsed = (time.time() - cohortstarttime)/60
  print "\n %d: %.2f elapsed minutes" % (len(filenames), elapsed)        
  return(outputs)

def writeForML():
  for cohort in ["website"]: # cohorts_list:
    print "\n\n***", cohort
    tester(cohort, pmcids[cohort + "0test"], "0", cohort + "0test")
    tester(cohort, pmcids[cohort + "1test"], "1", cohort + "1test")
    tester(cohort, pmcids[cohort + "0train"], "0", cohort + "0train")
    tester(cohort, pmcids[cohort + "1train"], "1", cohort + "1train")


def print_stats(ref_output, guess_output):
  print "precision:\t%.3f" % precision(ref_output, guess_output)
  print "recall:\t\t%.3f" % recall(ref_output, guess_output)
  print "fmeasure:\t%.3f" % fmeasure(ref_output, guess_output)

results = defaultdict()

def run():
#  tt = "train"
  tt = "test"
  
  # cohorts_list = ("smd", "arrayexpress", "geo", "genbank" ) # , "pdb")
  starttime = time.time()
  for cohort in cohorts_list:
    print "\n\n", cohort, "\n"
    subcohort = cohort + "1" + tt
    results[subcohort] = tester(cohort, pmcids[subcohort], "1", subcohort)
    print len(pmcids[subcohort])
    print subcohort
    for feature in features_list:
      value = recall([True for i in results[subcohort][feature]], results[subcohort][feature])
      print "%s: %.3f" % (feature, value)
  elapsed = (time.time() - starttime)/60
  print "\n %.2f elapsed minutes in total" % elapsed

  for cohort in cohorts_list:
    subcohort = cohort + "1" + tt
    hasDB = [list(pmcids[subcohort])[i] for i in range(len(results[subcohort]["grep"])) if results[subcohort]["grep"][i] == True]
    hasDB_set = set(hasDB)
    pmcids[subcohort+"hasDB"] = set(hasDB)
    pmcids[subcohort+"noDB"] = pmcids[subcohort].difference(pmcids[subcohort+"hasDB"])

    for key in sorted(pmcids.keys()):
      if key.startswith(cohort):
        print key, len(pmcids[key])

    results[subcohort + "hasDB"] = {}
    results[subcohort + "noDB"] = {}
    results[subcohort + "allText"] = {}
    for feature in features_list:
      results[subcohort + "hasDB"][feature] = [results[subcohort][feature][i] for i in range(len(results[subcohort]["grep"])) if results[subcohort]["grep"][i] == True]
      results[subcohort + "noDB"][feature] = [results[subcohort][feature][i] for i in range(len(results[subcohort]["grep"])) if results[subcohort]["grep"][i] == False]
      results[subcohort + "allText"][feature] = [results[subcohort][feature][i] for i in range(len(results[subcohort]["grep"])) if not results[subcohort]["grep"][i] == None]

  for cohort in cohorts_list:
    subcohort = cohort + "1" + tt
    for eachcohort in [subcohort + "allText", subcohort + "hasDB"]:
      print "\n\n"
      for feature in features_list:
        value = recall([True for i in results[eachcohort][feature]], results[eachcohort][feature])
        print "%s %s: %.3f, n=%d" % (eachcohort, feature, value, len(results[eachcohort][feature]))

  for cohort in cohorts_list:
    print cohort
    subcohortneg = cohort + "0" + tt
    subcohortpos = cohort + "1" + tt + "hasDB"
    results[subcohortneg] = tester(cohort, pmcids[subcohortneg], "0", subcohortneg)

    for feature in features_list:    
      print "\n***", feature, "***"
      ref_output = [False for i in results[subcohortneg][feature]] + [True for i in results[subcohortpos][feature]]
      guess_output = results[subcohortneg][feature] + results[subcohortpos][feature]
      print_stats(ref_output, guess_output)

  
def run_test():
  #filenames = get_filename_list(cohort)

  (verb1_out, accession1_out, db1_out, quickverbs1_out) = tester(cohort + "1", pmcids1_set, "no telling")  #test1 train1
  (verb0_out, accession0_out, db0_out, quickverbs0_out) = tester(cohort + "0", pmcids0_set, "no telling")
  guess_output_verb = verb0_out + verb1_out
  guess_output_accession = accession0_out + accession1_out
  guess_output_db = db0_out + db1_out
  guess_output_quickverbs = quickverbs0_out + quickverbs1_out
  ref_output = [False for i in verb0_out] + [True for i in verb1_out]
  print_stats(ref_output, guess_output_verb)
  print_stats(ref_output, guess_output_accession)
  print_stats(ref_output, guess_output_db) 
  print_stats(ref_output, guess_output_quickverbs)
  print_stats(guess_output_accession, guess_output_verb)

#combo1 = [verb1_out[i] or accession1_out[i] for i in range(len(verb1_out))]

nums = [`i`+`j` for i in xrange(0, 10) for j in xrange(10)]

def dogrep():
  FH = open(dirout + "grepcohortDBs2.txt", "w")
  #output = defaultdict(dict)
  for num in nums:
    for (directory, subdirectorylist, filelist) in os.walk(dirarticle + num):
        for filename in filelist:
            if filename.endswith(".nxml"):
                pmcid = filename[:-5]  # except the .nxml
                lasttwo = pmcid[-2:]
                print lasttwo, " ", 
                text = get_text(os.path.join(dirarticle, lasttwo) + "/" + filename)
                #print "\n" + pmcid + ":\t",
                        
                for cohort in ["website"]:  #cohorts_list:
                        pattern = cohorts[cohort]["grepcompile"]
                        if pattern.search(text):
                                #output[cohort][pmcid] = 1
                                FH.write(cohort + "|" + pmcid + "\n")
                                FH.flush()
                                print "\n" + pmcid + ": " + cohort + "\n"
                        #else:
                        #        output[pattern][pmcid] = 0
                        #        print 0

  #for cohort in output.keys():
  #        print "\n", cohort
  #        print len(output[cohort].keys()), " file matches"
  FH.close()
  #return(output)

# o = dogrep()

  
def calcPR(b, c, d):
  alpha = 0.5
  r = float(d)/(d+c)
  p = float(d)/(d+b)
  f = 1.0/(alpha/p + (1-alpha)/r)
  print "precision:\t%.3f" % p
  print "recall:\t\t%.3f" % r
  print "fmeasure:\t%.3f" % f
