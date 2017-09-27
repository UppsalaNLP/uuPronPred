from dynet import *
from utils import Sentence, PronClass, read_prons, write_prons
from operator import itemgetter
from itertools import chain
import utils, time, random
import numpy as np

# This code is partially based on:
#  BIST parser by Eli Kiperwasser and Yoav Goldberg: https://github.com/elikip/bist-parser
#  uuparser by Miryam de Lhoneux et al.: https://github.com/UppsalaNLP/uuparser


class PronounLSTM:

    def __init__(self, words, w2i, s_pos, s_deps, lemmas, l2i, t_pos, ch, options):
        
        self.model = Model()
        self.trainer = AdamTrainer(self.model)
        random.seed(1)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        # dimensions #
        self.lstmdims = options.lstm_dims * 2     # lstm
        self.clstmdims = options.chlstm_dims * 2  # character lstm
        self.wdims = options.wembedding_dims   # (source) word dimensions
        self.pdims = options.pembedding_dims   # (source) pos dimensions
        self.ddims = options.dembedding_dims   # (source) dependency dimensions
        self.cdims = options.cembedding_dims   # (source) character dimensions
        self.ldims = options.lembedding_dims   # (target) lemma dimensions
        self.tpdims = options.tpembedding_dims # (target) pos dimensions

        # vocabularies #
        self.wordsCount = words
        self.lemmasCount = lemmas
        # +3 for UNKNOWN (0), PAD (1), INITIAL (2)
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}  # (source) vocabulary, words
        self.lvocab = {word: ind+3 for word, ind in l2i.iteritems()} # (target) vocabulary, lemmas
        self.s_pos = {word: ind+3 for ind, word in enumerate(s_pos)} # (source) vocabulary, pos
        self.t_pos = {word: ind+3 for ind, word in enumerate(t_pos)} # (target) vocabulary, pos
        self.s_deps = {word: ind+3 for ind, word in enumerate(s_deps)} # (source) vocabulary, dep

        self.chars = { ind: word+3 for word, ind in enumerate(ch)}  # (source) characters

        # other #
        self.classes = PronClass(options.langPair)
        self.numClasses = self.classes.getSize()
        self.headFlag = options.headFlag         # use the pronoun's dependency head in the final prediction
        self.updateLimit = options.updateLimit   # how many errors to collect before updating LSTMs
        self.debug = options.debug
        self.pronEmbedding = options.pronEmbedding
        self.nnvecs = 2 + (1 if self.headFlag else 0)  # how many input-"words" (from LSTMS) for final predictions
        self.defDropout = options.defaultDropRate
        self.classWeighting = options.classWeighting

        sdims = self.wdims + self.pdims + self.ddims + self.clstmdims # source dimensions
        tdims = self.ldims + self.tpdims # target dimensions

        print "SDIMS: ", sdims, " TDIMS: ", tdims

        self.blstmFlag = options.blstmFlag  # use one layer BI-LSTM (not fully tested)
        self.bibiFlag = options.bibiFlag    # use two layer BI-LSTM

        # IF BIBIFLAG --> 2 LSTMs #
        if self.bibiFlag:
            # 1st source LSTM
            self.surfaceBuilders = [VanillaLSTMBuilder(1, sdims, self.lstmdims * 0.5, self.model), 
                                    VanillaLSTMBuilder(1, sdims, self.lstmdims * 0.5, self.model)]
            # 2nd source LSTM
            self.bsurfaceBuilders = [VanillaLSTMBuilder(1, self.lstmdims, self.lstmdims * 0.5, self.model), 
                                     VanillaLSTMBuilder(1, self.lstmdims, self.lstmdims * 0.5, self.model)]
            # 1st target LSTM
            self.tsurfaceBuilders = [VanillaLSTMBuilder(1, tdims, self.lstmdims * 0.5, self.model), 
                                    VanillaLSTMBuilder(1, tdims, self.lstmdims * 0.5, self.model)]
            # 2nd target LSTM
            self.tbsurfaceBuilders = [VanillaLSTMBuilder(1, self.lstmdims, self.lstmdims * 0.5, self.model),
                                     VanillaLSTMBuilder(1, self.lstmdims, self.lstmdims * 0.5, self.model)]
        
        # 1 LSTM #
        else:
            #elif self.blstmFlag:
            # NOTE: not properly tested. Should be used with caution. 
            # The recommendation is to not use --disable-bibi-lstm
            self.surfaceBuilders = [VanillaLSTMBuilder(1, sdims, self.lstmdims * 0.5, self.model), LSTMBuilder(1, sdims, self.lstmdims * 0.5, self.model)]
            self.tsurfaceBuilders = [VanillaLSTMBuilder(1, tdims, self.lstmdims * 0.5, self.model), LSTMBuilder(1, tdims, self.lstmdims * 0.5, self.model)]


        # Characters #
        self.charBuilders = [VanillaLSTMBuilder(1, self.cdims, self.clstmdims*0.5, self.model),
                             VanillaLSTMBuilder(1, self.cdims, self.clstmdims*0.5, self.model)]


        ### hidden layers 
        ## as default there is only 1 hidden layer
        self.hidden_units = options.hidden_units   # 1st layer
        self.hidden2_units = options.hidden2_units # 2nd layer

        # *PAD* is intended for paddding
        self.vocab['*PAD*'] = 1
        self.lvocab['*PAD*'] = 1
        self.s_pos['*PAD*'] = 1
        self.s_deps['*PAD*'] = 1
        self.t_pos['*PAD*'] = 1
        self.chars['*PAD*'] = 1

        # not used??
        self.vocab['*INITIAL*'] = 2
        self.lvocab['*INITIAL*'] = 2
        self.s_pos['*INITIAL*'] = 2
        self.s_deps['*INITIAL*'] = 2
        self.t_pos['*INITIAL*'] = 2
        self.chars['*INITIAL*'] = 2
        

        # LOOKUPS (embeddings)#
        self.wlookup = self.model.add_lookup_parameters((len(words) + 3, self.wdims))   # (source) word lookup
        self.llookup = self.model.add_lookup_parameters((len(lemmas) + 3, self.ldims))  # (target) lemma lookup
        self.plookup = self.model.add_lookup_parameters((len(s_pos) + 3, self.pdims))   # (source) pos lookup
        self.dlookup = self.model.add_lookup_parameters((len(s_deps) + 3, self.ddims))  # (source) dependency lookup
        self.tplookup = self.model.add_lookup_parameters((len(t_pos) + 3, self.tpdims)) # (target) pos lookup
        self.clookup = self.model.add_lookup_parameters((len(ch) + 3, self.cdims))      # (source) char lookup

        # LSTMs #
        self.word2lstmS = self.model.add_parameters((self.lstmdims, sdims)) # word to source LSTM
        self.word2lstmT = self.model.add_parameters((self.lstmdims, tdims)) # word to target LSTM
        self.word2lstmbias = self.model.add_parameters((self.lstmdims))     # word to LSTM bias

        self.chPadding = self.model.add_parameters((self.clstmdims))
        
        # HIDDEN LAYERS #
        self.hidLayer = self.model.add_parameters((self.hidden_units, self.lstmdims * self.nnvecs + (self.wdims if self.pronEmbedding else 0))) # 1st layer
        self.hidBias = self.model.add_parameters((self.hidden_units)) # 1st bias

        self.hid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units)) # 2nd layer
        self.hid2Bias = self.model.add_parameters((self.hidden2_units)) # 2nd bias

        self.outLayer = self.model.add_parameters((self.numClasses, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units)) # out layer
        self.outBias = self.model.add_parameters((self.numClasses)) # out bias



    ##############
    #  EVALUATE  #
    ##############
    # sPron = source pronoun 
    # tPron = target pronoun 
    # spHead = source pronoun head 
    # spEmb = source pronoun embedding
    def __evaluate(self, sPron, tPron, spHead, spEmb):  
        
        input = concatenate(filter(None, [sPron, tPron, spHead, spEmb]))
            

	# IF 2 HIDDEN LAYERS #
        if self.hidden2_units > 0: 
            output = (self.outLayer.expr() * self.activation(self.hid2Bias.expr() + self.hid2Layer.expr() * self.activation(self.hidLayer.expr() * input + self.hidBias.expr())) + self.outBias.expr())
        # 1 hidden layer
        else:
            output = (self.outLayer.expr() * self.activation(self.hidLayer.expr() * input + self.hidBias.expr()) + self.outBias.expr())
            
        # SOFTMAX
        scores = softmax(output)

        return scores

    
    #  SAVE  #
    def Save(self, filename):
        self.model.save(filename)

    #  LOAD  #
    def Load(self, filename):
        self.model.load(filename)

    #  INITIALISE  #
    def Init(self):
        paddingWordVec = self.wlookup[1]
        paddingLemmaVec = self.llookup[1]

        paddingPosVec = self.plookup[1] if self.pdims > 0 else None
        paddingDepVec = self.dlookup[1] if self.ddims > 0 else None
        paddingTposVec = self.tplookup[1] if self.tpdims > 0 else None

        paddingCvec = self.chPadding.expr() if self.cdims > 0 else None

        # filter and concatenate source vectors
        svec = concatenate(filter(None, [paddingWordVec, paddingPosVec, paddingDepVec, paddingCvec])) 
        # filter and concatenate target vectors
        tvec = concatenate(filter(None, [paddingLemmaVec, paddingTposVec]))              

        paddingVecS = tanh(self.word2lstmS.expr() * svec + self.word2lstmbias.expr()) # source vectors
        paddingVecT = tanh(self.word2lstmT.expr() * tvec + self.word2lstmbias.expr()) # target vectors
        


    #  SOURCE EMBEDDINGS  #
    def getSourceWordEmbeddings(self, sentence, train):
        # FOREACH  Word #
        for root in sentence.tokens:
            c = float(self.wordsCount.get(root.word, 0))
            # dropout based on frequency 
            dropFlag =  not train or (random.random() < (c/(0.25+c))) 

            # for characters
            if self.cdims > 0:
                if self.defDropout > 0 and train:
                    self.charBuilders[0].set_dropout(self.defDropout)  
                    self.charBuilders[1].set_dropout(self.defDropout)
                forward  = self.charBuilders[0].initial_state()
                backward = self.charBuilders[1].initial_state()

                for char, charRev in zip(root.word, reversed(root.word)):
                    forward = forward.add_input(self.clookup[self.chars.get(char,0)])
                    backward = backward.add_input(self.clookup[self.chars.get(charRev,0)])

                root.chVec = concatenate([forward.output(), backward.output()])
            else:
                root.chVec = None

            root.wordvec = self.wlookup[int(self.vocab.get(root.word, 0)) if dropFlag else 0]
            root.posvec = self.plookup[int(self.s_pos.get(root.pos,0))] if self.pdims > 0 else None
            root.depvec = self.dlookup[int(self.s_deps.get(root.dep,0))] if self.ddims > 0 else None

            root.ivec = concatenate(filter(None, [root.wordvec, root.posvec, root.depvec, root.chVec]))


        # IF 1+ BiLSTM #
        if self.blstmFlag: 
            if self.defDropout > 0 and train:
                self.surfaceBuilders[0].set_dropout(self.defDropout)
                self.surfaceBuilders[1].set_dropout(self.defDropout)
            forward  = self.surfaceBuilders[0].initial_state()
            backward = self.surfaceBuilders[1].initial_state()

            for froot, rroot in zip(sentence.tokens, reversed(sentence.tokens)):
                forward = forward.add_input( froot.ivec )
                backward = backward.add_input( rroot.ivec )
                froot.fvec = forward.output()
                rroot.bvec = backward.output()
            for root in sentence.tokens:
                root.vec = concatenate( [root.fvec, root.bvec] ) # both directions

            # IF 2 BiLSTMs #
            if self.bibiFlag:
                if self.defDropout > 0 and train:
                    self.bsurfaceBuilders[0].set_dropout(self.defDropout)
                    self.bsurfaceBuilders[1].set_dropout(self.defDropout)
                bforward  = self.bsurfaceBuilders[0].initial_state()
                bbackward = self.bsurfaceBuilders[1].initial_state()

                for froot, rroot in zip(sentence.tokens, reversed(sentence.tokens)):
                    bforward = bforward.add_input( froot.vec )
                    bbackward = bbackward.add_input( rroot.vec )
                    froot.bfvec = bforward.output()
                    rroot.bbvec = bbackward.output()
                for root in sentence.tokens:
                    root.vec = concatenate( [root.bfvec, root.bbvec] )

        # ELSE: no biLSTM # (not tested!)
        else:
            for root in sentence.tokens:
                root.ivec = (self.word2lstm.expr() * root.ivec) + self.word2lstmbias.expr()
                root.vec = tanh( root.ivec )    


    #  TARGET EMBEDDINGS  #
    def getTargetWordEmbeddings(self, sentence, train):
        # FOREACH WORD #
        for root in sentence.tokens:
            c = float(self.lemmasCount.get(root.lemma, 0))
            dropFlag =  not train or (random.random() < (c/(0.25+c)))

            root.lemmavec = self.llookup[int(self.lvocab.get(root.lemma, 0)) if dropFlag else 0]
            root.posvec = self.tplookup[int(self.t_pos.get(root.pos,0))] if self.tpdims > 0 else None

            root.ivec = concatenate(filter(None, [root.lemmavec, root.posvec]))

  
        # IF 1+ BiLSTM #
        if self.blstmFlag:
            if self.defDropout > 0 and train:
                self.tsurfaceBuilders[0].set_dropout(self.defDropout)
                self.tsurfaceBuilders[1].set_dropout(self.defDropout)
            forward  = self.tsurfaceBuilders[0].initial_state()
            backward = self.tsurfaceBuilders[1].initial_state()

            for froot, rroot in zip(sentence.tokens, reversed(sentence.tokens)):
                forward = forward.add_input( froot.ivec )
                backward = backward.add_input( rroot.ivec )
                froot.fvec = forward.output()
                rroot.bvec = backward.output()
            for root in sentence.tokens:
                root.vec = concatenate( [root.fvec, root.bvec] )

            # IF 2 BiLSTMs #
            if self.bibiFlag:
                if self.defDropout > 0 and train:
                    self.tbsurfaceBuilders[0].set_dropout(self.defDropout)
                    self.tbsurfaceBuilders[1].set_dropout(self.defDropout)
                bforward  = self.tbsurfaceBuilders[0].initial_state()
                bbackward = self.tbsurfaceBuilders[1].initial_state()

                for froot, rroot in zip(sentence.tokens, reversed(sentence.tokens)):
                    bforward = bforward.add_input( froot.vec )
                    bbackward = bbackward.add_input( rroot.vec )
                    froot.bfvec = bforward.output()
                    rroot.bbvec = bbackward.output()
                for root in sentence.tokens:
                    root.vec = concatenate( [root.bfvec, root.bbvec] )

        # ELSE: no BiLSTMs # (not tested!)
        else:
            for root in sentence.tokens:
                root.ivec = (self.word2lstm.expr() * root.ivec) + self.word2lstmbias.expr()
                root.vec = tanh( root.ivec )    


    #  PREDICT  #
    def Predict(self, devData):
        preds = []

        # FOREACH Dev-Data Sentence #
        for iSentence, sentence in enumerate(devData):
            self.Init()
            self.getSourceWordEmbeddings(sentence.source, False) # False = predict!
            self.getTargetWordEmbeddings(sentence.target, False) # False = predict!


            # FOREACH pronoun to be predicted #
            for pronounInstance in sentence.pronouns:
                sPron = sentence.source.tokens[pronounInstance.sIndex].vec
                tPron = sentence.target.tokens[pronounInstance.tIndex].vec
                spHead = sentence.source.tokens[pronounInstance.pHeadIndex].vec if self.headFlag else None
                spEmb = self.wlookup[int(self.vocab.get(pronounInstance.sPron, 0))] if self.pronEmbedding else None
                scores = self.__evaluate(sPron, tPron, spHead, spEmb)
                probs = scores.npvalue()
                best = np.argmax(probs)
                preds.append(best)

        return preds

    #  TRAIN  #
    def Train(self, trainData):

        # TIMER
        start = time.time()

        shuffledData = trainData  
        random.shuffle(shuffledData)

        mloss = 0.0
        eloss = 0.0
        etotal = 0
        errs = []
        eeloss = 0.0
        eetotal = 0

        self.Init()

        #for weighted loss
        #dict_of_counts, dict_of_indexes(not really needed here)
        if self.classWeighting:
            data_distrib, i_sentences = utils.getDistribution(trainData)

        # FOREACH training sentence #
        for iSentence, sentence in enumerate(shuffledData):

            # PRINT STATUS every 100 #
            if iSentence % 100 == 0 and iSentence != 0:
                print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Time', time.time()-start
                eeloss += eloss
                eetotal += etotal
                start = time.time()
                eloss = 0.0
                etotal = 0
                
            self.getSourceWordEmbeddings(sentence.source, True) # True = train!
            self.getTargetWordEmbeddings(sentence.target, True) # True = train!

            # FOREACH pronoun #
            for pronounInstance in sentence.pronouns:
                etotal += 1

                sPron = sentence.source.tokens[pronounInstance.sIndex].vec
                tPron = sentence.target.tokens[pronounInstance.tIndex].vec
                if self.headFlag:
                    if pronounInstance.pHeadIndex == -1:
                        # this is the case where the pronoun is the root.
                        # currently we use the first word as a proxy for the root (stupid)
                        # TODO: fix this in some more principled way? (not vey common though...)
                        #         possibly, concatenate last forward and first backward
                        spHead = sentence.source.tokens[0].vec
                    else:
                        spHead = sentence.source.tokens[pronounInstance.pHeadIndex].vec  
                else:
                    spHead = None

                # source pronoun embeddings
                spEmb = self.wlookup[int(self.vocab.get(pronounInstance.sPron, 0))] if self.pronEmbedding else None

                # SCORE #
                scores = self.__evaluate(sPron, tPron, spHead, spEmb)

                # LOSSES #
                loss = pickneglogsoftmax(scores, self.classes.getClassIndex(pronounInstance.pClass))
                if self.classWeighting:
                    #weighted loss
                    weight = 1 - ((data_distrib[pronounInstance.pClass] * 1.0) / sum(data_distrib.values()))
                    loss = scalarInput(weight)*loss
               
                eloss += loss.scalar_value()
                errs.append(loss)
                    

            # IF ENOUGH ERRORS --> UPDATE #
            if len(errs) > self.updateLimit: 
                eerrs = esum(errs) # * (1.0/(float(len(errs))))
                scalar_loss = eerrs.scalar_value()
                eerrs.backward()
                self.trainer.update() 
                errs = []

                renew_cg()
            self.Init()
        # END OF FOREACH training sentence #

        # update on all remaining errors collected at the end of the data
        if len(errs) > 0:
            eerrs = (esum(errs)) # * (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            renew_cg()

        self.trainer.update_epoch() 
        print "Loss: ", eeloss/iSentence  
