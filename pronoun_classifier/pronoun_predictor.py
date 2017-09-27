from optparse import OptionParser
from pronounLSTM import PronounLSTM
import pickle, utils, os, time, sys
import utils

# This code is partially based on:
#  BIST parser by Eli Kiperwasser and Yoav Goldberg: https://github.com/elikip/bist-parser
#  uuparser by Miryam de Lhoneux et al.: https://github.com/UppsalaNLP/uuparser


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="pron_train", help="Annotated PRON train file", metavar="FILE")
    parser.add_option("--dev", dest="pron_dev", help="Annotated PRON dev file", metavar="FILE")
    parser.add_option("--test", dest="pron_test", help="Annotated PRON test file", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="pronoun.model")
    parser.add_option("--test-out", dest="test_out", help="Name of output file at test time", metavar="FILE", default="test_pred.pron_out")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=10)
    parser.add_option("--dembedding", type="int", dest="dembedding_dims", default=15)
    parser.add_option("--lembedding", type="int", dest="lembedding_dims", default=100)
    parser.add_option("--tpembedding", type="int", dest="tpembedding_dims", default=10)
    parser.add_option("--lpembedding", type="int", dest="lpembedding_dims", default=100)
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=12)
    parser.add_option("--chlstmdims", type="int", dest="chlstm_dims", default=50)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=100) 
    parser.add_option("--dynet-seed", type="int", dest="seed", default=7)
    parser.add_option("--disable-bibi-lstm", action="store_false", dest="bibiFlag", default=True)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--usehead", action="store_true", dest="headFlag", default=False)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--disablePredictEval", action="store_false", dest="predictEval", default=True)
    parser.add_option("--dynet-mem", type="int", dest="cnn_mem", default=512)
    parser.add_option("--continue", dest="continueTraining", action="store_true", default=False)
    parser.add_option("--continueModel", dest="continueModel", help="Load model file, when continuing to train a previously trained model", metavar="FILE", default=None)
    parser.add_option("--debug", action="store_true", dest="debug", default=False)
    parser.add_option("--langPair", type="string", dest="langPair", default="en-de")
    parser.add_option("--updateLimit", type="int", dest="updateLimit", default=25)
    parser.add_option("--use-pron-emb", action="store_true", dest="pronEmbedding", default=False)
    parser.add_option("--defaultDropRate", type="float", dest="defaultDropRate", help="default value for dropout, in LSTMS", default=0.33)
    parser.add_option("--offlineSampling", action="store_true", dest="offlineSampling", default=False)
    parser.add_option("--samples-path", type="string", dest="samplesPath", default="./")
    parser.add_option("--class-weighting", action="store_true", dest="classWeighting", default=False)
    parser.add_option("--filter-rare", type="int", dest="filter_rare", default=-1)
    parser.add_option("--online-sampling", action="store_true", dest="onlineSampling", default=False, help="the proportion of the data to use in each epoch")
    parser.add_option("--online-sample-prop", type="float", dest="sampleProp", default=0.1, help="sample with equal proportions of classes isntead of using dev distribution")
    parser.add_option("--online-sampling-equal", action="store_true", dest="sampleEqual", default=False)
    parser.add_option("--first-epoch", type="int", dest="first_epoch", default=0)    

    (options, args) = parser.parse_args()

    # if characters are not used, set charlstm to 0 as well!!
    if options.cembedding_dims <= 0:
        options.chlstm_dims = 0

    noSampling = not (options.onlineSampling or options.offlineSampling)

    ############
    # TRAINING #
    ############
    if not options.predictFlag:
        # TODO: sanity checks (?)

        # READ DATA #
        print 'Reading dev data'
        devData = []
        with open(options.pron_dev, 'r') as pronFP:
            devData = list(utils.read_prons(pronFP))

        #write only pronouns, for evaluation purposes
        gold_path = os.path.join(options.output, 'devdata.gold')
        utils.write_gold(gold_path, devData)

        trainData = []
        if noSampling:
            print 'Reading training data'
            with open(options.pron_train, 'r') as pronFP:
                trainData = list(utils.read_prons(pronFP))

            print 'Preparing vocab'
            if not options.continueTraining:
                s_words, w2i, s_pos, s_deps, t_lemmas, l2i, t_pos, ch = utils.vocab(trainData, options.filter_rare)

        elif options.offlineSampling:
            if not options.continueTraining:
                # save vocab by reading through file without storing examples!
                s_words, w2i, s_pos, s_deps, t_lemmas, l2i, t_pos, ch, pronFreqs = utils.vocabFromFile(options.pron_train, options.filter_rare)
            
            f_id = 0
            Samples = options.samplesPath + options.langPair + "/"
            dirs = os.listdir(Samples)

        else: # do onlineSampling

            # save vocab by reading through file without storing examples!
            # if not options.continueTraining:  read this only for pronfreqs, regardless of continue. 
            #  Overwrite the other stuff later if necessary (could be cleaned up!)
            s_words, w2i, s_pos, s_deps, t_lemmas, l2i, t_pos, ch, pronFreqs = utils.vocabFromFile(options.pron_train, options.filter_rare)

            # store pronoun percentages in dev
            pronPercent =  utils.getDistributionPercentage(devData)
            pronProbs = utils.getPronounProbabilities(pronFreqs, pronPercent, options.sampleProp, options.sampleEqual) 


        if not options.continueTraining:
            with open(os.path.join(options.output, options.params), 'w') as paramsfp:
                pickle.dump((s_words, w2i, s_pos, s_deps, t_lemmas, l2i, t_pos, ch, options), paramsfp)
            print 'Finished collecting vocab'
            
            print 'Initializing pronoun lstm:'
            predictor = PronounLSTM(s_words, w2i, s_pos, s_deps, t_lemmas, l2i, t_pos, ch, options)

        else:
            with open(options.params, 'r') as paramsfp:
                s_words, w2i, s_pos, s_deps, t_lemmas, l2i, t_pos, ch, stored_opt = pickle.load(paramsfp)
            
            print 'Initializing pronoun lstm:'
            predictor = PronounLSTM(s_words, w2i, s_pos, s_deps, t_lemmas, l2i, t_pos, ch, stored_opt)

            # read in an already trained model, and continue to train it!
            print "continue model: ", options.continueModel
            predictor.Load(options.continueModel)
   
        #################
        # FOREACH EPOCH #
        #################
        
        for epoch in xrange(options.first_epoch, options.epochs):
            print 'Starting epoch', epoch

            data = trainData   #default when no sampling is used

            if options.onlineSampling:
                data = utils.readSample(options.pron_train, pronProbs) 

            elif options.offlineSampling:
                print 'sample file --->', dirs[f_id]
                with open(Samples + dirs[f_id], 'r') as pronFP:
                    data = list(utils.read_prons(pronFP))
                    if f_id < len(dirs):
                        f_id += 1
                    else:
                        f_id == 0

            predictor.Train(data)
            print "Training done", epoch
            devpath = os.path.join(options.output, 'dev_epoch_' + str(epoch+1) + '.pron_out')
            utils.write_prons(devpath, predictor.Predict(devData), predictor.classes)
            utils.evaluate_pronouns(gold_path, devpath, devpath + '.res', options.langPair)
            print 'Finished predicting dev'
            predictor.Save(os.path.join(options.output, options.model + str(epoch+1)))
            

    ##############
    # PREDICTION #
    ##############
    else:  
        print 'Reading test data'
        testData = []
        with open(options.pron_test, 'r') as pronFP:
            testData = list(utils.read_prons(pronFP))

        if options.predictEval:
            #write only pronouns, for eval purposes
            gold_path = os.path.join(options.output, 'testdata.gold')
            utils.write_gold(gold_path, testData)


        with open(options.params, 'r') as paramsfp:
            s_words, w2i, s_pos, s_deps, t_lemmas, l2i, t_pos, ch, stored_opt = pickle.load(paramsfp)

        predictor = PronounLSTM(s_words, w2i, s_pos, s_deps, t_lemmas, l2i, t_pos, ch, stored_opt)
        predictor.Load(options.model)
        testpath = os.path.join(options.output, options.test_out)
        ts = time.time()
        pred = list(predictor.Predict(testData))
        te = time.time()
        utils.write_prons(testpath, pred, predictor.classes)
        if options.predictEval:
            utils.evaluate_pronouns(gold_path, testpath, testpath + '.res', options.langPair)
        
        print 'Finished predicting test',te-ts
