# BiLSTM-based pronoun predictor

Code for cross-lingual pronoun prediction, based on BiLSTM representations of the source and target sentence, enriched by dependency and character information.

Version 1.0 was used for the Uppsala submissiont to the [DiscoMT 2017 shared task on cross-lingual pronoun prediction](https://www.idiap.ch/workshop/DiscoMT/shared-task)

Developed by Sara Stymne, Sharid Loáiciga and Fabienne Cap. Uppsala University. 2017.


## Required software

* Python 2.7 interpreter

* DyNet library (dynet 1.0 was used)


## Using the code

### Data format

Our system uses it's own internal data formats. There are two scripts for converting from Disco-MT format to our input format, given a dependency parse of the sentences with pronouns in either Conll or Conll-u format. There is another script for converting from our output format to DiscoMT-format, given the DiscoMT original input file.

* Convert discoMT format to internal data file format:
    * get-lstm-features-raw-conll.perl
    * get-lstm-features-raw-conllu.perl
* Convert internal output format to discoMT format:
    * lstm2disco.perl


### Training a system

To train the system you need to have training data and development data (for evaluating the system after each epoch). The command below shows how to run the system. There are many more parameters, please see the code to learn about these. After each epoch the system will write the current model as well as develoment results in the results directory. It also writes one parameter file there.

```
python pronoun_predictor.py --train [training data] --outdir [results directory] --dynet-seed N  --langPair [language pair] --epochs N  --dev [development data]  [--use-pron-emb]  [--usehead] 
```

### Applying a system

To apply the system on some test data, you need to supply a trained model and a parameter file.

```
python pronoun_predictor.py --predict --outdir [results directory] --dynet-seed N  --langPair [language pair]  --test [test data]  --model [model file] --params [params file] [--predictEval]
```

### Choosing the best epoch

Given a run with dev data, there are a scripts for choosing the best epoch, which gives options for different metrics. We recommend using the average between macro-recall and accuracy (used in our DiscoMT 2017 system, see citation below)

* pick-best-epochs-metrics.perl


### Sampling training data

To address skewed classes in the training data, we have developed several sampling schemes. In the paper we call them *equal*, *proportional* and *offline sampling*. 

Equal and proportional sampling is called online sampling in the code and share parameters. To use it during training use the following set of parameters:

* --online-sampling : activates online sampling
* --online-sample-prop 0.xx : sets the proportion of the training data used in each epoch to 0.xx
* --online-sampling-equal : turns on *equal* sampling, i.e. we aim to use an equal amount of trining instances for each class (given that enough instances of each class are present in the training data). The default is to use *proportional* sampling, where we use the distribution from the dev data, for choosing samples.

Overall we had the best results on the shared task with equal sampling.


**Offline sampling**
 --offlineSampling True : activates the offline sampling option.

 The offline sampling requires you to precompute your samples and feed the location as an additional parameter:

 --samples-path /directorywithyoursamples/

 Samples can be created using the offlinesampling.py script. 

 This sampling option is fast and cheap in terms of memory. The performance is on par with proportional sampling.

### Simple domain adaptation

For the DiscoMT 2017 submission we use a simple type of domain adaptation, also known as fine tuning, where we start training with all data, and continue to train with only in-domain data. In order to activate this, first train the system as normal, then train the system with only the desired data using the following parameters:

* --continue : activates the continuation training of an existing model
* --continueModel : the (already trained) model that should be read in
* --params : the param file written during original training

All parameters directly related to the network are also read from the parameter file, whereas parameters realted to training needs to be given. 


## Citation

If you make use of this software for research purposes, we'll appreciate if you cite the following:

```
@InProceedings{Stymnediscomt17,
author = {Sara Stymne and Sharid Lo{\'{a}}iciga and Fabienne Cap},
title = {A {BiLSTM}-based System for Cross-lingual Pronoun Prediction},
booktitle = {Proceedings of the Third Workshop on Discourse in Machine Translation},
year = {2017},
Address = {Copenhagen, Denmark},
publisher = {Association for Computational Linguistics},
pages     = {47--53}
}
```

## Acknowledgements

We would like to thank Eliyahu Kiperwasser and Miryam de Lhoneux for sharing their code and for valuable discussions. Our code partly builds on:

* BIST parser by Eli Kiperwasser and Yoav Goldberg: https://github.com/elikip/bist-parser
* uuparser by Miryam de Lhoneux et al.: https://github.com/UppsalaNLP/uuparser


## License

This software is released under the terms of the Apache License, Version 2.0.


## Contact

For questions and usage issues, please contact sara.stymne@lingfil.uu.se
