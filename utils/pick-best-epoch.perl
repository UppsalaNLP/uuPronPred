#!/usr/bin/perl -w

# Prints out the best epoch (based on the average of macro-R and accuracy)
# for the runs in a folder

use strict;
use warnings;

if (@ARGV < 2) {
    die "Usage: pick-best-epoch.perl directory max-iter lang\n";
}


my $dir = $ARGV[0];
my $max = $ARGV[1];
my $lang = $ARGV[2];

my $i = 1;

my $best_score=0;
my $best_epoch = -1;

my $best_detail;

while ($i <= $max) {

    my $file = "$dir/dev_epoch_$i.pron_out.res";
    open (IN, $file) or die("could not open $file for reading\n");
	#unless (open (IN, $file)) { 
	#	my $cmd = "parser-based/utils/evaluate-pronouns.perl  $dir/devdata.gold $dir/dev_epoch_$i.pron_out $lang > $dir/dev_epoch_$i.pron_out.res";
	#	system($cmd);
	#	open (IN, $file) or die("Could still not open the res file...\n");
    #}
    my ($acc, $macroR);
    while (<IN>) {
	if (/Micro-averaged:.* F1 ([0-9\.]+)/) {
	    $acc = $1;
	}
	if (/Macro-averaged R: ([0-9\.]+)/) {
	    $macroR = $1;
	    last;
	}
    }
    my $score = ($acc+$macroR)/2;
    if ($score > $best_score) {
		$best_score = $score;
		$best_epoch = $i;
		$best_detail = sprintf "%.4f/%.4f", $macroR, $acc;
    }    

    $i++;
    close IN;
}

print "Best epoch: $best_epoch\n$best_score\n$best_detail\n";

#Micro-averaged: P 0.72488 R 0.72488 F1 0.72488
#Macro-averaged R: 0.45906 F1: 0.44673
