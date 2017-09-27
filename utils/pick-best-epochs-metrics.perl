#!/usr/bin/perl -w

# Prints out the best epoch (based on different metrics) 
# for the runs in a folder
#     average (macro-R, accuracy)
#     accuracy
#     macro-R


use strict;
use warnings;

if (@ARGV < 2) {
    die "Usage: pick-best-epoch-options.perl directory max-iter\n";
}


my $dir = $ARGV[0];
my $max = $ARGV[1];

my $i = 1;

my %best;
my @types = ("MR", "Acc", "Both");
foreach my $t (@types) {
    $best{$t}{"score"} = 0;
    $best{$t}{"epoch"} = -1;
    $best{$t}{"detail"} = "";
}

while ($i <= $max) {

    my $file = "$dir/dev_epoch_$i.pron_out.res";
    unless (open (IN, $file)) { # or die("could not open $file for reading\n");
		$i++;
		close IN;
		next;
    }
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
    my $average = ($acc+$macroR)/2;
    my $score = $average;
    if ($score > $best{"Both"}{"score"}) {
		$best{"Both"}{"score"} = $score;
		$best{"Both"}{"epoch"} = $i;
		$best{"Both"}{"detail"} = sprintf "%.4f/%.4f/%.4f", $average, $macroR, $acc;
    }
    $score = $acc;
    if ($score > $best{"Acc"}{"score"}) {
		$best{"Acc"}{"score"} = $score;
		$best{"Acc"}{"epoch"} = $i;
		$best{"Acc"}{"detail"} = sprintf "%.4f/%.4f/%.4f", $average, $macroR, $acc;
    }
    $score = $macroR;
    if ($score > $best{"MR"}{"score"}) {
		$best{"MR"}{"score"} = $score;
		$best{"MR"}{"epoch"} = $i;
		$best{"MR"}{"detail"} = sprintf "%.4f/%.4f/%.4f", $average, $macroR, $acc;
    }
    #print STDERR "$score\n";
    

    $i++;
    close IN;
}

printf "Best epoch Average: %i\n%.5f\n%s\n\n", $best{"Both"}{"epoch"}, $best{"Both"}{"score"}, $best{"Both"}{"detail"};
printf "Best epoch Macro-R: %i\n%.5f\n%s\n\n", $best{"MR"}{"epoch"}, $best{"MR"}{"score"}, $best{"MR"}{"detail"};
printf "Best epoch Accuracy: %i\n%.5f\n%s\n\n", $best{"Acc"}{"epoch"}, $best{"Acc"}{"score"}, $best{"Acc"}{"detail"};


#Micro-averaged: P 0.72488 R 0.72488 F1 0.72488
#Macro-averaged R: 0.45906 F1: 0.44673
