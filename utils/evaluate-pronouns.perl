#!/usr/bin/perl -w


use strict;
use warnings;

open (my $sin, "<", $ARGV[0]) or die("could not open $ARGV[0] for reading\n");
open (my $gin, "<", $ARGV[1]) or die("could not open $ARGV[1] for reading\n");

#my $nClass = $ARGV[2]
my $lang = $ARGV[2];
my %numClasses = ("en-de" => 5, "de-en" => 9, "es-en" => 7, "en-fr" => 8);
my $nClass = $numClasses{$lang};


#reads files with one pronoun instance per line, and assumes the first 
# white-space separated token on each line is the prediction 

my %matchesG;
my %matchesP;
while (my $sl = <$sin>) {
    my $gl = <$gin>;

      my @sline = split(" ", $sl);
      my @gline = split(" ", $gl);

      $matchesG{$gline[0]}{$sline[0]}++;
      $matchesP{$sline[0]}{$gline[0]}++;
      if ($gline[0] ne $sline[0]) {
		  $matchesG{$gline[0]}{"FN"}++; 
		  $matchesP{$sline[0]}{"FP"}++; 
      }
}

my $totalF = 0.0;
my $totalR = 0.0;

my ($ttp,$tfp,$tfn) = (0.0, 0.0, 0.0);

while ( my ($gold, $preds) = each %matchesG ) {
    my $tp = exists $matchesG{$gold}{$gold}? $matchesG{$gold}{$gold} : 0;
    my $fn = exists  $matchesG{$gold}{"FN"} ? $matchesG{$gold}{"FN"} : 0;
    my $fp = exists  $matchesP{$gold}{"FP"} ?  $matchesP{$gold}{"FP"} : 0;
    $ttp += $tp;
    $tfp += $fp;
    $tfn += $fn;
    my $p = ($tp+$fp)!=0 ? $tp/($tp+$fp) : 0.0;
    my $r = $tp/($tp+$fn);
    my $f = $p+$r!=0 ? 2*$p*$r/($p+$r) : 0;
    $totalF += $f;
    $totalR += $r;
    printf "%10s: TP=%d FN=%d FP=%d; P %.3f R %.3f F1 %.3f\n", $gold, $tp, $fn, $fp, $p, $r, $f;
}

my $tp = $ttp/($ttp+$tfp);
my $tr = $ttp/($ttp+$tfn);
my $tf = 2*$tp*$tr/($tp+$tr);
printf "Micro-averaged: P %.5f R %.5f F1 %.5f\n", $tp, $tr, $tf;

printf "Macro-averaged R: %.5f F1: %.5f\n", $totalR/$nClass, $totalF/$nClass;
