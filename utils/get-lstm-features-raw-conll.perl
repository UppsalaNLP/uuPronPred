#!/usr/bin/perl -w

# Creates a file with the input format for the BiLSTM system,
# based on a discoMT-file, and a dependency parsed file in conll format

use strict;
use warnings;

if (@ARGV < 4) {
    die "Usage: get-raw-features.perl orig-file  output-file  language-pair parsed-file  [lc]\n";
}

my $lc = 0;

if (@ARGV > 4) {
    $lc = 1;
}

my $of = shift;  #orig
my $ff = shift;  #out features
my $dir = shift; # lang direction
my $pf = shift;  # parsed file


open (my $in, "<", $of) or die("could not open $of for reading\n");
open (my $fout, ">", $ff) or die("could not open $ff for writing\n");
open (my $pin, "<", $pf) or die("could not open $pf for reading\n");

my @classes;

if ($dir eq "en-de") {
    @classes = ("er", "sie", "es", "man", "OTHER");
}
elsif ($dir eq "de-en") {
    @classes = ("he", "she", "it", "they", "you", "this", "these", "there", "OTHER");
}
elsif ($dir eq "fr-en") {
    @classes = ("he", "she", "it", "they", "this", "these", "there", "OTHER");
}
else {  #default: en-fr
    @classes = ("ce", "elle", "elles", "il", "ils", "cela", "on", "OTHER");
}

my %classHash = map { $_ => 1 } @classes;


my $lineNum = 0;
my $docId = -1;
my @parse_lines;
my @head_nums;


while (my $line = <$in>) {

    $lineNum++;
	chomp $line;
    my @parts = split(/\t/, $line); 
    my @foreign_words = ( $lc ?  split(/\s+/, lc($parts[3])) : split(/\s+/, $parts[3]));
    my @source_words = ( $lc ?  split(/\s+/, lc($parts[2])) : split(/\s+/, $parts[2]));

	next if $parts[3] !~ /REPLACE/;
	
    my $parsed_sent = "";
    while (!eof($pin) && $parsed_sent ne $parts[2]) {	
		$parsed_sent = "";
		@parse_lines = ();
		@head_nums = ();
		my $first = "";
		while (<$pin>) {
			last if /^$/;      #stop if end-of-sentence
			my @parse_word = split(/\t/);
			$parsed_sent .= "$first$parse_word[1]";
			$first = " ";
			
			my $word = $lc ? lc($parse_word[1]) : $parse_word[1];
			push (@parse_lines, "$word|$parse_word[5]|$parse_word[11]|$parse_word[9]");
			push (@head_nums, "$parse_word[9]");		
		}
	}

	

	my @alignments = split(/\s+/, $parts[4]);
	my @pronouns = split(/ /, $parts[0]);
	my @xpronouns = split(/ /, $parts[1]);
	my $exCounter = 0;

	
	#index 1:word, 5:POS, 7:morph, 9:head, 11:label
	for (my $i=0; $i<@parse_lines; $i++) {
		my $headWord = "root";
		if ($head_nums[$i] > 0) {
			$headWord = $lc ? lc($source_words[$head_nums[$i]-1]) : $source_words[$head_nums[$i]-1];
		}

		print $fout  "$parse_lines[$i]|$headWord ";
	}
	print $fout  "\t";
	print $fout  $lc ? lc($parts[3]) : $parts[3];  #target words
	print $fout  "\n";


	my $foreign_index = 0;

	foreach (@foreign_words) {

		if (/replace_(\d+)/i) {
			my $sindex = $1;

			my $headIndex = $head_nums[$sindex];
			print $fout  "$pronouns[$exCounter] $xpronouns[$exCounter] $source_words[$sindex] $sindex $foreign_index $headIndex\n";

			$exCounter++;

		}
		$foreign_index++;
	}
	

}


