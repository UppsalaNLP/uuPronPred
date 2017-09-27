#!/usr/bin/perl -w

# This scripts converts from the format of the BiLSTM
# to the official DiscoMT format

use strict;
use warnings;

if (@ARGV < 2) {
    die "Usage: lstm2disco.perl lstm-file orig-file\n";
}

my $lf = shift;
my $of = shift;

open(LIN, $lf) or die("could not open $lf for reading\n");
open (OIN, $of) or die("could not open $of for reading\n");



while (my $line = <OIN>) {
    my @parts = split(/\t/, $line);

    if ($line !~/REPLACE_/i) {   #no examples
		print $line;
		next;
    }
    
	my @prons =();

	my @ftokens = split(/ /, $parts[3]);
    foreach (@ftokens) {
		#print "$_ \n";
		if (/replace_(\d+)/i) {
			my $pred = <LIN>;
			chomp $pred;
			push (@prons, $pred);
		}
	}

	$parts[0] = join(" ", @prons);

    print join("\t", @parts);
}
