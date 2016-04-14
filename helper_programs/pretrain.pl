#!/usr/bin/perl -w
# pretrain.pl model_1_best.nn uzb-eng.trn.uzb uzb-eng.trn.eng uzb-eng.dev.uzb uzb-eng.dev.eng model_1_uzb_eng.nn

use strict;
$ENV{'LD_LIBRARY_PATH'}=qq{/usr/usc/cuda/7.0/lib64:$ENV{'LD_LIBRARY_PATH'}};

my ($parent, $trn1, $trn2, $dev1, $dev2, $child) = @ARGV;
my $usage = 'pretrain.pl parent-model trn1 trn2 dev1 dev2 child-model';
my $train = '/home/nlg-05/zoph/MT_Experiments/new_experiments_2/deniz_stuff/code/RNN_MODEL';
my $opts1 = '-l 0.5 -A 0.9 -d 0.5 -P -0.05 0.05 -L 100 -w 5 --clip-cell 50 1000 -m 128 --attention-model 1 --feed_input 1 --screen-print-rate 300 --random-seed 1';
my $opts2 = '--train-source-input-embedding true --train-source-RNN true --train-attention-target-RNN true --train-target-RNN true --train-target-input-embedding false --train-target-output-embedding false';

warn("Reading source vocab from $trn1\n");
my %trn1vocab;
open(TRN1, $trn1) or die $usage;
while(<TRN1>) { $trn1vocab{$_}++ for split; }
close(TRN1);
my @trn1vocab = sort { $trn1vocab{$b} <=> $trn1vocab{$a} } keys(%trn1vocab);

# for my $w (@trn1vocab) { print "$trn1vocab{$w}\t$w\n"; }

warn("Replacing source vocab from $parent to $child.last\n");
open(PM, $parent) or die $usage;
open(CM, ">$child.last") or die $usage;
$_ = <PM>; die unless /^(\d+) (\d+) (\d+) (\d+)$/; print CM;
my ($nlayer, $nhidden, $ntarget, $nsource) = ($1,$2,$3,$4);
die unless scalar(@trn1vocab) >= $nsource;
my $moption = " -M ".("0 " x ($nlayer-1))."1 1 ";
$_ = <PM>; die unless /^=+$/; print CM;
$_ = <PM>; die unless /^0 <UNK>$/; print CM;
while(<PM>) {
    if (/^=+$/) { print CM; last; }
    my ($i,$w) = split;
    print CM "$i $trn1vocab[$i-1]\n";
}
while(<PM>) { print CM; }
close(PM); close(CM);

my $cmd = qq{$train -n 1000 -C $trn1 $trn2 $child.last -B $child -a $dev1 $dev2 $opts1 $opts2 $moption};
warn("$cmd\n");
system($cmd);
