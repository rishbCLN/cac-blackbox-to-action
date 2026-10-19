# Project-local search paths for the official IEEE Access fonts and class.
# Overleaf and local latexmk both read this configuration.
my $path_sep = $^O eq 'MSWin32' ? ';' : ':';
for my $var ('TEXINPUTS', 'TFMFONTS', 'T1FONTS', 'TEXFONTMAPS') {
    $ENV{$var} = './paper/ieee-access//' . $path_sep . ($ENV{$var} // '');
}
$pdf_mode = 1;
