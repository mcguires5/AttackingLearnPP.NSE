function eps2pdf(directory)

% convert all EPS to PDF
eps = dir([ directory '/*.eps']);

epstopdf ='/usr/local/texlive/2012/bin/x86_64-darwin/epstopdf'; %command name to convert from eps to pdf files

for i=1:numel(eps)
    system([epstopdf ' ' directory '/' eps(i).name ' --outfile=' directory '/' eps(i).name(1:end-3) 'pdf']);
end
