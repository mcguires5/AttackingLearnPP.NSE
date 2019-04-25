function [] = pathsForCode()
    clearvars
    close all
    addpath(fullfile('.','SINDy'));
    addpath(fullfile('.','Learn++NSE'));
    addpath(fullfile('.','ConceptDriftData'));
    addpath(genpath(fullfile('.','advlearn')));
    p = py.sys.path;
    insert(p,int32(0), pwd + "\advlearn\")
    insert(p,int32(0), pwd + "\advlearn\advlearn\attacks\poison\")
    insert(p,int32(0), pwd + "\advlearn\advlearn\attacks\")
end