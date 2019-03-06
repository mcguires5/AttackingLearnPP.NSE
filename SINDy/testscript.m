% @Author: Delengowski
% @Date:   2019-01-27 18:54:05
% @Last Modified by:   Delengowski_Mobile
% @Last Modified time: 2019-02-21 13:15:37

sindy = SINDy([(1:.01:10)'.^2 (1:.01:10)'.^2],.01,1,10,50);
sindy.lambda = 2e-4;
sindy.polyOrder = 5;
%sindy.useSine = 1;
%sindy.useExp = 1;
%sindy.sineMultiplier = 4;
%sindy.expMultiplier = 3;
sindy.buildModel;
sindy.displayLibrary;
sindy.displayModel('stacked');