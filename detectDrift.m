% @Author: Sean Mcguire
% @Date:   2018-09-15 22:51:39
% @Last Modified by:   Delengowski_Mobile
% @Last Modified time: 2019-04-02 11:08:44
%
% FAST_COMPOSE: Implementation of FAST COMPOSE
% "COMP"acted "O"bject "S"ample "E"xtraction
%
% Description:
%
% Syntax:  [drift, Fail] = detectDrift(train, curData, SINDY, StdevThresh)
%
% Inputs:
%        1.) train - all training data points used to characterize
%        distribution
%        2.) curData - observed data being compared against SINDy all
%        time steps
%        3.) SINDY - the learned function from SINDy
%        4.) StdevThresh - tolerence of algorithm (hyperparameter) to
%        detect drift
%
%    classificationAlgorithm - A classification algorithm, FAST COMPOSE
%                   works such that any algorithm will work. 
%                   Available classification algorithms:
%                   1. 'Cluster and Label'
% Outputs:
%        1.) drift - True or False (i.e. was a drift detected at all)
%        2.) Fail - logical vector indicating at what timesteps did a drift
%        occur
%
% Example: 
%    Line 1 of example
%    Line 2 of example
%    Line 3 of example
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
% 
% Author: Sean Mcguire
% email: mcguires5@students.rowan.edu; 
 
%------------- BEGIN CODE --------------
function [drift, Fail] = detectDrift(train, curData, SINDY, StdevThresh)
 
[m,n] = size(train);
[m2,n2] = size(curData);
% Normalize all points to time step 1
clusteredPos = zeros(m, n);
stdev = zeros(n,1);
for dim = 1:n
    clusteredPos(:, dim) = train(:,dim) - SINDY(1:m,dim);
    stdev(dim) = std(clusteredPos(:,dim));
%     if stdev(dim) == 0
%         stdev(dim) = 1;
%     end
end
%disp('-----')
%disp(['Actual Mean: ',num2str(curData(end,:))])
%disp(['SINDy Mean: ',num2str(SINDY(end,:))])
Fail = zeros(1,n);
drift = false;
for dim = 1:n
    for i = 1:m2
        % See where it falls farther than 4 stdev
        if curData(i, dim) == SINDY(i, dim)
            drift = false;
        elseif curData(i, dim) > SINDY(i, dim) + StdevThresh*stdev(dim) || curData(i, dim) < SINDY(i, dim) - StdevThresh*stdev(dim)
            if Fail(dim) == 0
                %disp(['Divergence from SINDy Occured at timestep ', num2str(i), ' in dimension ', num2str(dim)])
                drift = true;
                Fail(dim) = i;
            end
        end
    end
end
%if isempty(Fail)
%   disp(['No Divergence from SINDy  Detected'])
%end
end
%------------- END CODE --------------