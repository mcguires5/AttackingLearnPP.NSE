classdef SINDy < handle
% @Author: delengowski
% @Date:   2019-01-24 18:01:17
% @Last Modified by:   delengowski
% @Last Modified time: 2019-04-04 17:25:42
% This class is used for creating SINDy objects to perform sparse dynamics
%
% (S)parse
% (I)dentification of
% (N)onlinear
% (Dy)namics
%
% Is code written for the works of
% S.L.Brunton, J.L.Proctor,and J.N.Kutz,"Discovering governing equations from data by sparse identification of nonlinear dynamical systems,"
% Proceedings of the National Academy of Sciences,vol.113,no.15,pp.3932-3937,2016.
%
% Additionally uses files from the following sources:
% 1.) http://faculty.washington.edu/kutz/page26/
% 2.) https://in.mathworks.com/matlabcentral/fileexchange/26190-vchoosek
% 3.) https://in.mathworks.com/matlabcentral/fileexchange/28340-nsumk
% 4.) https://in.mathworks.com/matlabcentral/fileexchange/14317-findjobj-find-java-handles-of-matlab-graphic-objects
% 
% SINDy Properties:
%	data - The data whose dynamics we wish to learn (Required Constructor Arg)
%	dt - The time step between data points (Required Constructor Arg)
%	startTime - The start time of the data (Required Constructor Arg)
%	endTime - The end time of the data (Required Constructor Arg)
%	numTimeStepsToPredict - The time step past endTime to predict to (Required Constructor Arg)
%	derivatives - The derivatives of the data, (Required Constructor Arg, will be computed otherwise)
%	useTVRegDiff - Use TVRegDiff.m to calculate derivatives if not included in constructor 
%	lambda - Thresholding parameter for sparse regression
%	polyOrder - Maximum polynomial degree for constructing candidate functions
%	useSine - Use sine/cosine in candidate function library
%	sineMultiplier - Maximum sine/cosine multiplier in candidate function library
%	useExp - Use positive and negative exponentials in candidate function library
%	expMultiplier - Maximum exponential multiplier in candidate function library
%	useCustomPoolData - Use customPoolData function or Brunton supplied poolData for building library
%	nonDynamical - Indication is data is nondynamical or not. Useful for 1D data
%	iter - Number of iterations for TVRegDiff
%	alph - Regularization parameter for TVRegDiff
%	ep - Conditioning parameter for TVRegDiff
%	scale - Indication of data scale for TVRegDiff
%	plotflag - Plotting option for TVRegDiff
%	diagflag - Diagnostics option for TVRegDiff
% SINDy methods:
% buildModel - Method for building SINDy model.
% displayLibrary - Displays coefficients for library of candidate functions from sparse regression.
% displayModel - Displays a plot of the learned model with its extended time predictions along with the input data.
% displayDerivatives - Displays a plot of the derivatives either fed into constructor or calculated using gradient()/TVRegDiff.
% Examples:
% See readme at https://github.com/Delengowski/ClinicSpring19 for examples of use.
	properties
		% Data - a matrix of the data whose dynamics is wished to be learned. The rows of data are the
		% time of the data and columns are the dimensions. Must be a numeric matrix containing all 
		% real numbers that are not NaNs or Inf. Requires at least 3 points of time series data. 
		% Required argument in the constructor. 
		data(:,:) double {mustBeTimeSeries(data,'data'),mustBeFinite, mustBeNonsparse, mustBeNumeric, mustBeReal} = [0 0 0;0 0 0];
		% dt - a double whose value is a positive, finite real scalar that represents the time in between 
		% points of the time series data. Required argument in the constructor.
        dt(1,1) double {mustBePositive, mustBeFinite} = 0.1;
        % startTime - a double whose value is a positive, finite real scalar that is the start time
        % of the time series data. Required argument in the constructor.
        startTime(1,1) double {mustBeNonnegative, mustBeFinite} = 1;
        % endTime - a double whose value is a positive, finite real scalar that is the end time
        % of the time series data. Must be greater than the startTime of the data. Additionally,
        % the time vector represented by startTime:dt:endTime must be column vector whose number 
        % of rows is equal to the number of rows of the data. Required argument in the constructor.
        % Initialized as 2.
    	endTime(1,1) double {mustBePositive, mustBeFinite} = 2; 
    	% numTimeStepsToPredict - A double whose value is a positive, finite real scalar that is the number
    	% of time steps past endTime to generate the model. The time vector represented by 
    	% startTime:dt:endTime+numTimeStepsToPredict must be a column vector whose number of rows is equal to 
    	% the number of rows of generated model. Required argument in the constructor.
        numTimeStepsToPredict(1,1) double {mustBeNonnegative, mustBeFinite} = 1;
        % derivatives - A matrix of the derivatives of the data. Must have all finite numbers. Is an optional
        % argument in the constructor. If not included derivatives will either be calculated using built in
        % gradient() function or will be computed using TVRegDiff depending on the value of useTVRegDiff.
        derivatives(:,:) double {mustBeTimeSeries(derivatives,'derivatives'), mustBeFinite, mustBeNonsparse, mustBeNumeric, mustBeReal} = [0 0 0;0 0 0];
        % useTVRegDiff - a logical value indicating whether to calculate derivatives of the data using 
        % gradient() function or TVRegDiff.m, Initialized as 0. 
        useTVRegDiff(1,1) logical {mustBeInteger} = 0
        % lambda - a double positive finite scalar that is used in the sparse regression for learning 
        % candidate function coefficients. The sparse regression is a thresholded least squares problem.
        % All function coefficients less than lambda are zeroed out during the regression. Initialized 
        % as 2e-6.
        lambda(1,1) double {mustBePositive, mustBeFinite} = 2e-3
        % polyOrder - an integer double that must be greater than or equal to 1. Is the maximum
        % polynomial order when constructing polynomial candidate functions. 
        polyOrder(1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(polyOrder,1), mustBeFinite} = 5
        % useSine - a logical indicating whether to build candidate function with sine/cosine.
        useSine(1,1) logical {mustBeInteger, mustBeFinite} = 0
        % sineMultiplier - an integer double that must be greater than or equal to 1. Is the maximum
        % multiplier to go up to when constructing candidate functions with sinusoids. 
        sineMultiplier(1,1) double {mustBeInteger, mustBePositive, mustBeFinite} = 10
        % useExponentials - a logical indicating whether to build candidate function with positive
        % and negative power exponentials. 
        useExp(1,1) logical {mustBeInteger, mustBeFinite} = 0
        % expMultiplier - an integer double that must be greater than or equal to 1. Is 
        % the maximum multiplier to go up to when constructing candidate functions with exponentials.
        expMultiplier(1,1) double {mustBeInteger, mustBePositive, mustBeFinite} = 10
        % useCustomPoolData - a logical indicating whether to use custom files for building library
        % of candidate functions Theta. This custom implementation differs from Brunton et el's 
        % implementation by allowing greater flexibility at a slightly slower computational time.
        % When using customPoolData polyOrder is not restricted to 5 and the sine/cosine multiplier
        % can go above 10. Additionally, allows for exponentials to be included. 
        useCustomPoolData(1,1) logical {mustBeInteger, mustBeFinite} = 1
        % nonDynamical - A logical indicating whether the data is nondynamical or not. Very useful for 
        % 1D data fitting that is nondynamical. 
        nonDynamical(1,1) logical {mustBeInteger, mustBeFinite} = 1;
        % iter - An integer double that is the number of iterations to run the TVRegDiff algorithm for.
        % Number of iterations to run the main loop.  A stopping condition based on the norm of the gradient vector g
		% below would be an easy modification.  No default value.
        iter(1,1) double {mustBeInteger, mustBeFinite} = 10
        % alph - A positive double for TVRegDiff. Regularization parameter.  This is the main parameter to fiddle with.  
        % Start by varying by orders of magnitude until reasonable results are obtained.  A value to the nearest power of
        % 10 is usally adequate. No default value.  Higher values increase regularization strenght and improve conditioning.
		alph(1,1) double {mustBePositive, mustBeFinite} = 0.00002
		% ep - A positive double used for TVRegDiff. Parameter for avoiding division by zero. Default value is 1e-6. Results
		% should not be very sensitive to the value. Larger values improve conditioning and therefore speed, while smaller 
		% values give more accurate results with sharper jumps. 
		ep(1,1) double {mustBePositive, mustBeFinite} = 1e12
		% scale - A string for an option in TVRegDiff. "larger" or "small". Default is "small". "small" has somewhat better 
		% boundary behavior, but becomes unwieldy for data larger than 1000 entries or so. 'large' has simpler numerics but is
		% more efficient for large-scale problems. 'large' is more readily modified for higher-order derivatives, since the 
		% implicity differentiation matrix is square. 
 		scale(1,1) string {mustBeMember(scale,["large" "small"])} = "small"
		% plotflag - A logical for TVRegDiff. Flag whether to display plot at each iteration. Default is 0 (no). Use, but adds
		% significant run time. 
		plotflag(1,1) logical {mustBeInteger, mustBeFinite} = 0
		% diagflag - A logical for TVRegDiff. Flag whether to display diagnostics at each iteration. Default is 0 (no). Use for 
		% diagnosing preconditioning problems. When tolerance is not met, an early iterate being best is more worrying than a 
		% large relative residual. 
		diagflag(1,1) logical {mustBeInteger, mustBeFinite} = 0
    end
    %properties(SetAccess = private)
        %model(:,:) double
        %modelTimeVector(:,1) double
        %learnedFunctions table 
    %end
	methods % Start public methods
		function obj = SINDy() % Constructor 
			addpath(fullfile('SINDy','sparsedynamics','sparsedynamics','utils'));
			addpath(fullfile('SINDy','nsumk'));
			addpath(fullfile('SINDy','VChooseK'));
			addpath(fullfile('SINDy','findjobj'));
        end
		function [model, modelTimeVector, learnedFunctions] = buildModel(obj,data,dt,startTime,endTime,numTimeStepsToPredict,derivatives)
			% buildModel - Builds a model and populates the properties model, modelTimeVector, and learnedFunctions.
			% Model is a matrix whose rows is a point in time and columns are the dimensions. modelTimeVector is a 
			% column vector of the time steps, and learnedFunctions is a matlab table obj holding the coefficients 
			% for each function in each dimension. 
			% Example: obj.buildModel;
            obj.data = data;
            obj.dt = dt;
            obj.startTime = startTime;
            if endTime > obj.startTime % check to make sure endTime is not before startTime
                obj.endTime = endTime;
            else
                error('endTime must be greater than startTime');
            end
            obj.endTime = endTime;
            obj.numTimeStepsToPredict = numTimeStepsToPredict;
            if nargin == 6
                [numTimeSteps, numDims] = size(obj.data);
                derivativesNotIncluded = true;
                obj.derivatives = getDerivatives(obj,numDims,numTimeSteps,derivativesNotIncluded);
            end
            if nargin == 7
                if all(size(derivatives) == size(obj.data))
                    obj.derivatives = derivatives;
                else
                    error('derivatives must be same size as data');
                end
            end
			[numTimeSteps, numDims] = size(obj.data);
			Theta = buildLibrary(obj,numDims);
			Xi = performSparseDynamics(obj,Theta,numDims);
            learnedFunctions = createLearnedFunctionsTable(obj,Xi,numDims);
			[model, modelTimeVector] = performSparseGalerkin(obj,Xi,numDims,numTimeSteps);
        end
        %function displayLibrary(obj)
        	%% displayLibrary - Displays the library of learned function coefficients in a UI table. Each column
        	%% is the the derivative in that dimension. Each row is 1 function. The coefficients is how 
        	%% much that function contributes that the derivative in that dimension. 
        	%% Example: obj.displayLibrary;
        	%derivatives = obj.learnedFunctions.Properties.VariableNames;
        	%Functions = obj.learnedFunctions.Properties.RowNames;
        	%for iDim = 1:length(derivatives)
				%derivatives{iDim} = ['<HTML><sup>d</sup>&frasl;<sub>dt</sub> d<sub>',num2str(iDim),'</sub></HTML>'];
			%end
			%for iFunc = 1:length(Functions)
				%if Functions{iFunc}(1) == 'd'
					%Functions{iFunc} = insertAfter(Functions{iFunc},'d','<sub>');
					%Functions{iFunc} = insertBefore(Functions{iFunc},'^','</sub>');
					%Functions{iFunc} = insertAfter(Functions{iFunc},'(','<sup>');
					%Functions{iFunc} = insertBefore(Functions{iFunc},')','</sup>');
					%Functions{iFunc} = erase(Functions{iFunc},'^');
					%Functions{iFunc} = erase(Functions{iFunc},'(');
					%Functions{iFunc} = erase(Functions{iFunc},')');
					%Functions{iFunc} = erase(Functions{iFunc},'*');
					%Functions{iFunc} = ['<HTML>',Functions{iFunc},'</HTML>'];
				%end
				%if Functions{iFunc}(1) == 'e'
					%Functions{iFunc} = insertAfter(Functions{iFunc},'d','<sub>');
					%Functions{iFunc} = insertBefore(Functions{iFunc},')','</sub>');
					%Functions{iFunc} = insertAfter(Functions{iFunc},'(','<sup>');
					%Functions{iFunc} = insertAfter(Functions{iFunc},')','</sup>');
					%Functions{iFunc} = erase(Functions{iFunc},'^');
					%Functions{iFunc} = erase(Functions{iFunc},'(');
					%Functions{iFunc} = erase(Functions{iFunc},')');
					%Functions{iFunc} = erase(Functions{iFunc},'*');
					%Functions{iFunc} = ['<HTML>',Functions{iFunc},'</HTML>'];
				%end
				%if contains(Functions{iFunc},'cos') || contains(Functions{iFunc},'sin')
					%Functions{iFunc} = insertAfter(Functions{iFunc},'d','<sub>');
					%Functions{iFunc} = insertBefore(Functions{iFunc},')','</sub>');
					%Functions{iFunc} = erase(Functions{iFunc},'*');
					%Functions{iFunc} = ['<HTML>',Functions{iFunc},'</HTML>'];
				%end
            %end
            %f1 = figure;
            %t = uitable('Data',obj.learnedFunctions{:,:},'ColumnName',derivatives,...
            %'RowName',Functions,'Units', 'Normalized', 'Position',[0, 0, 1, 1],'FontSize',15);
            %FontSize = 8;
%%             hs = '<html><font size="+2">'; %html start
%% 			he = '</font></html>'; %html end
%% 			cnh = cellfun(@(x)[hs x he],derivatives,'uni',false); %with html
%% 			rnh = cellfun(@(x)[hs x he],Functions,'uni',false); %with html
%% 			set(t,'ColumnName',cnh,'RowName',rnh) %apply
            %%get the row header
			%jscroll=findjobj(t);
			%rowHeaderViewport=jscroll.getComponent(4);
			%rowHeader=rowHeaderViewport.getComponent(0);
			%height=rowHeader.getSize;
			%rowHeader.setSize(40,180)
			%%resize the row header
			%newWidth=125; %100 pixels.
			%rowHeaderViewport.setPreferredSize(java.awt.Dimension(newWidth,0));
			%height=rowHeader.getHeight;
			%newHeight = 200;
			%rowHeader.setPreferredSize(java.awt.Dimension(newWidth,height));
			%rowHeader.setSize(newWidth,newHeight);
        %end
        %function displayModel(obj,style)
        	%% displayModel - Plots the modeled data as well as the original data. Will plot traditional 1D, 2D, or 3D
        	%% for data of that size. Or it will display stacked plots of each dimension vs time if specified by the 
        	%% argument <style>. If data is greater than 3 dimensions then it will do stacked plot for each dimension 
        	%% regardless of <style>. 
        	%% Example: obj.displayModel('traditional'); 
        	%% Example: obj.displayModel('stacked');
        	%numDims = size(obj.data,2);
        	%if numDims > 3
        		%style = 'stacked';
            %end
        	%switch style
        		%case 'stacked'
        			%dimensions = 1:numDims;
        			%dimensions = "d" + dimensions;
        			%variables = ["t" dimensions];
        			%T = array2table([obj.modelTimeVector obj.model],'VariableNames',cellstr(variables));
        			%NANS = NaN([(size(obj.model(:,1),1) - size(obj.data(:,1),1)) 1],'double'); 
        			%for iDim = 1:size(obj.data,2)
        				%T.(['dataD',num2str(iDim)]) = [obj.data(:,iDim);NANS];
        				%T = mergevars(T,[string(['d',num2str(iDim)]) string(['dataD',num2str(iDim)])],...
                                      %'NewVariableName',['d',num2str(iDim)]);
    				%end
    				%f1 = figure;
        			%P = stackedplot(T,cellstr(dimensions),"xvar","t");
        			%for iDim = 1:numDims
        				%P.AxesProperties(iDim).LegendLabels{1} = [P.AxesProperties(iDim).LegendLabels{1}(1:end-1) 'data'];
        				%P.AxesProperties(iDim).LegendLabels{2} = [P.AxesProperties(iDim).LegendLabels{2}(1:end-1) 'model'];
        				%P.LineProperties(iDim).LineWidth =[4,2];
        				%P.LineProperties(iDim).LineStyle = {'-',':'};
    				%end
%
%
				%case 'traditional' 
		        	%if numDims == 1
%
		    		%elseif numDims == 2
%
					%elseif numDims == 3
%
					%end					
			%end
        %end
    end % End Public methods
	methods (Access = private) % Start private methods
		function [Theta] = customPoolData(obj,data,numDims)
			if (obj.nonDynamical == true) && size(data,1) > 1 % Convert data to time arrays if its not being fed to from ODE45, this is for building initial library
				data = (obj.startTime:obj.dt:obj.endTime)';
				data = repmat(data,1,numDims);
			end
			if (obj.nonDynamical == true) && numDims > 1  && size(data,1) == 1 % If we get data from ODE45, which is 1 point, repmat for each dim. This is for getting derivatives for ODE45
				data = repmat(data,1,numDims);
			end

			numObs = size(data,1);
			% Preallocate for Theta, get number of functions and create zero fill matrix
			numFuncs = 1;
            polynomialOrders = 1:obj.polyOrder;
            powers = cell(size(polynomialOrders,1),1);
			numIntegerFactors = cell(size(polynomialOrders,1),1);
			for iPower = polynomialOrders
                [numIntegerFactors{iPower},powers{iPower}] = nsumk(numDims,iPower);
				numFuncs = numFuncs + numIntegerFactors{iPower}; 
            end
			if obj.useSine == 1
				numFuncs = numFuncs + 2*obj.sineMultiplier*numDims;
			end
			if obj.useExp == 1
				numFuncs = numFuncs + 2*obj.expMultiplier*numDims;
			end
			Theta = zeros(numObs,numFuncs);

			% Constant 
			counter = 1;
			Theta(:,1) = ones(numObs,1);

			% Polynomials 
			for iPower = polynomialOrders
				for iIntergerFactor = 1:numIntegerFactors{iPower}
                    integerFactor = powers{iPower}(iIntergerFactor,:);
					counter = counter + 1;
					blah = zeros(numObs,sum(integerFactor));
                    counter2 = 1;
                    for iDim = 1:numDims
						if integerFactor(iDim) == 0

                        elseif integerFactor(iDim) == 1
                            blah(:,counter2) = data(:,iDim);
                            counter2 = counter2 + 1;
                        else % integerFactor(iDim) > 1
							upperIndx = counter2 + integerFactor(iDim) - 1;
							blah(:,counter2:upperIndx) = repmat(data(:,iDim),1,integerFactor(iDim));
							counter2 = upperIndx;
                            counter2 = counter2 + 1;
						end
					end
					Theta(:,counter) = prod(blah,2);
				end
			end

			% Exponentials
			if obj.useExp == 1
      			multipliers = 1:obj.expMultiplier;
       			multipliers = repmat(multipliers,numObs,1);
       			for iDim = 1:numDims
                    counter = counter + 1;
                    upperIndx = counter + size(multipliers,2) - 1;
       				Theta(:,counter:upperIndx) = exp(multipliers.*data(:,iDim));
                    counter = upperIndx;
       				counter = counter + 1;
                    upperIndx = counter + size(multipliers,2) - 1;
       				Theta(:,counter:upperIndx) = exp(-1*multipliers.*data(:,iDim));
                    counter = upperIndx;
			 	end
			end

			% Sine & Cosine
		    if obj.useSine == 1
      			multipliers = 1:obj.sineMultiplier;
       			multipliers = repmat(multipliers,numObs,1);
   				for iDim = 1:numDims
       				counter = counter + 1;
                    upperIndx = counter + size(multipliers,2) - 1;
       				Theta(:,counter:upperIndx) = sin(multipliers.*data(:,iDim));
                    counter = upperIndx;
       				counter = counter + 1;
                    upperIndx = counter + size(multipliers,2) - 1;
       				Theta(:,counter:upperIndx) = cos(multipliers.*data(:,iDim));
                    counter = upperIndx;
				end
    		end
		end
		function learnedFunctions = customPoolDataLIST(obj,Xi,numDims)
			% Preallocate for Theta
			numFuncs = 1;
		    polynomialOrders = 1:obj.polyOrder;
		    powers = cell(size(polynomialOrders,1),1);
			numIntegerFactors = cell(size(polynomialOrders,1),1);
			for iPower = polynomialOrders
		        [numIntegerFactors{iPower},powers{iPower}] = nsumk(numDims,iPower);
				numFuncs = numFuncs + numIntegerFactors{iPower}; 
		    end
			if obj.useSine == 1
				numFuncs = numFuncs + 2*obj.sineMultiplier*numDims;
			end
			if obj.useExp == 1
				numFuncs = numFuncs + 2*obj.expMultiplier*numDims;
			end
			Functions = strings(numFuncs,1);

			% Constant 
			counter = 1;
			Functions(counter) = "1";
			dimensionStrings = "d"+string(1:numDims);

			% Polynomials 
			for iPower = polynomialOrders
				for iIntergerFactor = 1:numIntegerFactors{iPower}
					counter = counter + 1;
					Functions(counter) = strjoin("d" + string([1:numDims]) + "^(" + string(powers{iPower}(iIntergerFactor,:)) + ")",'*');
				end
			end

			% Exponentials
			if obj.useExp == 1
		        multipliers = 1:obj.expMultiplier;
		        for iDim = 1:numDims
		            counter = counter + 1;
		            upperIndx = counter + size(multipliers,2) - 1;
		            Functions(counter:upperIndx) = "e^(" + string(multipliers) + "*" + dimensionStrings(iDim) + ")"; 
		            counter = upperIndx;
		            counter = counter + 1;
		            upperIndx = counter + size(multipliers,2) - 1;
		            Functions(counter:upperIndx) = "e^(-" + string(multipliers) + "*" + dimensionStrings(iDim) + ")"; 
		            counter = upperIndx;
		        end
			end

			% Sine & Cosine
		    if obj.useSine == 1
					multipliers = 1:obj.sineMultiplier;
					for iDim = 1:numDims
						counter = counter + 1;
		                upperIndx = counter + size(multipliers,2) - 1;
						Functions(counter:upperIndx) = "sin(" + string(multipliers) + "*" + dimensionStrings(iDim) + ")"; 
		                counter = upperIndx;
						counter = counter + 1;
		                upperIndx = counter + size(multipliers,2) - 1;
						Functions(counter:upperIndx) = "cos(" + string(multipliers) + "*" + dimensionStrings(iDim) + ")";
                        counter = upperIndx;
				end
			end
		    for iDim = 1:numDims
                field = cell(1,numDims);
                field{iDim} = char(strcat(dimensionStrings(iDim),"dot"));
		    	learnedFunctions.(field{iDim}) = Xi(:,iDim);
			end
			learnedFunctions = struct2table(learnedFunctions);
			learnedFunctions.Properties.RowNames = cellstr(Functions);		
		end
    	function derivatives = getDerivatives(obj,numDims,numTimeSteps,derivativesNotIncluded)
			if derivativesNotIncluded % I.e. if derivatives weren't included in the constructor
				if obj.useTVRegDiff == true
			        % If we use TVRegDiff to calculate derivatives, make sure the 
			        % appropriate hyperparameters, functionOptions, and printingFlags
			        % are set as they're not set in constructor. 
			        u0 = gradient(obj.data',obj.dt)';
					derivatives = zeros(numTimeSteps,numDims);
                    for iDim = 1:numDims
						derivatives(:,iDim) = TVRegDiff(obj.data(:,iDim),...
											   obj.iter,...
											   obj.alph,...
											   u0,...
											   obj.scale,...
											   obj.ep,...
											   obj.dt,...
											   obj.plotflag,...
											   obj.diagflag);
                    end
                    if all(size(derivatives) == size(obj.data))

                    else
                        error(['derivatives must be same size as data,',...
                              'check TVRegDiff.m documentation']);
                    end
				else % Use gradient()
					derivatives = gradient(obj.data',obj.dt)';
				end
			else
				derivatives = obj.derivatives;
			end
    	end
		function Theta = buildLibrary(obj,numDims)
			if obj.useCustomPoolData == true
				if computer ~= "PCWIN64"
					error(['mex files were compiled on a 64 bit windows system',...
					  ' please perform "mex VChooseK.c" to use customPoolData.m',...
					  newline,'Valid C compilers can be found at ',...
					  'https://in.mathworks.com/support/requirements/supported-compilers.html',...
					  newline,'Or set <object>.useCustomPoolData = "no"']);
				end
				[Theta] = customPoolData(obj,obj.data,numDims);
			else
				if obj.polyOrder > 5 
					warning(['using non mex version poolData.m, ',...
					     'polyOrder reduced to 5,sineMultiplier set to 10, ',...
					     'no exponentials used.'])
					obj.polyOrder = 5;
				end
				[Theta] = poolData(obj.data,numDims,obj.polyOrder,obj.useSine);
			end
    	end
    	function Xi = performSparseDynamics(obj,Theta,numDims)
			Xi = sparsifyDynamics(Theta,obj.derivatives,obj.lambda,numDims);
		end
        function [learnedFunctions] = createLearnedFunctionsTable(obj,Xi,numDims)
            if obj.useCustomPoolData == true
	            learnedFunctions = customPoolDataLIST(obj,Xi,numDims);
    		else
    			yin = cellstr("d" + (1:numDims));
    			tempLearnedFunctions = poolDataLIST(yin,Xi,numDims,obj.polyOrder,obj.useSine);
                for iDim = 1:numDims
    				temp.(join(tempLearnedFunctions{1,iDim+1},'')) = vertcat(tempLearnedFunctions{2:end,iDim+1});
                end
				learnedFunctions = struct2table(temp);
				dummyStrings = strings(1,size(tempLearnedFunctions,1)-1);
				for i = 2:size(tempLearnedFunctions,1)
					tempLearnedFunctions{i,1} = join(tempLearnedFunctions{i,1},'');
					dummyStrings(i-1) = tempLearnedFunctions{i,1};
				end
				learnedFunctions.Properties.RowNames = dummyStrings;
            end
        end
        function dy = customSparseGalerkin(t,y,ahat,numDims,obj)
			if obj.nonDynamical == true
			    y = t;
			end

			yPool = customPoolData(obj,y',numDims);
			dy = (yPool*ahat)';
		end
		function [model, modelTimeVector] = performSparseGalerkin(obj,Xi,numDims,numTimeSteps)
            tSpan = (obj.startTime:obj.dt:obj.endTime); 
            if size(tSpan,2) == numTimeSteps % Make sure number of timesteps is equal for the time vector and data
                
            else
                error(['time vector created by startTime,dt,endTime does',...
                    'does not match with number of time steps in data,',...
                    'check values']);
            end
            tf = obj.endTime + obj.numTimeStepsToPredict;
            initialConditions = obj.data(1,:);
            if obj.useCustomPoolData == true
                sol = ode45(@(t,x)customSparseGalerkin(t,x,Xi,numDims,obj),tSpan,initialConditions);
                if obj.numTimeStepsToPredict > 0
	                ext_sol = odextend(sol,@(t,x)customSparseGalerkin(t,x,Xi,numDims,obj),tf);
                    model = deval(ext_sol,obj.startTime:obj.dt:tf)';
                else 
                	
                	model = deval(sol,obj.startTime:obj.dt:tf)';
                end
            modelTimeVector = (obj.startTime:obj.dt:tf)';
            else
				sol = ode45(@(t,x)sparseGalerkin(t,x,Xi,...
                                 obj.polyOrder,...
                                 obj.useSine),...
		                         tSpan,...
		                         initialConditions);
				if obj.numTimeStepsToPredict > 0
		            ext_sol = odextend(sol,@(t,x)sparseGalerkin(t,x,Xi,...
		                                             obj.polyOrder,...
		                                             obj.useSine),...
		                                			 tf);
		            model = deval(ext_sol,obj.startTime:obj.dt:tf)';
                else
            	model  = deval(sol,obj.startTime:obj.dt:tf)';
                end
            modelTimeVector = (obj.startTime:obj.dt:tf)';
            end
		end
	end % End Private Methods
end % End Classdef
function mustBeTimeSeries(a,property)
    if (size(a,1) > 1 && size(a,2) > 1) || (iscolumn(a) && size(a,1) > 1)

    else
        error(['Value assigned to ',property,' must be a matrix or column vector with at least 2 elements'])
    end
end