classdef libraryFunction < handle

	properties (SetAccess = private)
		functionHandle(1,1) {mustBefunctionHandle} = @(x) x
		dimensionVariables(1,:) {mustBeString(dimensionVariables,'dimensionVariables')} = "a"
		coefficientVariables(1,:) {mustBeString(coefficientVariables,'coefficientVariables')} = "a"
	end
	properties (SetAccess = private, Hidden = true)
		numDimVariables(1,1) {mustBeNumeric} = 1;
		numCoeffVariables(1,1) {mustBeNumeric} = 1;
	end
	properties
		coefficientValues(1,:) {mustBeCell} = {10:11,1:10};
	end
	methods
		function obj = libraryFunction(functionHandle,dimensionVariables,coefficientVariables,coefficientValues)
			obj.functionHandle = functionHandle;
			if nargin == 1

			elseif nargin == 4
				obj.dimensionVariables = dimensionVariables;
				obj.coefficientVariables = coefficientVariables;
				obj.coefficientValues = coefficientValues;
				obj.numDimVariables = length(dimensionVariables);
				obj.numCoeffVariables = length(coefficientVariables);
				if (nargin(functionHandle) == obj.numCoeffVariables + obj.numDimVariables)
					if obj.numCoeffVariables == length(coefficientValues)

					else
						error('Number of coefficientVariables must equal number of cells in coeffcientValues')
					end
				else
					error('Number of inputs to functionHandle must be equal to length of dimensionVariables plus length of coefficientVariables')
				end
			end
		end
		function set.coefficientValues(obj,value)
			if length(value) == obj.numCoeffVariables
				obj.coefficientValues = value;
			else
				error('Must be a range of values for each coefficientVariables,1 cell per coefficientVariable')
			end
		end
	end
end
function mustBefunctionHandle(x)
	if isa(x,'function_handle')
		return
	else
		error('functionHandle must be a function handle')
	end
end
function mustBeString(x,property)
	if isa(x,'string')
		if all(isrow(x))
			return
		else
			error([property,' must be a row vector NOT a column vector'])
		end
	else
		error(['All values of ',property,' must be a string'])
	end
end
function mustBeCell(x)
	if isa(x,'cell')
		if all(cellfun(@isnumeric, x))
			if all(cellfun(@isscalar, x))
				error('Each cell of coefficientValues cell array must contain more than one value, set value in function if constant')
			else
				if all(cellfun(@isrow,x))
					return
				else
					error(['Each cell of coefficientValues cell array must be a row vector'])
				end
			end
		else
			error('all values within coefficientValues cell array must be numeric')
		end
	else
		error('coefficientValues must be cell containing valid coefficient values for each coefficient')
	end
end