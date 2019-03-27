        function displayLibrary(obj)
        	% displayLibrary - Displays the library of learned function coefficients in a UI table. Each column
        	% is the the derivative in that dimension. Each row is 1 function. The coefficients is how 
        	% much that function contributes that the derivative in that dimension. 
        	% Example: obj.displayLibrary;
        	derivatives = obj.learnedFunctions.Properties.VariableNames;
        	Functions = obj.learnedFunctions.Properties.RowNames;
        	for iDim = 1:length(derivatives)
				derivatives{iDim} = ['<HTML><sup>d</sup>&frasl;<sub>dt</sub> d<sub>',num2str(iDim),'</sub></HTML>'];
			end
			for iFunc = 1:length(Functions)
				if Functions{iFunc}(1) == 'd'
					Functions{iFunc} = insertAfter(Functions{iFunc},'d','<sub>');
					Functions{iFunc} = insertBefore(Functions{iFunc},'^','</sub>');
					Functions{iFunc} = insertAfter(Functions{iFunc},'(','<sup>');
					Functions{iFunc} = insertBefore(Functions{iFunc},')','</sup>');
					Functions{iFunc} = erase(Functions{iFunc},'^');
					Functions{iFunc} = erase(Functions{iFunc},'(');
					Functions{iFunc} = erase(Functions{iFunc},')');
					Functions{iFunc} = erase(Functions{iFunc},'*');
					Functions{iFunc} = ['<HTML>',Functions{iFunc},'</HTML>'];
				end
				if Functions{iFunc}(1) == 'e'
					Functions{iFunc} = insertAfter(Functions{iFunc},'d','<sub>');
					Functions{iFunc} = insertBefore(Functions{iFunc},')','</sub>');
					Functions{iFunc} = insertAfter(Functions{iFunc},'(','<sup>');
					Functions{iFunc} = insertAfter(Functions{iFunc},')','</sup>');
					Functions{iFunc} = erase(Functions{iFunc},'^');
					Functions{iFunc} = erase(Functions{iFunc},'(');
					Functions{iFunc} = erase(Functions{iFunc},')');
					Functions{iFunc} = erase(Functions{iFunc},'*');
					Functions{iFunc} = ['<HTML>',Functions{iFunc},'</HTML>'];
				end
				if contains(Functions{iFunc},'cos') || contains(Functions{iFunc},'sin')
					Functions{iFunc} = insertAfter(Functions{iFunc},'d','<sub>');
					Functions{iFunc} = insertBefore(Functions{iFunc},')','</sub>');
					Functions{iFunc} = erase(Functions{iFunc},'*');
					Functions{iFunc} = ['<HTML>',Functions{iFunc},'</HTML>'];
				end
            end
            f1 = figure;
            t = uitable('Data',obj.learnedFunctions{:,:},'ColumnName',derivatives,...
            'RowName',Functions,'Units', 'Normalized', 'Position',[0, 0, 1, 1],'FontSize',15);
            FontSize = 8;
%             hs = '<html><font size="+2">'; %html start
% 			he = '</font></html>'; %html end
% 			cnh = cellfun(@(x)[hs x he],derivatives,'uni',false); %with html
% 			rnh = cellfun(@(x)[hs x he],Functions,'uni',false); %with html
% 			set(t,'ColumnName',cnh,'RowName',rnh) %apply
            %get the row header
			jscroll=findjobj(t);
			rowHeaderViewport=jscroll.getComponent(4);
			rowHeader=rowHeaderViewport.getComponent(0);
			height=rowHeader.getSize;
			rowHeader.setSize(40,180)
			%resize the row header
			newWidth=125; %100 pixels.
			rowHeaderViewport.setPreferredSize(java.awt.Dimension(newWidth,0));
			height=rowHeader.getHeight;
			newHeight = 200;
			rowHeader.setPreferredSize(java.awt.Dimension(newWidth,height));
			rowHeader.setSize(newWidth,newHeight);
        end