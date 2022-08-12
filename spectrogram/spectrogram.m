clear all;clc;
Path = 'F:\healthy\';	% the path of voice 
File = dir(fullfile(Path,'*.wav'));	   
Filename = {File.name}';				% read file name
Len = length(File);						% read the number of files 
Output_path='F:\spectrogram_H\';
for i = 1:Len           
%     name = [Path,File(i).name];
    [y1,fs]= audioread([Path,'\',File(i).name]);   
    filename=File(i).name;
    c=y1(1:22050);
    c=y1;
    c=c.';
    I= mapping(c);
    newfilename=[Output_path,filename(1:end-4),'.jpg'];
    imwrite(I,newfilename)
end
