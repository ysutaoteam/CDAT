function [I]= mapping(c)
load clrmap;
 [S,F,T,P]=spectrogram(c,1024,512,1024,4000);
   spec=10*log10(abs(P)); 


min_spec=min(min(spec));
max_spec=max(max(spec));
step_spec=(max_spec-min_spec)/63;

spec64=zeros(size(spec)); 
I=zeros(size(spec,1),size(spec,2),3);

for i=1:size(spec,1)
    for j=1:size(spec,2)
       spec64(i,j)=floor((spec(i,j)-min_spec)/step_spec)+1;
       I(i,j,:)=clrmap(spec64(i,j),:);
    end
end
end
