function [spec]= nengliang(c)
[S,F,T,P]=spectrogram(c,1024,512,1024,4000);
 spec=10*log10(abs(P)); 
 
 
 