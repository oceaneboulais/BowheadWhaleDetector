
function YdB=synthesize_FM_image(fstart,fend,duration)
SNRdB=10;

%function [y,ysignal,ynoise]=simulate_FMsweep_with_noise(Fs,Nfft,ovlap,fstart,fend,total_window_time,tduration,tstart,SNRdB,SNRchc, no_noise);
% Create an FM sweep in white noise
% Input:
%   Fs: sampling rate in Hz
%   Nfft: FFT size used for computing spectrogram (not really needed).
%   fstart: start frequency of sweep in Hz
%   fend: end freqency of sweep in Hz
%   total_wind_time: total length of time series in seconds
%   tduration: duration of sweep in seconds
%   tstart: time at which sweep begins in output time series
%   SNRdB: desired SEL SNR in dB
%   SNRchc: 'SEL' or 'PSD'- how should SNR be defined?
%   no_noise:  if exists, don't add any noise to signal regardless of value
%       of SNRdB.
% Output:
%   y: export time series, sampled at Fs Hz
y=simulate_FMsweep_with_noise(1000,256,0.9,fstart,fend,3,duration,1,SNRdB,'PSD');
[~,F,T,Y]=spectrogram(y,256,round(0.9*256),256,1000);
YdB=10*log10(abs(Y));
YdB=YdB-max(max(YdB))+SNRdB;
YdB(YdB<0)=0;
YdB=YdB(1:121,1:104);
YdB=single(YdB)./single(max(max(YdB)));

%imagesc(T,F,YdB);axis xy