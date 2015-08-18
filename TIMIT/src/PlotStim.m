function [outSpectrum, outfreqs] = PlotStim(stimulus,fs,plotflag)

    if nargin<2
        plotflag=0;
    end
    % Define some constants
   % fs          = 44100; % sampling frequency
    initialFreq = 250;  % define frequency range that i care about
    endFreq     = 8000;

    DBNOISE     = 65; % Affects the noise floor (i.e. what is blue)
    DBMAX       = 65;

    % %% Resample the stimulus if not at 48kHz
     if fs ~= 48000,
         stimulus = resample(stimulus,48000,fs);
         fs = 48000;
     end

    % Compute the spectrogram
    [outSpectrum outfreqs] = MakeSpectrogram(stimulus,fs,DBNOISE,DBMAX);
 
    if plotflag
        % Plot the stimulus envelope and the spectrogram
        fig           = figure;
        ax(1)         = subplot(3,1,1);
        envelope_time = (1/fs:1/fs:length(stimulus)/fs)*1000;
        plot(envelope_time,stimulus,'k-')
        axis tight
        set(gca,'XTick',[])

        ax(2) = subplot(3,1,2:3);
        imagesc(outSpectrum);axis xy
        set(gca,'YTick',[1 size(outSpectrum,1)])
        set(gca,'YTickLabel',[initialFreq endFreq])
        xlabel('Time (msec)')
        ylabel('Frequency')
        linkaxes(ax,'x')
    end
end

function [outSpectrum outfreqs] = MakeSpectrogram(stimulus,fs,DBNOISE,DBMAX,binsize,frameCount,increment,nFTfreqs,initialFreq,endFreq)

    % Define some inputs, if not passed by user
    if nargin == 0,
        [file path] = uigetfile('.wav');
        filename = file(1:end-4);
        curDir = cd;
        cd(path)
        [stimulus,fs,nbits] = wavread(filename);
        DBNOISE = 65; %Affects the noise floor (i.e. what is blue)
        DBMAX = 65;
    elseif nargin == 2,
        DBNOISE = 65; %Affects the noise floor (i.e. what is blue)
        DBMAX = 65;
    elseif nargin == 3,
        DBMAX = 65;
    end

    % % Resample the stimulus to a multiple of 1000, if necessary
    % if fs~=48000,
    %     stimulus = resample(stimulus,48000,fs);
    %     fs = 48000;
    % end

    % Do some other processing
    ampsamprate = 1000; % This is input by the user in strfpak
    fwidth      = 125;  % This is input by the user in strfpak
    initialFreq = 250;  % define frequency range that i care about
    endFreq     = 8000;
    binsize     = floor(1/(2*pi*fwidth)*6*fs);           % size of time bin over which fft is computed (in samples)
    nFTfreqs    = binsize;                               % # of frequencies at which fft is computed
    increment   = floor(fs/ampsamprate);                 % # samples to shift for each new bin start point

    % Zero-pad the stimulus appropriately
    stimulus = stimulus';
    stimulus = [zeros(1,binsize/2) stimulus zeros(1, binsize/2)];
    
    stimsize    = length(stimulus);
    frameCount  = floor((stimsize-binsize)/increment)+1; % # of bins

    % Compute the gaussian filter
    wx2  = ((1:binsize)-binsize/2).^2;
    wvar = (binsize/6)^2;
    ws   = exp(-0.5*(wx2./wvar));

    % Compute spectrogram of entire stimulus (Use a sliding fourier transform)
    s = zeros(binsize/2+1, frameCount);
    for i=1:frameCount
        start = (i-1)*increment + 1;
        last  = start + binsize - 1;
        f     = zeros(binsize, 1);
        f(1:binsize) = ws.*stimulus(start:last);
        binspec = fft(f);
        s(:,i)  = binspec(1:(nFTfreqs/2+1));
    end

    % Translate to dB, rectify
    tmp = max(0, 20*log10(abs(s)./max(max(abs(s))))+DBNOISE);
    tmp = min(DBMAX,tmp);

    % Edit out the appropriate range of frequencies
    select      = 1 : nFTfreqs/2+1;
    fo          = (select-1)'*fs/nFTfreqs;
    freq_range  = find(fo>=initialFreq & fo<=endFreq);
    outSpectrum = tmp(freq_range,:);
    outfreqs    = fo(freq_range);
end
