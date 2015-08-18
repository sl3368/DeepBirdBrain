%% Bird Classification
clear all; close all; clc

%% Import songs in batches and create spectograms
% % In batches of 1000

start=1;
end_batch=1;

for i_batch = start:end_batch
	filepath='/vega/stats/users/sl3368/TIMIT_process/TimitWav/test/';
	filenames=dir('/vega/stats/users/sl3368/TIMIT_process/TimitWav/test/');
	%load /vega/stats/users/sl3368/LifeClef_processing/trainInds.mat
	i_n = 1;
	batchSize = 1344; 
	%i_batch
	batchInds = (i_batch-1)*batchSize+1:i_batch*batchSize;
	for i = batchInds
		%batchInds(1)
		%size(trainInds)
		%i
		i_song = filenames(i+2).name;
		song_fp=strcat(filepath,i_song);
		if exist(song_fp,'file')
		     [Y{i},Fs{i}] = audioread(song_fp);
		     [outSpectrumt{i_n}, freqs] = PlotStim(Y{i},Fs{i},0);
		     outSpectrum{i_n} = single(outSpectrumt{i_n}');
		     batchSizes(i) = size(outSpectrum{i_n},1);
		     songSums(i,:) = sum(outSpectrum{i_n},1);
		else
		     batchSizes(i) = NaN;
		     songSums(i,1:60) = NaN;
		end
		i_n = i_n+1;
		%i
	end
 
	save(sprintf('/vega/stats/users/sl3368/Data_LC/timit_processing/batchSizes%d.mat',i_batch),'batchSizes')
	save(sprintf('/vega/stats/users/sl3368/Data_LC/timit_processing/songSums%d.mat',i_batch),'songSums')
	save(sprintf('/vega/stats/users/sl3368/Data_LC/timit_processing/outSpectrum%d.mat',i_batch),'outSpectrum','-v7.3')

	clear all; 
end

%% Find max batch size

start=1;
end_batch=1;
batchSize = 1344;
for i_batch = start:end_batch
       load(sprintf('/vega/stats/users/sl3368/Data_LC/timit_processing/batchSizes%d.mat',i_batch))
       if i_batch < 1410 
           batchInds = (i_batch-1)*batchSize+1:i_batch*batchSize;
           batchSize_all(batchInds) = batchSizes(batchInds);
       else
           batchInds = 24001:24607;%(i_batch-1)*batchSize+1:i_batch*batchSize;
           batchSize_all(batchInds) = batchSizes(batchInds);
       end       
end
disp('Mean: ')
mean(batchSize_all)
disp('Std. Dev: ')
std(batchSize_all)
disp('Max: ')
max(batchSize_all)
disp('Min: ')
min(batchSize_all)
hist(batchSize_all,100);
%maxBatchSize = 500000;

%% Find mean and standard deviation
% Mean 
start=1;
end_batch=1;
batchSize = 1344;
for i_batch = start:end_batch
     load(sprintf('/vega/stats/users/sl3368/Data_LC/timit_processing/songSums%d.mat',i_batch))
     if i_batch < 1410 
         batchInds = (i_batch-1)*batchSize+1:i_batch*batchSize;
         songSums_all(batchInds,:) = songSums(batchInds,:);
     else
         batchInds = 24001:24607;%(i_batch-1)*batchSize+1:i_batch*batchSize;
        songSums_all(batchInds,:) = songSums(batchInds,:);
     end
     
end

realSum = sum(songSums_all);
totalLength = sum(batchSize_all);
allsongMean = realSum/totalLength;
save /vega/stats/users/sl3368/Data_LC/timit_processing/allsongMean allsongMean

% Standard deviation
sqrt((1/(size(songSums,1)-1))*sum((bsxfun(@minus,songSums,mean(songSums))).^2))
stdStep=[];
load /vega/stats/users/sl3368/Data_LC/timit_processing/allsongMean.mat
for i_batch = start:end_batch
	load(sprintf('/vega/stats/users/sl3368/Data_LC/timit_processing/outSpectrum%d.mat',i_batch))
	if i_batch<141
        	for i_song = 1:batchSize  
                	stdStep = [stdStep;  sum(bsxfun(@minus,outSpectrum{i_song},allsongMean).^2)];          
         	end
     	else
        	for i_song = 1:607 
             		stdStep = [stdStep;  sum(bsxfun(@minus,outSpectrum{i_song},allsongMean).^2)];      
         	end    
     	end
	clearvars outSpectrum 
end
save /vega/stats/users/sl3368/Data_LC/timit_processing/stdStep stdStep
allsongSTD = sqrt((1/(sum(batchSize_all)-1))*sum(stdStep));
save /vega/stats/users/sl3368/Data_LC/timit_processing/allsongSTD allsongSTD

%% Pad songs

start=1;
end_batch=1;
padded_song_length=5000;
for i_batch=start:end_batch

	batchSize = 1344; %number of songs in each saved portion of the spectogram

	% Load data
	load(sprintf('/vega/stats/users/sl3368/Data_LC/timit_processing/outSpectrum%d.mat',i_batch))
	
	% OutSpectrum is a cell array of 1000 spectograms 

	% Annoying indexing 
	newInds = randperm(batchSize);
	batchInds = (i_batch-1)*batchSize+1:i_batch*batchSize;
	trainInds_thisbatch = batchInds(newInds);

	% Data parameters
	maxBatchSize = padded_song_length;

	num = 1;
	for i = 1:batchSize 
	    thisStim{i} = outSpectrum{newInds(i)};
	end

	% Pad each song
	useMask = [];
	stimulus = [];
	song_vec = find(~cellfun(@isempty,thisStim));
	for ind_song = 1:length(song_vec); % cycle through all songs
	    i_song = song_vec(ind_song);
	    thissong = thisStim{i_song};
	    if size(thissong,1) > maxBatchSize %cut off songs that are larger than the chosen batch size
		    paddedStim = [thissong(1:maxBatchSize,:); zeros(size(thissong,2),(maxBatchSize-size(thissong,1)))'];
		    stimulus = [stimulus; paddedStim]; %all time by freq
		    useMask = [useMask ones(maxBatchSize,1)'];
	    else
		    paddedStim = [thissong; zeros(size(thissong,2),(maxBatchSize-size(thissong,1)))'];
		    stimulus = [stimulus; paddedStim]; %all time by freq
		    useMask = [useMask ones(size(thissong,1),1)' zeros((maxBatchSize-size(thissong,1)),1)'];
	    end
	    %i_song
	end
	useMask = useMask';
	% useMask is a binary vector where 1s indicate the actual song and 0s
	% indicate zero padding

	%% Normalize
	load /vega/stats/users/sl3368/Data_LC/timit_processing/allsongSTD
	load /vega/stats/users/sl3368/Data_LC/timit_processing/allsongMean
	maskedstimulus = stimulus;
	maskedstimulus(useMask==0,:)=[]; % I only normalize the nonpadded parts because I want the padded sections to remain zero (much less room to store)
	stimulus_zscore = zeros(size(stimulus,1),size(stimulus,2));
	stimulus_zscore(useMask==1,:) = bsxfun(@minus,maskedstimulus,allsongMean); % subtract the mean
	stimulus_zscore(useMask==1,:) = bsxfun(@rdivide,stimulus_zscore(useMask==1,:),allsongSTD); % divide by the standard deviation
	stimulus_zscore_all = stimulus_zscore;
	whos
	useMask_all = useMask;
	stimulus_zscore = stimulus_zscore_all(1:batchSize*maxBatchSize,:);
	trainInds_small = trainInds_thisbatch(1:batchSize);
	useMask = useMask_all(1:batchSize*maxBatchSize);
	save(sprintf('/vega/stats/users/sl3368/Data_LC/timit/test/timit_stim_%d.mat',i_batch),'stimulus_zscore','useMask','-v7.3')
	save(sprintf('/vega/stats/users/sl3368/Data_LC/timit/test/timit_inds_%d.mat',i_batch),'trainInds_small','newInds','-v7.3')
	disp('Save Complete..')
end
%% Split into smaller parts and saving

%save_inds = (i_batch-1)*4+1:i_batch*4;
%
%% Part 1
%stimulus_zscore = stimulus_zscore_all(1:part_length*maxBatchSize,:);
%trainInds_small = trainInds_thisbatch(1:part_length);
%useMask = useMask_all(1:part_length*maxBatchSize);
%save(sprintf('/vega/stats/users/erb2180/supervised_class/LifeClef/LC_stim250_%d.mat',save_inds(1)),'stimulus_zscore','useMask','-v7.3')
%save(sprintf('/vega/stats/users/erb2180/supervised_class/LifeClef/LC_inds250_%d.mat',save_inds(1)),'trainInds_small','newInds','-v7.3')
%
%% Part 2
%stimulus_zscore = stimulus_zscore_all(part_length*maxBatchSize+1:2*part_length*maxBatchSize,:);
%size(stimulus_zscore)
%trainInds_small = trainInds_thisbatch(part_length+1:2*part_length);
%useMask = useMask_all(part_length*maxBatchSize+1:2*part_length*maxBatchSize);
%save(sprintf('/vega/stats/users/erb2180/supervised_class/LifeClef/LC_stim250_%d.mat',save_inds(2)),'stimulus_zscore','useMask','-v7.3')
%save(sprintf('/vega/stats/users/erb2180/supervised_class/LifeClef/LC_inds250_%d.mat',save_inds(2)),'trainInds_small','newInds','-v7.3')
%
%% Part 3
%stimulus_zscore = stimulus_zscore_all(2*part_length*maxBatchSize+1:3*part_length*maxBatchSize,:);
%size(stimulus_zscore)
%trainInds_small = trainInds_thisbatch(2*part_length+1:3*part_length);
%useMask = useMask_all(2*part_length*maxBatchSize+1:3*part_length*maxBatchSize);
%save(sprintf('/vega/stats/users/erb2180/supervised_class/LifeClef/LC_stim250_%d.mat',save_inds(3)),'stimulus_zscore','useMask','-v7.3')
%save(sprintf('/vega/stats/users/erb2180/supervised_class/LifeClef/LC_inds250_%d.mat',save_inds(3)),'trainInds_small','newInds','-v7.3')
%
%% Part 4
%stimulus_zscore = stimulus_zscore_all(3*part_length*maxBatchSize+1:4*part_length*maxBatchSize,:);
%size(stimulus_zscore)
%trainInds_small = trainInds_thisbatch(3*part_length+1:4*part_length);
%useMask = useMask_all(3*part_length*maxBatchSize+1:4*part_length*maxBatchSize);
%save(sprintf('/vega/stats/users/erb2180/supervised_class/LifeClef/LC_stim250_%d.mat',save_inds(4)),'stimulus_zscore','useMask','-v7.3')
%save(sprintf('/vega/stats/users/erb2180/supervised_class/LifeClef/LC_inds250_%d.mat',save_inds(4)),'trainInds_small','newInds','-v7.3')
%%end
