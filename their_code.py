# Converted from their_code.ipynb

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Audio, display
from pynwb import NWBHDF5IO

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, IntSlider, FloatSlider, Dropdown, Checkbox

from brain_audio_decoder import BrainAudioDecoder
from custom_decoder import CustomBrainAudioDecoder
from brain_audio_decoder_viz import BrainAudioDecoderViz

# Define paths
path_bids = './SingleWordProductionDutch-iBIDS'
path_output = './features'
path_results = './results'

# Create directories if they don't exist
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_results, exist_ok=True)

# extract features: 
if __name__=="__main__":
    winL = 0.05
    frameshift = 0.01
    modelOrder = 4
    stepSize = 5
    path_bids = r'./SingleWordProductionDutch-iBIDS'
    path_output = r'./features'
    participants = pd.read_csv(os.path.join(path_bids,'participants.tsv'), delimiter='\t')
    for p_id, participant in enumerate(participants['participant_id']):
        
        #Load data
        io = NWBHDF5IO(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_ieeg.nwb'), 'r')
        nwbfile = io.read()
        #sEEG
        eeg = nwbfile.acquisition['iEEG'].data[:]
        eeg_sr = 1024
        #audio
        audio = nwbfile.acquisition['Audio'].data[:]
        audio_sr = 48000
        #words (markers)
        words = nwbfile.acquisition['Stimulus'].data[:]
        words = np.array(words, dtype=str)
        io.close()
        #channels
        channels = pd.read_csv(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_channels.tsv'), delimiter='\t')
        channels = np.array(channels['name'])

        #Extract HG features
        feat = extractHG(eeg,eeg_sr, windowLength=winL,frameshift=frameshift)

        #Stack features
        feat = stackFeatures(feat,modelOrder=modelOrder,stepSize=stepSize)
        
        #Process Audio
        target_SR = 16000
        audio = scipy.signal.decimate(audio,int(audio_sr / target_SR))
        audio_sr = target_SR
        scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
        os.makedirs(os.path.join(path_output), exist_ok=True)
        scipy.io.wavfile.write(os.path.join(path_output,f'{participant}_orig_audio.wav'),audio_sr,scaled)   

        #Extract spectrogram
        melSpec = extractMelSpecs(scaled,audio_sr,windowLength=winL,frameshift=frameshift)
        
        #Align to EEG features
        words = downsampleLabels(words,eeg_sr,windowLength=winL,frameshift=frameshift)
        words = words[modelOrder*stepSize:words.shape[0]-modelOrder*stepSize]
        melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize,:]
        #adjust length (differences might occur due to rounding in the number of windows)
        if melSpec.shape[0]!=feat.shape[0]:
            tLen = np.min([melSpec.shape[0],feat.shape[0]])
            melSpec = melSpec[:tLen,:]
            feat = feat[:tLen,:]
        
        #Create feature names by appending the temporal shift 
        feature_names = nameVector(channels[:,None], modelOrder=modelOrder)

        #Save everything
        np.save(os.path.join(path_output,f'{participant}_feat.npy'), feat)
        np.save(os.path.join(path_output,f'{participant}_procWords.npy'), words)
        np.save(os.path.join(path_output,f'{participant}_spec.npy'), melSpec)
        np.save(os.path.join(path_output,f'{participant}_feat_names.npy'), feature_names)

# reconstruction minimal
if __name__=="__main__":
    feat_path = r'./features'
    result_path = r'./results'
    pts = ['sub-%02d'%i for i in range(1,11)]

    winLength = 0.05
    frameshift = 0.01
    audiosr = 16000

    nfolds = 10
    kf = KFold(nfolds,shuffle=False)
    est = LinearRegression(n_jobs=5)
    pca = PCA()
    numComps = 50
    
    #Initialize empty matrices for correlation results, randomized contols and amount of explained variance
    allRes = np.zeros((len(pts),nfolds,23))
    explainedVariance = np.zeros((len(pts),nfolds))
    numRands = 1000
    randomControl = np.zeros((len(pts),numRands, 23))

    for pNr, pt in enumerate(pts):
        #Load the data
        spectrogram = np.load(os.path.join(feat_path,f'{pt}_spec.npy'))
        data = np.load(os.path.join(feat_path,f'{pt}_feat.npy'))
        labels = np.load(os.path.join(feat_path,f'{pt}_procWords.npy'))
        featName = np.load(os.path.join(feat_path,f'{pt}_feat_names.npy'))
        
        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Save the correlation coefficients for each fold
        rs = np.zeros((nfolds,spectrogram.shape[1]))
        for k,(train, test) in enumerate(kf.split(data)):
            #Z-Normalize with mean and std from the training data
            mu=np.mean(data[train,:],axis=0)
            std=np.std(data[train,:],axis=0)
            trainData=(data[train,:]-mu)/std
            testData=(data[test,:]-mu)/std

            #Fit PCA to training data
            pca.fit(trainData)
            #Get percentage of explained variance by selected components
            explainedVariance[pNr,k] =  np.sum(pca.explained_variance_ratio_[:numComps])
            #Tranform data into component space
            trainData=np.dot(trainData, pca.components_[:numComps,:].T)
            testData = np.dot(testData, pca.components_[:numComps,:].T)
            
            #Fit the regression model
            est.fit(trainData, spectrogram[train, :])
            #Predict the reconstructed spectrogram for the test data
            rec_spec[test, :] = est.predict(testData)

            #Evaluate reconstruction of this fold
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                rs[k,specBin] = r

        #Show evaluation result
        print('%s has mean correlation of %f' % (pt, np.mean(rs)))
        allRes[pNr,:,:]=rs

        #Estimate random baseline
        for randRound in range(numRands):
            #Choose a random splitting point at least 10% of the dataset size away
            splitPoint = np.random.choice(np.arange(int(spectrogram.shape[0]*0.1),int(spectrogram.shape[0]*0.9)))
            #Swap the dataset on the splitting point 
            shuffled = np.concatenate((spectrogram[splitPoint:,:],spectrogram[:splitPoint,:]))
            #Calculate the correlations
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[:,specBin], shuffled[:,specBin])
                randomControl[pNr, randRound,specBin]=r


#         #Save reconstructed spectrogram
#         os.makedirs(os.path.join(result_path), exist_ok=True)
#         np.save(os.path.join(result_path,f'{pt}_predicted_spec.npy'), rec_spec)
        
        #Synthesize waveform from spectrogram using Griffin-Lim
#         reconstructedWav = createAudio(rec_spec,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
#         wavfile.write(os.path.join(result_path,f'{pt}_predicted.wav'),int(audiosr),reconstructedWav)

#         #For comparison synthesize the original spectrogram with Griffin-Lim
#         origWav = createAudio(spectrogram,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
#         wavfile.write(os.path.join(result_path,f'{pt}_orig_synthesized.wav'),int(audiosr),origWav)

#     #Save results in numpy arrays          
#     np.save(os.path.join(result_path,'linearResults.npy'),allRes)
#     np.save(os.path.join(result_path,'randomResults.npy'),randomControl)
#     np.save(os.path.join(result_path,'explainedVariance.npy'),explainedVariance)

# Initialize decoder
decoder = CustomBrainAudioDecoder(
    path_bids=path_bids,
    path_output=path_output,
    path_results=path_results,
    win_length=0.05,
    frameshift=0.01,
    model_order=4,
    step_size=5,
    n_components=50
)

visualizer = BrainAudioDecoderViz(decoder)

participants = decoder.load_participants()
print(f"Found {len(participants)} participants")

# Alternative: Explicitly define all participants
all_participants = [f'sub-{i:02d}' for i in range(1, 11)]
print(f"All participants: {all_participants}")

# Choose a participant to analyze
participant_id = 'sub-01'

# Check if features are already extracted
if not os.path.exists(os.path.join(path_output, f'{participant_id}_feat.npy')):
    print(f"Extracting features for {participant_id}...")
    decoder.extract_features_for_participant(participant_id)
else:
    print(f"Features already extracted for {participant_id}")

# Load participants
participants = decoder.load_participants()
print(f"Found {len(participants)} participants")
participants.head()

participant_id = 'sub-02'

# Check if features already exist
if not os.path.exists(os.path.join(path_output, f'{participant_id}_feat.npy')):
    print(f"Extracting features for {participant_id}...")
    features, spectrogram, words, feature_names = decoder.extract_features_for_participant(participant_id)
else:
    print(f"Loading existing features for {participant_id}...")
    features, spectrogram, words, feature_names = decoder.load_features(participant_id)

print(f"Feature shape: {features.shape}")
print(f"Spectrogram shape: {spectrogram.shape}")
print(f"Words shape: {words.shape}")
print(f"Number of unique words: {len(np.unique(words))}")

@interact(
    participant_id=Dropdown(
        options=[f'sub-{i:02d}' for i in range(1, 11)],
        value='sub-01',
        description='Participant:'
    ),
    duration=FloatSlider(min=1, max=10, step=0.5, value=3, description='Duration (s):'),
    start_time=FloatSlider(min=0, max=20, step=0.5, value=0, description='Start time (s):'),
    channels=IntSlider(min=1, max=20, step=1, value=8, description='Channels:')
)
def interactive_eeg(participant_id, duration, start_time, channels):
    visualizer.plot_raw_eeg(
        participant_id=participant_id,
        duration=duration,
        start_time=start_time,
        channels_to_plot=channels
    )

@interact(
    participant_id=Dropdown(
        options=[f'sub-{i:02d}' for i in range(1, 11)],
        value='sub-01',
        description='Participant:'
    ),
    original=Checkbox(value=True, description='Original'),
    predicted=Checkbox(value=True, description='Predicted'),
    side_by_side=Checkbox(value=True, description='Side by side'),
    log_scale=Checkbox(value=True, description='Log scale')
)
def interactive_spectrogram(participant_id, original, predicted, side_by_side, log_scale):
    # Check if prediction exists
    if predicted and not os.path.exists(os.path.join(path_results, f'{participant_id}_predicted_spec.npy')):
        print(f"Warning: Predicted spectrogram not found for {participant_id}")
        print("Training model first...")
        decoder.train_test_model(participant_id, save_audio=False)
    
    visualizer.plot_spectrogram(
        participant_id=participant_id,
        original=original,
        predicted=predicted,
        side_by_side=side_by_side,
        log_scale=log_scale
    )

# @interact(
#     participant_id=Dropdown(
#         options=[f'sub-{i:02d}' for i in range(1, 11)],
#         value='sub-01',
#         description='Participant:'
#     ),
#     original=Checkbox(value=True, description='Original'),
#     predicted=Checkbox(value=True, description='Predicted'),
#     reconstructed=Checkbox(value=True, description='Reconstructed')
# )
# def interactive_audio(participant_id, original, predicted, reconstructed):
#     # Check if audio files exist
#     if predicted and not os.path.exists(os.path.join(path_results, f'{participant_id}_predicted.wav')):
#         print(f"Warning: Predicted audio not found for {participant_id}")
#         print("Training model first...")
#         decoder.train_test_model(participant_id, save_audio=True)
    
#     visualizer.play_audio(
#         participant_id=participant_id,
#         original=original,
#         predicted=predicted,
#         reconstructed=reconstructed
#     )

@interact(
    participant_id=Dropdown(
        options=[f'sub-{i:02d}' for i in range(1, 11)],
        value='sub-01',
        description='Participant:'
    ),
    original=Checkbox(value=True, description='Original'),
    predicted=Checkbox(value=True, description='Predicted'),
    reconstructed=Checkbox(value=True, description='Reconstructed'),
    side_by_side=Checkbox(value=True, description='Side by side')
)
def interactive_waveform(participant_id, original, predicted, reconstructed, side_by_side):
    # Check if audio files exist
    if predicted and not os.path.exists(os.path.join(path_results, f'{participant_id}_predicted.wav')):
        print(f"Warning: Predicted audio not found for {participant_id}")
        print("Training model first...")
        decoder.train_test_model(participant_id, save_audio=True)
    
    visualizer.plot_waveform(
        participant_id=participant_id,
        original=original,
        predicted=predicted,
        reconstructed=reconstructed,
        side_by_side=side_by_side
    )

@interact(
    participant_id=Dropdown(
        options=[f'sub-{i:02d}' for i in range(1, 11)],
        value='sub-01',
        description='Participant:'
    )
)
def plot_participant_channels(participant_id):
    visualizer.plot_channels_comparison(participant_id, duration_seconds=3)

@interact(
    participant_id=Dropdown(
        options=[f'sub-{i:02d}' for i in range(1, 11)],
        value='sub-01',
        description='Participant:'
    )
)
def analyze_participant_channels(participant_id):
    decoder.analyze_channels(participant_id)

# channel_results = decoder.analyze_individual_channels('sub-01')

# # 3. Analyze channels across all participants (this may take a while)
# all_results = decoder.analyze_channels_across_participants()

# 5. Interactive visualization (in Jupyter notebook)
interactive_widget = visualizer.interactive_channel_analysis()
display(interactive_widget)

# Interactive exploration (in Jupyter notebook)
interactive_widget = visualizer.plot_channel_matrix()
display(interactive_widget)

# 2. To display the interactive participant selector widget:
from IPython.display import display
selector_widget = visualizer.simple_participant_selector()
display(selector_widget)

# Or use the interactive widget
word_matrix_widget = visualizer.interactive_word_matrix()
display(word_matrix_widget)
