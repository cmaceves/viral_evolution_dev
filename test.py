import os 
import sys
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from more_itertools import distinct_permutations
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, GRU #can train using either GRU or LSTM
import random
import itertools

def parse_variants_samples(variants_dir, reference_seq):
    variants_files = [os.path.join(variants_dir, x) for x in os.listdir(variants_dir)]
    nuc_order = ["A", "C", "G", "T"]

    all_samples = []
    all_names = []
    for variant_file in variants_files:
        variants_name = variant_file.replace(variants_dir,"").strip("/").replace("_variants.tsv","")
        all_names.append(variants_name)
        populate = np.zeros((4, len(reference_seq)))
        df = pd.read_table(variant_file)
        for index, row in df.iterrows():
            pos = int(row['POS'])
            freq = float(row['ALT_FREQ'])
            nuc = row['ALT']
            if "+" in nuc or "-" in nuc:
                continue
            idx = nuc_order.index(nuc)
            populate[idx, pos] = freq 
        for i, col in enumerate(populate.T):
            total = sum(col)
            remainder = 1-total
            ref_nt = reference_seq[i]
            idx = nuc_order.index(ref_nt)
            populate[idx, i] = remainder
        all_samples.append(populate)
    return(all_samples, all_names)

def encoder_data():
    possible_permute = []
    scale = [round(x, 2) for x in np.linspace(0,1,100)]
    og_scale = copy.deepcopy(scale)
    scale.sort()
    if 0 in scale:
        scale.remove(0)

    possible = []
    for i in range(3000):
        possible.append([0.0, 0.0, 0.0, 1]) 
    
    for i, combo in enumerate(itertools.combinations(scale, 2)):
        if sum(combo) != 1:
            continue
        combo = list(combo)
        combo.extend([0.0, 0.0])
        for j in range(10):
            possible.append(combo)
    
    og_scale = [x for x in og_scale if x < 0.33]
    scale.extend(og_scale)
    down_possible = []
    
    for i, combo in enumerate(itertools.combinations(scale, 3)):
        if sum(combo) != 1:
            continue            
        combo = list(combo)
        y = [x for x in combo if x < 0.10]
        if len(y) == 0:
            continue
        combo.extend([0.0])
        down_possible.append(combo)
     
    og_scale = [x for x in og_scale if x < 0.25]
    scale.extend(og_scale)
    other = []
    for i, combo in enumerate(itertools.combinations(scale, 4)):
        if sum(combo) != 1:
            continue
        combo = list(combo)
        y = [x for x in combo if x < 0.05]
        z = [x for x in combo if 0.05 <= x < 0.10]
        if len(y) >= 2:
            down_possible.append(combo)
        if len(z) >= 2:
            other.append(combo)
    random.shuffle(down_possible)
    random.shuffle(other)
    print(len(down_possible))
    down_possible = random.sample(down_possible, 9000)
    other = random.sample(other, 1000)
    possible.extend(down_possible)
    possible.extend(other)
    dictionary_encoder = {}
    counter = 0
    all_data = []
    for tmp in possible:
        for permute in distinct_permutations(tmp, 4):
            possible_permute.append(list(permute))
            dictionary_encoder[counter] = set(permute)
            counter += 1
            all_data.append(permute)
    print(len(dictionary_encoder))
    random.shuffle(all_data)
    return(dictionary_encoder, all_data)

def read_variants_file(filename, reference_seq):
    start = 21563
    end = 25384
    df = pd.read_table(filename)
    nt_order = ['A', 'C', 'G', 'T']
    base_matrix = np.zeros((4, 3819))
    for index, row in df.iterrows():
        pos = row['POS'] - 1
        if start < (pos+1) < end:
            pass
        else:
            continue
        alt_freq = round( row['ALT_FREQ'],2)
        alt = row['ALT']
        ref = row['REF']
        if "+" in alt or '-' in alt:
            continue
        pos -= start
        ref_depth = row['REF_DP']
        alt_index = nt_order.index(alt)
        ref_index = nt_order.index(ref)
        

        base_matrix[alt_index,pos] = round(alt_freq,2) 
        total_depth = row['TOTAL_DP']
        ref_freq = round(ref_depth/total_depth,2)
        if ref_freq + alt_freq != 1:
            ref_freq += (1 - ref_freq - alt_freq)
        base_matrix[ref_index, pos] = round(ref_freq,2)

    for i,row in enumerate(base_matrix.T):
        all_zeros = not row.any()
        if all_zeros:
            nuc = reference_seq[i]
            j = nt_order.index(nuc)
            row[j] = 1
        base_matrix.T[i] = row
    return(base_matrix)

def read_sequence(filename):
    seq = ""
    with open(filename, 'r') as rfile:
        for line in rfile:
            line = line.strip()
            if line.startswith(">"):
                continue
            seq += line 
    return(seq)

def main():
    """
    In this first piece we build a simple autoencoder to embed nt prevalence arrays
    """
    """
    dictionary_encoder, all_data = encoder_data()
    random.shuffle(all_data)
    samples = np.array(all_data)
    samples = samples
    print(samples.shape)
    print(samples)

    activate = "tanh"
    #autoencoder to embed prevalence of nt (1-4)
    input_img = tf.keras.Input(shape=(4, ))
    layer1= tf.keras.layers.Dense(4, activation=activate)(input_img)
    layer0= tf.keras.layers.Dense(4, activation=activate)(layer1)
    layer5= tf.keras.layers.Dense(4, activation=activate)(layer0)
    encoded = tf.keras.layers.Dense(1, activation=activate)(layer5)
    layer2 = tf.keras.layers.Dense(4, activation=activate)(encoded)
    layer4 = tf.keras.layers.Dense(4, activation=activate)(layer2)
    layer3 = tf.keras.layers.Dense(4, activation=activate)(layer4)
    decoded = tf.keras.layers.Dense(4, activation='softmax')(layer3)
    autoencoder = tf. keras.Model(input_img, decoded)   

    mae  = tf.keras.losses.MeanAbsoluteError()
    metric = tf.keras.metrics.CosineSimilarity()
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    autoencoder.compile(loss=mae, optimizer=opt, metrics=metric)
    print(autoencoder.summary())
    print(samples.shape)
    print(samples)
    samples =np.asarray(samples).astype('float')
    autoencoder.fit(samples, samples, validation_split=0.15, epochs=15, shuffle=True, batch_size=124)

    test = np.array([0.13, 0.0, 0.0, 0.87])
    test = np.expand_dims(test, axis=0)
    pred = autoencoder.predict(test)
    print(pred)
    
    test = np.array([1.0, 0.0, 0.0, 0.0])
    test = np.expand_dims(test, axis=0)
    pred = autoencoder.predict(test)
    print(pred)
    
    test = np.array([0.35, 0.01, 0.0, 0.64])
    test = np.expand_dims(test, axis=0)
    pred = autoencoder.predict(test)
    print(pred)
    autoencoder.save('autoencoder.keras')
    sys.exit(0)
    """ 

    autoencoder = tf.keras.models.load_model('autoencoder.keras')
    encoder = tf.keras.Model(autoencoder.input, autoencoder.layers[4].output)
    decoder = tf.keras.Model(autoencoder.layers[5].input, autoencoder.output) 

    reference_seq = read_sequence("../sequence.fasta")

    #parse out the relative amount of stuff in each bam file
    variants_dir = "./variants"
    all_samples, sample_names = parse_variants_samples(variants_dir, reference_seq)

    #parse out the data related metadata of each sample
    metadata = "./ww_test_data/subset_ww.tsv"
    df = pd.read_table(metadata, usecols=['Unnamed: 0', 'collection_date'])
    df.rename(columns={'Unnamed: 0':'sample_name'}, inplace=True)

    #autoencode the samples 1 at a time to collapse dimensionality
    for i, filename in enumerate(variants_file):
        sample_matrix = read_variants_file(filename, reference_seq)
        ec = encoder.predict(sample_matrix.T)
        samples.append(np.squeeze(ec))
        raw.append(sample_matrix)
    

    samples = np.array(samples)
    imput_samples(samples)
    sys.exit(0)
    raw = np.array(raw)
    print(raw.shape)
    print(raw[:, :, 2040:2041])
    print(sample_matrix[:,2040:2041])
    print(samples[:, 2040:2041])
    print(samples.shape)


if __name__ == "__main__":
    main()
