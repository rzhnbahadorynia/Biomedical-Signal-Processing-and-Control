Python 3.12.1 (v3.12.1:2305ca5144, Dec  7 2023, 17:23:39) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Add, GlobalMaxPooling2D, TimeDistributed
from tensorflow.keras.optimizers import Adam

def build_two_stream_cnn_lstm(num_classes=7, num_frames=10, input_shape=(224, 224, 3)):
    """
    Constructs the proposed Two-Stream CNN-LSTM architecture.
    Uses GoogLeNet (InceptionV3 as baseline) for visual feature extraction 
    and parallel LSTMs for processing short-range and long-range temporal features.
    """
    # Model Input: A clip containing 10 short frames with dimensions of 224x224x3
    clip_input = Input(shape=(num_frames, *input_shape), name="Clip_Input")
    
    # 1. Visual Feature Extractor (GoogLeNet Backbone)
    # Using pre-trained InceptionV3 weights from ImageNet, excluding top dense layers
    base_cnn = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Flattening spatial features using Global Max Pooling 2D as described in the paper
    x = GlobalMaxPooling2D()(base_cnn.output)
    cnn_feature_extractor = Model(inputs=base_cnn.input, outputs=x, name="GoogLeNet_Extractor")
    
    # Apply the CNN backbone across all frames in the clip sequence using TimeDistributed
    visual_features_sequence = TimeDistributed(cnn_feature_extractor)(clip_input) # Output shape: (Batch, 10, Feature_Dim)
    
    # 2. Parallel Two-Stream LSTMs
    
    # Stream 1: Sequence-to-Sequence LSTM (Extracts short-term adjacent frame updates)
    # This layer returns the full sequence (return_sequences=True)
    lstm_seq2seq = LSTM(units=256, return_sequences=True, dropout=0.25, activation='sigmoid', name="LSTM_Seq2Seq")(visual_features_sequence)
    # Consolidate sequence into a single feature vector for downstream fusion
    lstm_seq2seq_out = LSTM(units=256, return_sequences=False, dropout=0.25, activation='sigmoid')(lstm_seq2seq)
    dense_stream1 = Dense(num_classes, activation=None, name="Dense_Stream_1")(lstm_seq2seq_out)
    
...     # Stream 2: Sequence-to-Vector LSTM (Extracts global, long-term clip context)
...     # This layer returns only the final hidden state vector (return_sequences=False)
...     lstm_seq2vec = LSTM(units=256, return_sequences=False, dropout=0.25, activation='sigmoid', name="LSTM_Seq2Vec")(visual_features_sequence)
...     dense_stream2 = Dense(num_classes, activation=None, name="Dense_Stream_2")(lstm_seq2vec)
...     
...     # 3. Stream Fusion Method
...     # Merging the two independent architectural streams via additive element-wise operations (Add)
...     merged_streams = Add(name="Stream_Merger")([dense_stream1, dense_stream2])
...     
...     # Final Output Layer: Dense Softmax layer for categorical 7-phase surgical classification
...     final_output = Dense(num_classes, activation='softmax', name="Surgical_Phase_Output")(merged_streams)
...     
...     # Initialize complete graph
...     model = Model(inputs=clip_input, outputs=final_output, name="Two_Stream_CNN_LSTM")
...     return model
... 
... # Instantiate and print topology summary of the network
... proposed_model = build_two_stream_cnn_lstm()
... proposed_model.summary()
... 
... # 4. Model Compilation
... # Setting hyperparameters according to section 3.3 (Adam Optimizer, learning rate=1e-5, momentum/beta1=0.9)
... opt = Adam(learning_rate=0.00001, beta_1=0.9)
... proposed_model.compile(
...     optimizer=opt,
...     loss='categorical_crossentropy',
...     metrics=['accuracy']
... )
... 
