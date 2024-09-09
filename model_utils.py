from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Concatenate, Multiply

def create_sae_model(input_shape):
    """创建自编码器模型"""
    input_img = Input(shape=input_shape)
    x = Flatten()(input_img)

    encoded = Dense(256, activation='relu', activity_regularizer=regularizers.l2(0.1))(x)
    encoded = Dense(128, activation='relu', activity_regularizer=regularizers.l2(0.1))(encoded)
    encoded = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.1))(encoded)

    decoded = Dense(128, activation='relu', activity_regularizer=regularizers.l2(0.1))(encoded)
    decoded = Dense(256, activation='relu', activity_regularizer=regularizers.l2(0.1))(decoded)
    decoded = Dense(input_shape[0] * input_shape[1], activation='sigmoid', activity_regularizer=regularizers.l2(0.1))(decoded)
    decoded = Reshape((input_shape[0], input_shape[1], 1))(decoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    return autoencoder, encoder

def attention_3d_block(inputs):
    """定义3D注意力模块"""
    a = Dense(inputs.shape[-1], activation='softmax')(inputs)
    output_attention_mul = Multiply()([inputs, a])
    return output_attention_mul

def create_attention_model(input_shape_1, input_shape_2, behavior_shape):
    """创建包含注意力机制的融合模型"""
    input_1 = Input(shape=input_shape_1)
    input_2 = Input(shape=input_shape_2)
    behavior_input_1 = Input(shape=behavior_shape)
    behavior_input_2 = Input(shape=behavior_shape)

    attention_output_1 = attention_3d_block(input_1)
    attention_output_2 = attention_3d_block(input_2)

    concatenated_1 = Concatenate()([attention_output_1, behavior_input_1])
    concatenated_2 = Concatenate()([attention_output_2, behavior_input_2])

    attention_output_final_1 = attention_3d_block(concatenated_1)
    attention_output_final_2 = attention_3d_block(concatenated_2)

    final_output = Concatenate()([attention_output_final_1, attention_output_final_2])

    model = Model(inputs=[input_1, input_2, behavior_input_1, behavior_input_2], outputs=final_output)
    return model
