import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, Dropout, ReLU, Concatenate


def inception(input_x, conv1x1_filters, reduce3x3_filters, conv3x3_filters, reduce5x5_filters, conv5x5_filters,
              pool_proj_filters):
    """
    Description:
        inception 구조를 구현하기 위한 함수입니다.

    :param input_x: inception 의 입력으로 들어온 이전 레이어의 출력값입니다.
    :param conv1x1_filters: 1x1 conv 에 사용하는 filter 수 입니다.
    :param reduce3x3_filters: 3x3 conv 진행 전, 차원 축소에 사용하는 filter 수 입니다.
    :param conv3x3_filters: 3x3 conv 에 사용하는 filter 수 입니다.
    :param reduce5x5_filters: 5x5 conv 진행 전, 차원 축소에 사용하는 filter 수 입니다.
    :param conv5x5_filters: 5x5 conv 에 사용하는 filter 수 입니다.
    :param pool_proj_filters: projection 에 사용하는 filter 수 입니다.

    :return:
        inception_output: conv1x1, conv3x3, conv5x5, pool_proj 을 concatenate 하여 리턴합니다.
    """
    # 1x1 convolution
    conv1x1 = Conv2D(filters=conv1x1_filters, kernel_size=(1, 1), padding="SAME")(input_x)
    conv1x1 = ReLU()(conv1x1)

    # 1x1 convolution and 3x3 convolution
    conv3x3 = Conv2D(filters=reduce3x3_filters, kernel_size=(1, 1), padding="SAME")(input_x)
    conv3x3 = Conv2D(filters=conv3x3_filters, kernel_size=(3, 3), padding="SAME")(conv3x3)
    conv3x3 = ReLU()(conv3x3)

    # 1x1 convolution and 5x5 convolution
    conv5x5 = Conv2D(filters=reduce5x5_filters, kernel_size=(1, 1), padding="SAME")(input_x)
    conv5x5 = Conv2D(filters=conv5x5_filters, kernel_size=(5, 5), padding="SAME")(conv5x5)
    conv5x5 = ReLU()(conv5x5)

    # 3x3 max pooling and 1x1 convolution
    pool_proj = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="SAME")(input_x)
    pool_proj = Conv2D(filters=pool_proj_filters, kernel_size=(1, 1), padding="SAME")(pool_proj)
    pool_proj = ReLU()(pool_proj)

    # concatenate outputs of each convolution in an inception architecture
    inception_output = Concatenate()([conv1x1, conv3x3, conv5x5, pool_proj])

    return inception_output


def googlenet():
    """
    Description:
        GoogLeNet 구조를 구현하기 위한 함수입니다.
        모델 입력 shape: (n, 224, 224, 3)
        모델 리턴 shape: (n, 1000)

    :return:
        model: (n, 224, 224, 3) shape 의 데이터를 입력받아 (n, 1000) 를 리턴하는 모델을 생성합니다.
    """
    model_input = Input(shape=(224, 224, 3))

    # depth 1
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="SAME")(model_input)
    max1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="SAME")(conv1)

    # depth 2
    conv2 = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding="SAME")(max1)
    max2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="SAME")(conv2)

    # depth 3
    inception3a = inception(max2, 64, 96, 128, 16, 32, 32)
    inception3b = inception(inception3a, 128, 128, 192, 32, 96, 64)

    # depth 4
    max4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="SAME")(inception3b)
    inception4a = inception(max4, 192, 96, 208, 16, 48, 64)
    inception4b = inception(inception4a, 160, 112, 224, 24, 64, 64)
    inception4c = inception(inception4b, 128, 128, 256, 24, 64, 64)
    inception4d = inception(inception4c, 112, 144, 288, 32, 64, 64)
    inception4e = inception(inception4d, 256, 160, 320, 32, 128, 128)

    # depth 5
    max5 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="SAME")(inception4e)
    inception5a = inception(max5, 256, 160, 320, 32, 128, 128)
    inception5b = inception(inception5a, 384, 192, 384, 48, 128, 128)

    # output
    global_avg = GlobalAveragePooling2D()(inception5b)
    drop = Dropout(rate=0.4)(global_avg)
    relu = ReLU()(drop)
    output = Dense(1000, activation="softmax")(relu)

    model = Model(model_input, output)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # n의 개수
    number = 1

    # 가상의 입력 데이터 생성
    input_array = np.random.rand(number, 224, 224, 3)

    # 가상의 라벨 생성
    y = np.random.randint(1000, size=number)
    y = to_categorical(y, 1000)

    # 모델 생성
    googlenet_model = googlenet()

    # 모델 학습
    history = googlenet_model.fit(x=input_array, y=y, epochs=1)
    pass
