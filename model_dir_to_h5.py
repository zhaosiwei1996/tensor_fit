import tensorflow as tf

# 首先，从目录加载模型
model = tf.keras.models.load_model('/Users/wubo/Downloads/zhaosiwei/sign-language-model.h5/')

# 然后，保存模型为一个 .h5 文件
model.save('./sign-language-model.h5_20230717')
