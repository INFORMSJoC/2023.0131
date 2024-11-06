# coding: utf-8
import tensorflow as tf
import numpy as np
import math
import tensorflow as tf
from time import time
from _auxiliaryTools.ExtractData import Dataset
from _auxiliaryTools.Evaluation import get_test_list_mask
epsilon = 1e-9

def ini_word_embed(num_words, latent_dim):
    word_embeds = np.random.rand(num_words, latent_dim)
    return word_embeds

def word2vec_word_embed(num_words, latent_dim, path, word_id_dict):
    word2vect_embed_mtrx = np.zeros((num_words, latent_dim))
    with open(path, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            row_id = word_id_dict.get(arr[0])
            vect = arr[1].strip().split(" ")
            for i in range(len(vect)):
                word2vect_embed_mtrx[row_id, i] = float(vect[i])
            line = f.readline()

    return word2vect_embed_mtrx

def get_train_instance(train):
    user_input, item_input, rates = [], [], []

    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        rates.append(train[u,i])
    return user_input, item_input, rates

def get_train_instance_batch(count, batch_size, user_input, item_input, ratings, user_reviews, item_reviews, user_masks, item_masks):
    users_batch, items_batch, user_input_batch, item_input_batch, user_mask_batch, item_mask_batch, labels_batch= [], [], [], [], [], [], []

    for idx in range(batch_size):
        index = (count*batch_size + idx) % len(user_input)
        users_batch.append(user_input[index])
        items_batch.append(item_input[index])
        user_input_batch.append(user_reviews.get(user_input[index]))
        item_input_batch.append(item_reviews.get(item_input[index]))
        user_mask_batch.append(user_masks.get(user_input[index]))
        item_mask_batch.append(item_masks.get(item_input[index]))
        labels_batch.append([ratings[index]])

    return users_batch, items_batch, user_input_batch, item_input_batch, user_mask_batch, item_mask_batch, labels_batch
# user_batch, item_batch, user_input_batch, item_input_batch, user_mask_batch, item_mask_batch, rates_batch

def multi_head_cross_attention_layer(num_filters,cU,cI,rand_matrix):
    sec_dim = int(cU.get_shape()[1])
    tmpcU = tf.reshape(cU, [-1, num_filters])
    cU_mul_rand = tf.reshape(tf.matmul(tmpcU, rand_matrix), [-1, sec_dim, num_filters])
    f = tf.matmul(cU_mul_rand, cI, transpose_b=True)
    f = tf.expand_dims(f, -1)
    att1 = tf.tanh(f)

    pool_user = tf.reduce_mean(att1, 2)
    pool_item = tf.reduce_mean(att1, 1)

    user_flat = tf.squeeze(pool_user, -1)
    item_flat = tf.squeeze(pool_item, -1)
    weight_user = tf.nn.softmax(user_flat)
    weight_item = tf.nn.softmax(item_flat)

    weight_user_exp = tf.expand_dims(weight_user, -1)
    weight_item_exp = tf.expand_dims(weight_item, -1)

    weighted_U = cU * weight_user_exp
    weighted_I = cI * weight_item_exp

    return weighted_U, weighted_I

def asps_gate(word_dim, contextual_words, asps_embeds, W_word, W_asps, b_gate, drop_out):
    W_word = tf.nn.dropout(W_word, drop_out)
    W_asps = tf.nn.dropout(W_asps, drop_out)
    contextual_words_reshape = tf.reshape(contextual_words, [-1, num_filters])
    asps_gate = tf.reshape(
        tf.nn.sigmoid(tf.matmul(contextual_words_reshape, W_word) + tf.matmul(asps_embeds, W_asps) + b_gate),
        [-1, word_dim, num_filters])
    gated_contextual_words = contextual_words * asps_gate
    return gated_contextual_words

def asps_prj(word_num, gated_contextual_embeds, W_prj):
    gated_contextual_embeds_reshape = tf.reshape(gated_contextual_embeds, [-1, num_filters])
    return tf.reshape(tf.matmul(gated_contextual_embeds_reshape, W_prj), [-1, word_num, num_filters])

def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return (vec_squashed)

def simple_self_attn(word_dim, u_hat, input_masks, caps_b, b_ij, cap_1_b_inputs):
    third_dim = int(u_hat.get_shape()[2])

    u_hat = tf.reshape(u_hat, shape=[-1, word_dim, int(third_dim / num_filters), num_filters, 1])
    u_hat_stopped = tf.stop_gradient(u_hat)

    b_ij = tf.expand_dims(tf.expand_dims(cap_1_b_inputs * b_ij, -1), -1)

    for r_itr in range(itr_0):
        if (r_itr > 0):
            b_ij = b_ij * tf.expand_dims(tf.expand_dims(tf.expand_dims(input_masks, -1), -1), -1)
        c_ij = tf.nn.softmax(b_ij, dim=1)

        if r_itr == itr_0 - 1:
            s_j = tf.multiply(c_ij, u_hat)
            s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True) + caps_b
            v_j = squash(s_j)

        elif r_itr < itr_0 - 1:
            s_j = tf.multiply(c_ij, u_hat_stopped)
            s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True) + caps_b
            v_j = squash(s_j)

            v_j_tiled = tf.tile(v_j, [1, word_dim, 1, 1, 1])
            u_produce_v = tf.reduce_sum(u_hat_stopped * v_j_tiled, axis=3,
                                        keep_dims=True)

            b_ij += u_produce_v

    return tf.squeeze(v_j, [1, -1])

def pack_vects(vect_1, vect_2, type):
    if type == "mul":
        return tf.multiply(vect_1, vect_2)
    elif type == "sub":
        return tf.subtract(vect_1, vect_2)
    elif type == "both":
        return tf.concat([tf.multiply(vect_1, vect_2), tf.subtract(vect_1, vect_2)], 2)
    
import tensorflow as tf

# attention
def attention(inputs, attention_size, time_major=False, return_alphas=False):

    hidden_size = inputs.shape[2].value

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
    
def eval_model(users, items, users_inputs, items_inputs, users_masks_inputs, items_masks_inputs, dropout_rate,
               b_inputs, predict_rating, sess, user_tests, item_tests, user_input_tests,
               item_input_tests, user_mask_tests, item_mask_tests, rate_tests, b_test,rmses, maes):
    predicts = sess.run(predict_rating, feed_dict={users: user_tests, items: item_tests, users_inputs: user_input_tests,
                                                   items_inputs: item_input_tests, users_masks_inputs: user_mask_tests,
                                                   items_masks_inputs: item_mask_tests, b_inputs: b_test,
                                                dropout_rate: 1.0})
    row, col = predicts.shape
#     print(predicts)
    for r in range(row):
        rmses.append(pow((predicts[r, 0] - rate_tests[r][0]), 2))
        maes.append(abs(predicts[r, 0] - rate_tests[r][0]))
    return rmses, maes


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    latent_dim = 50#denote as k
    word_latent_dim = 300#denote as d in paper
    num_filters = 50#number of filters of convolution operation
    window_size = 3#denote as c in paper
    max_doc_length = 300
    learning_rate = 0.001
    epochs = 200
    batch_size = 200
    num_aspect = 5#denote as M in paper
    _EPSILON = 10e-9
    itr_0 = 2#number of iteration of self attention, we use 2
    lambda_1 = 0.001
    gama = 0.5#denote as λ in paper
    drop_out = 0.5
    v_dim = 50
    gama = 0.01
    
    # loading data
    firTime = time()
    dataSet = Dataset(max_doc_length, "./outputs/output_G/") #!!!!!在这里更改数据所在目录!!!!!
    word_dict, user_reviews, item_reviews, user_masks, item_masks, train, testRatings = dataSet.word_id_dict, dataSet.userReview_dict, dataSet.itemReview_dict, dataSet.userMask_dict, dataSet.itemMask_dict, dataSet.trainMtrx, dataSet.testRatings
    secTime = time()

    num_users, num_items = train.shape
    print ("load data: %.3fs" % (secTime - firTime))
    print (num_users, num_items)  
    
    #load word embeddings
    word_embedding_mtrx = ini_word_embed(len(word_dict), word_latent_dim)
    # word_embedding_mtrx = word2vec_word_embed(len(word_dict), word_latent_dim,
    #                                           "Directory of pretrained WordEmbedding.out",
    #                                           word_dict)

    print ("shape", word_embedding_mtrx.shape)

    # get train instances
    user_input, item_input, rateings = get_train_instance(train)

    # get test/val instances
    user_test, item_test, user_input_test, item_input_test, user_mask_test, item_mask_test, rating_input_test = get_test_list_mask(
        200, testRatings, user_reviews, item_reviews, user_masks, item_masks)
    #user_tests, item_tests, user_input_test, item_input_test, rating_input_test = get_test_list(200, testRatings, user_reviews, item_reviews)
    
    b_train = np.zeros([batch_size, 1, 1])
    
    # 变量定义及初始化
    users = tf.placeholder(tf.int32, shape=[None], name='users')
    items = tf.placeholder(tf.int32, shape=[None], name='items')
    users_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length], name='users_inputs')
    items_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length], name='items_inputs')
    users_masks_inputs = tf.placeholder(tf.float32, shape=[None, max_doc_length], name='users_masks_inputs')
    items_masks_inputs = tf.placeholder(tf.float32, shape=[None, max_doc_length], name='items_masks_inputs')
    b_inputs = tf.placeholder(tf.float32, shape=[None, 1, 1], name='b_inputs')
    ratings = tf.placeholder(tf.float32, shape=[None, 1], name='ratings')
    dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

    text_embedding = tf.Variable(word_embedding_mtrx, dtype=tf.float32)
    padding_embedding = tf.Variable(np.zeros([1, word_latent_dim]), dtype=tf.float32)

    word_embeddings = tf.concat([text_embedding, padding_embedding], 0)

    # interaction embedding
    user_bias = tf.Variable(tf.random_normal([num_users, 1], mean=0, stddev=0.02))
    item_bias = tf.Variable(tf.random_normal([num_items, 1], mean=0, stddev=0.02))
    user_bs = tf.nn.embedding_lookup(user_bias, users)
    item_bs = tf.nn.embedding_lookup(item_bias, items)

    
    #-----从这里开始是模型的核心部分-----
    # 1. Embedding Layer
    user_reviews_representation = tf.nn.embedding_lookup(word_embeddings, users_inputs)
    item_reviews_representation = tf.nn.embedding_lookup(word_embeddings, items_inputs)
    user_reviews_representation_expnd = tf.expand_dims(user_reviews_representation, -1)
    item_reviews_representation_expnd = tf.expand_dims(item_reviews_representation, -1)

    # 2. Multiple Convolution Operations
    W_conv_u = tf.Variable( tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3), name="review_W_u")
    W_conv_i = tf.Variable(tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3), name="review_W_i")
    convU = tf.nn.conv2d(user_reviews_representation_expnd, W_conv_u, strides=[1, 1, word_latent_dim, 1], padding='SAME')
    convI = tf.nn.conv2d(item_reviews_representation_expnd, W_conv_i, strides=[1, 1, word_latent_dim, 1], padding='SAME')
    cU = tf.nn.relu(tf.squeeze(convU, 2))
    cI = tf.nn.relu(tf.squeeze(convI, 2))

    #print("cU->>>>>>      ",cU)
    #print("cI->>>>>>      ",cI)
    
    # 3. Multi-head Cross-attention Layer
    rand_matrix = tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.3), name="review_rand_matrix")
    weighted_U, weighted_I = multi_head_cross_attention_layer(num_filters, cU, cI,rand_matrix)

    #print("weighted_U->>>>>>      ",weighted_U)
    #print("weighted_I->>>>>>      ",weighted_I)
    
    user_word_dim = int(weighted_U.get_shape()[1])
    item_word_dim = int(weighted_I.get_shape()[1])
    
    # 4. Bi-layer Gate Mechanism
    user_viewpoint_embeddings = [
        tf.Variable(tf.truncated_normal([1, num_filters], stddev=0.3), name="user_aspect_embedding_{}".format(i)) for i
        in range(num_aspect)]
    item_aspect_embeddings = [
        tf.Variable(tf.truncated_normal([1, num_filters], stddev=0.3), name="user_aspect_embedding_{}".format(i)) for i
        in range(num_aspect)]

    W_word_gate_u = [
        tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="user_word_gate_u_{}".format(i))
        for i in range(num_aspect)]
    W_asps_gate_u = [
        tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="user_asps_gate_u_{}".format(i))
        for i in range(num_aspect)]
    W_word_gate_i = [
        tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="item_word_gate_i_{}".format(i))
        for i in range(num_aspect)]
    W_asps_gate_i = [
        tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="item_asps_gate_i_{}".format(i))
        for i in range(num_aspect)]
    b_gate_u = [tf.Variable(tf.constant(0.1, shape=[num_filters]), name="gate_b_u_{}".format(i)) for i in
                range(num_aspect)]
    b_gate_i = [tf.Variable(tf.constant(0.1, shape=[num_filters]), name="gate_b_i_{}".format(i)) for i in
                range(num_aspect)]
    
    user_pre_attn_contextual_embeds, item_pre_attn_contextual_embeds = [], []
    for i in range(num_aspect):
        user_pre_attn_contextual_embeds.append(
            asps_gate(user_word_dim, weighted_U, user_viewpoint_embeddings[i], W_word_gate_u[i],
                      W_asps_gate_u[i], b_gate_u[i], dropout_rate))
        item_pre_attn_contextual_embeds.append(
            asps_gate(item_word_dim, weighted_I, item_aspect_embeddings[i], W_word_gate_i[i],
                      W_asps_gate_i[i], b_gate_i[i], dropout_rate))
        

    # 5. Aspect Self-Attention
    W_prj_u = [tf.Variable(tf.truncated_normal([num_filters, num_filters], mean=0, stddev=0.3),
                            name="user_project_W_{}".format(i)) for i in range(num_aspect)]
    b_prj_u = [tf.Variable(tf.constant(0.1, shape=[1, 1, 1, num_filters, 1]), name="user_project_b_{}".format(i)) for i in
               range(num_aspect)]
    b_ij_u = [tf.constant(np.zeros([1, max_doc_length, 1], dtype=np.float32), name="user_bij_{}".format(i)) for i in
              range(num_aspect)]

    W_prj_i = [tf.Variable(tf.truncated_normal([num_filters, num_filters], mean=0, stddev=0.3),
                            name="item_project_W_{}".format(i)) for i in range(num_aspect)]
    b_prj_i = [tf.Variable(tf.constant(0.1, shape=[1, 1, 1, num_filters, 1]), name="item_project_b_{}".format(i)) for i in
               range(num_aspect)]
    b_ij_i = [tf.constant(np.zeros([1, max_doc_length, 1], dtype=np.float32), name="item_bij_{}".format(i)) for i in
              range(num_aspect)]

    user_viewpoint_words = []
    item_viewpoint_words = []
    for i in range(num_aspect):
        user_viewpoint_words.append(asps_prj(user_word_dim, user_pre_attn_contextual_embeds[i], W_prj_u[i]))
        item_viewpoint_words.append(asps_prj(item_word_dim, item_pre_attn_contextual_embeds[i], W_prj_i[i]))
         
    user_viewpoint_embeddings = []
    item_aspect_embeddings = []
    for i in range(len(W_prj_u)):
        u_viewpoint_v = simple_self_attn(user_word_dim, user_viewpoint_words[i], users_masks_inputs,
                                                        b_prj_u[i], b_ij_u[i], b_inputs)
        user_viewpoint_embeddings.append(u_viewpoint_v)
    for i in range(len(W_prj_i)):
        i_asps_v = simple_self_attn(item_word_dim, item_viewpoint_words[i], items_masks_inputs,
                                                        b_prj_i[i], b_ij_i[i], b_inputs)
        item_aspect_embeddings.append(i_asps_v)

        
    # 6. Generating Aspect Pairs
    aspect_embeds_pack = []
    for i in range(num_aspect):
        user_asps_embed = user_viewpoint_embeddings[i]
        for j in range(num_aspect):
            item_asps_embed = item_aspect_embeddings[j]
            aspect_embeds_pack.append(pack_vects(user_asps_embed, item_asps_embed, "both"))
    aspect_pairs = tf.concat(aspect_embeds_pack, 1)
    num_intrn = int(aspect_pairs.get_shape()[1])
    cap1_latent_dim = int(aspect_pairs.get_shape()[-1])

    
    # 7. Joint-Attention
    ATTENTION_SIZE=50
    with tf.name_scope('Attention_layer'):
        attention_output, alphas = attention(aspect_pairs, ATTENTION_SIZE, return_alphas=True)
        tf.summary.histogram('alphas', alphas)
    print("attention_output->>>>>>      ",attention_output)

    
    # 8. Final Representation using FM and FNN
    W_mlp = tf.Variable(tf.random_normal([num_filters*2, latent_dim], mean=0, stddev=0.02), name="review_W_mlp")
    W_mlp = tf.nn.dropout(W_mlp, dropout_rate)
    b_mlp = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="review_b_mlp")

    aspect_pairs_embeds = tf.nn.relu(tf.matmul(attention_output, W_mlp) + b_mlp)
    print("aspect_pairs_embeds->>>>>>      ",aspect_pairs_embeds)
    
    w_0 = tf.Variable(tf.zeros(1), name="review_w_0")
    w_1 = tf.Variable(tf.truncated_normal([1, latent_dim], stddev=0.3), name="review_w_1")
    v = tf.Variable(tf.truncated_normal([latent_dim, v_dim], stddev=0.3), name="review_v")

    J_1 = w_0 + tf.matmul(aspect_pairs_embeds, w_1, transpose_b=True)
    print("J_1->>>>>>      ",J_1)

    embeds_1 = tf.expand_dims(aspect_pairs_embeds, -1)
    embeds_2 = tf.expand_dims(aspect_pairs_embeds, 1)


    J_2 = tf.reduce_sum(tf.reduce_sum(tf.multiply(tf.matmul(embeds_1, embeds_2), tf.matmul(v, v, transpose_b=True)),2), 1, keepdims=True)
    J_3 = tf.trace(tf.multiply(tf.matmul(embeds_1, embeds_2), tf.matmul(v, v, transpose_b=True)))
    J_total = (J_1 + 0.5 * (J_2 - J_3))
    
    # 预测评分
    predict_rating = J_total + user_bs + item_bs
    loss = tf.reduce_mean(tf.squared_difference(predict_rating, ratings))

    #print("predict_rating->>>>>>      ",predict_rating)
    
    loss +=lambda_1 * (tf.nn.l2_loss(rand_matrix) + tf.nn.l2_loss(user_bs) + tf.nn.l2_loss(item_bs))+ tf.nn.l2_loss(W_conv_u)+tf.nn.l2_loss(W_conv_i)+tf.nn.l2_loss(W_word_gate_u)+tf.nn.l2_loss(W_word_gate_i)+tf.nn.l2_loss(W_prj_u)+tf.nn.l2_loss(W_prj_i)
    #print("loss->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",loss)


# Model training and evaluation
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            t = time()
            loss_total = 0.0
            count = 0.0
            for i in range(int(math.ceil(len(user_input) / float(batch_size)))):
                user_batch, item_batch, user_input_batch, item_input_batch, user_mask_batch, item_mask_batch, rates_batch= get_train_instance_batch(
                    i, batch_size, user_input, item_input, rateings,user_reviews, item_reviews, user_masks, item_masks)
                _, loss_val = sess.run([train_step, loss],
                                                             feed_dict={users: user_batch, items: item_batch,
                                                                        users_inputs: user_input_batch,
                                                                        items_inputs: item_input_batch,
                                                                        users_masks_inputs: user_mask_batch,
                                                                        items_masks_inputs: item_mask_batch,
                                                                        ratings: rates_batch,
                                                                        b_inputs: b_train,
                                                                        dropout_rate: drop_out})
                loss_total += loss_val
                count += 1.0
            t1 = time()
            mses, maes = [], []
            for i in range(len(user_input_test)):
                b_test = np.zeros([len(user_input_test[i]),1, 1])
                eval_model(users, items, users_inputs, items_inputs, users_masks_inputs, items_masks_inputs,
                           dropout_rate, b_inputs, predict_rating, sess, user_test[i],
                           item_test[i], user_input_test[i],
                           item_input_test[i], user_mask_test[i], item_mask_test[i], rating_input_test[i], b_test, mses, maes)

            mse = np.array(mses).mean()
            mae = np.array(maes).mean()
            t2 = time()
            print ("epoch%d  time: %.3fs  loss: %.3f  test time: %.3fs  mse: %.3f  mae: %.3f"%(e, (t1 - t),loss_total/count,(t2 - t1), mse, mae))
            
            saver = tf.train.Saver()
            saver.save(sess, "model/model_{}/epoch{}".format(e, e))
            
            # 将对应的变量写入文件中
            f = open('var_epoch_{}.txt'.format(e), 'a')
            for i in range(len(user_input_test)):
                b_test = np.zeros([len(user_input_test[i]),1, 1])
                predicts = sess.run(predict_rating, feed_dict={users: user_test[i], items: item_test[i], users_inputs: user_input_test[i],
                                                               items_inputs: item_input_test[i], users_masks_inputs: user_mask_test[i],
                                                               items_masks_inputs: item_mask_test[i], b_inputs: b_test,
                                                               dropout_rate: 1.0})
            f.write("user items predict_rating")
            
            for (x, y, z) in zip(user_test[i], item_test[i], predicts.tolist()):
                print(x, y, z[0], file=f)
            
            print(rand_matrix, file=f)
            
            f.close()
#             break

        # 保存最终的模型
#             saver.save(sess, "model/v1")
