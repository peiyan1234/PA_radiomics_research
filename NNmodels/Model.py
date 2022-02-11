import tensorflow as tf
import tensorflow_probability as tfp

print(f"Tensorflow API Version: {tf.version.VERSION}")
print(f"Tensorflow API Compiler Version: {tf.version.COMPILER_VERSION}")

import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import datetime

from sklearn import metrics

class SNeL_RFS_ANNC():
    """
    An implementation of artifical neural network 
    with a constrained sparse layer for radiomics feature selection
    """

    def __init__(self, cfg=None, store=None, gpu=None, batchsize=None, learning_rate=None, ID=None):
        """
        Network Constructor
        """
        
        self.ID = ID
        self.cfg = cfg
        self.store = store
        self.gpu   = gpu

        #os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.gpu}"
        self.batchsize = batchsize
        self.learning_rate = learning_rate
    
    def __del__(self):

        print("the instance of SNeL_RFS_ANNC was deleted.")
    
    def initModel(self):

        self.params = []

        self.initGPU()
        self.init_tf_session()

        self.lambda_s = np.float64(0.1)#tf.constant(value=0.1, dtype=tf.float64)
        self.lambda_a = np.float64(0.1)#tf.constant(value=0.1, dtype=tf.float64)

        self.layers = self.cfg.sections()
        
        input_layer  = self.layers[0]
        input_size = self.cfg.getint(input_layer, 'width')

        self.x_ = tf.compat.v1.placeholder(shape=[None,input_size], dtype=tf.dtypes.float64)
        self.y_ = tf.compat.v1.placeholder(shape=[None], dtype=tf.dtypes.int32)
        
        self.construct_NN()

        self.loss, self.loss_softmax = self.loss_func(flag="Train")
        self.loss_val, self.loss_softmax_val = self.loss_func(flag="Validate")

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.store, f'logs/{current_time}') 

        self.step = -1

        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.train_op = self.backpropagation()
        
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                              max_to_keep=5, pad_step_number=True)

        if ( not os.path.isdir(log_dir) ):
            os.makedirs(log_dir)

        file_writer = tf.compat.v1.summary.FileWriter(log_dir, self.sess.graph)
    
    def initGPU(self):

        if ( self.gpu is not None ):
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.gpu}"
                self.config = tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False)
                self.config.gpu_options.allow_growth = True
            except RuntimeError as e:
                print(e)

    def init_tf_session(self):

        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session(config=self.config)
    
    def init_variables(self):

        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
    
    def construct_NN(self):

        input_layer  = self.layers[0]
        output_layer = self.layers[-1]

        input_size = self.cfg.getint(input_layer, 'width')
        output_size= self.cfg.getint(output_layer,'class')
        
        for n in range( 1, len(self.layers) ):

            if (1==n):
                output_size= self.cfg.getint(self.layers[n], 'size')
                if ( 'SNeL_FS' == self.layers[n] ):
                    weights_initializer = tf.constant_initializer(value= 0.5 / float(input_size) )

                    self.params.append(
                        tf.Variable(
                        weights_initializer(shape=[input_size, output_size], dtype=tf.dtypes.float64),
                        name=self.layers[n].replace(" ", "_") )
                    )

            elif ( n < len(self.layers)-1 ): # hidden layers
                input_size = self.cfg.getint(self.layers[n-1], 'size')
                output_size= self.cfg.getint(self.layers[n], 'size')
                if ( 'hidden layer' in self.layers[n] ):
                    
                    self.params.append(
                        tf.Variable(
                        tf.random.truncated_normal(shape=[input_size, output_size], dtype=tf.dtypes.float64,
                                          mean=0, stddev=1),
                        name=self.layers[n].replace(" ", "_") )
                    )

                    self.params.append(
                        tf.Variable(
                        tf.random.truncated_normal(shape=[1, output_size], dtype=tf.dtypes.float64,
                                          mean=0, stddev=1),
                        name=f'{self.layers[n].replace(" ", "_")}_bias')
                    )

            else: # output layer
                input_size = self.cfg.getint(self.layers[n-1], 'size')
                output_size= self.cfg.getint(output_layer,'class')

                self.params.append(
                    tf.Variable(
                    tf.random.truncated_normal(shape=[input_size, output_size], dtype=tf.dtypes.float64,
                                      mean=0, stddev=1),
                    name=self.layers[n].replace(" ", "_") )
                )

                self.params.append(
                    tf.Variable(
                    tf.random.truncated_normal(shape=[1, output_size], dtype=tf.dtypes.float64,
                                          mean=0, stddev=1),
                    name=f'{self.layers[n].replace(" ", "_")}_bias'
                )
                )

    def NN_forward(self, is_train=True, reuse=None):
        
        with tf.compat.v1.variable_scope(name_or_scope="SNeL_RFS_ANNC_model", reuse=reuse):

            depth = 0
            for n in range(1, len(self.layers)):

                if (1==n):
                    A_k = tf.linalg.matmul( self.x_, self.params[depth] )
                    self.A_k = A_k
                    Z = tf.identity(A_k)
                    depth+=1
                elif ( n < len(self.layers)-1 ): # hidden layers
                    if ( A_k is None ):
                        print("NN_forward not working, A_k are still not assigned values")
                        exit()

                    Z = tf.linalg.matmul( Z, self.params[depth] ) + self.params[depth+1]
                    depth+=2
                    if ( 'hidden layer' in self.layers[n] and 'relu' == self.cfg.get(self.layers[n], 'activation') ):
                        Z = tf.nn.relu(Z)

                else: # output layer
                    if ( Z is None ):
                        print("NN_forward not working, Z is still not assigned values")
                        exit()

                    Logits = tf.linalg.matmul( Z, self.params[depth] ) + self.params[depth+1]
                    if ( 'softmax' == self.layers[n] ):
                        Logits = tf.nn.softmax(Logits)

            self.logits = Logits
            self.pred = tf.math.argmax(Logits, 1)

    def adjust_learning_rate(self, lr=None):

        self.learning_rate = np.float64(lr)
    
    def set_KKTmultipliers(self, lambda_s = None, lambda_a = None):

        self.lambda_s = np.float64(lambda_s) #tf.constant(value=lambda_s, dtype=tf.float64)
        self.lambda_a = np.float64(lambda_a) #tf.constant(value=lambda_a, dtype=tf.float64)
    
    def loss_func(self, flag=None):

        def cal_KKT_omega_s(gpu_id=None, params=None, lambda_s=None):

            A = tf.math.reduce_sum(params , axis=0)
            A = tf.math.subtract(A, tf.constant( 1, dtype=tf.dtypes.float64 ) )
            A = tf.math.maximum(tf.constant( 0, dtype=tf.dtypes.float64 ), A)
            omega_s = tf.math.reduce_sum(A)
            KKT_omega_s = tf.math.multiply(lambda_s, omega_s)

            return KKT_omega_s

        def cal_KKT_omega_a(gpu_id=None, params=None, lambda_a=None):

            W = params
            Var_matrix = tf.matmul( tf.matmul( a=W, b=tfp.stats.covariance(x=self.x_), transpose_a=True), W)
            Var_matrix = tf.math.subtract(tf.constant( 1, dtype=tf.dtypes.float64 ), Var_matrix)
            Var_matrix = tf.math.maximum(tf.constant( 0, dtype=tf.dtypes.float64 ), Var_matrix)
            omega_a = tf.linalg.trace(Var_matrix)
            KKT_omega_a = tf.math.multiply(lambda_a, omega_a)

            return KKT_omega_a

        if ( "Train" == flag ):
            self.NN_forward(is_train=True, reuse=False)
            
        elif ( "Validate" == flag ):
            self.NN_forward(is_train=True, reuse=True)
        else:
            print("Wrong flag for loss_func input argument")
            exit()

        loss_softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits)
        loss_KKTpart  = cal_KKT_omega_s(self.gpu, self.params[0], self.lambda_s) + cal_KKT_omega_a(self.gpu, self.params[0], self.lambda_a)
        loss = loss_softmax + loss_KKTpart

        return loss, loss_softmax
    
    def backpropagation(self):

        def calculate_grad_omega_s(gpu_id=None, params=None):
            W      = params
            W_k    = tf.math.reduce_sum(tf.abs(W), axis=0)
            W_cond = tf.broadcast_to(W_k, W.shape)
            comparison = tf.math.less_equal( W_cond, tf.constant( 1, dtype=tf.dtypes.float64 ) )
            return tf.where( comparison, tf.zeros_like(W), tf.math.sign(W) )

        def calculate_grad_omega_a(gpu_id=None, params=None, A_k = None):
            W          = params
            Var_A_k    = tf.math.square( A_k )
            Var_A_k_cond = tf.broadcast_to( Var_A_k, W.shape )
            comparison = tf.math.greater_equal( Var_A_k_cond, tf.constant( 1, dtype=tf.dtypes.float64 ) )
            derivates = tf.math.multiply( tf.constant( -2, dtype=tf.dtypes.float64 ), 
                                        tf.matmul( a=self.x_, b=self.A_k, transpose_a=True) )
            return tf.where( comparison, tf.zeros_like(W), derivates )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(tf.compat.v1.trainable_variables())
            loss, loss_softmax= self.loss_func(flag="Train")
            self.params = tf.compat.v1.trainable_variables()

        grads = tape.gradient(loss_softmax, self.params)
        grad_omega_s = calculate_grad_omega_s(self.gpu, self.params[0])
        grad_omega_a = calculate_grad_omega_a(self.gpu, self.params[0], self.A_k)
        if (grads is not None and grads[0] is not None):
            grads[0] = grads[0] + grad_omega_s + grad_omega_a
        
        return optimizer.apply_gradients(zip(grads, self.params))

    def run_train(self, x=None, label=None, step=0):

        self.step=step

        feed = {self.x_ : x, 
                self.y_ : label}

        _,loss,loss_softmax = self.sess.run([self.train_op, 
                                                self.loss, self.loss_softmax], 
                                                feed)
        
        return loss,loss_softmax
    
    def run_eval(self, X=None, Y=None, best_val_f1=0.0, best_report=None, step=0):
        
        preds = []
        labels = []
        for i in range( Y.size ):
            x = X[i, :]
            y = Y[i, :]

            feed = {self.x_ : np.reshape(x, (1, x.size)), 
                    self.y_ : y+1}
            
            loss_val, pred = self.sess.run([self.loss_val, self.pred], feed)

            if ( pred == 0 ):
                preds.append("Left")
            elif ( pred == 1 ):
                preds.append("Bilateral")
            else:
                preds.append("Right")

            if ( y+1 == 0 ):
                labels.append("Left")
            elif ( y+1 == 1 ):
                labels.append("Bilateral")
            else:
                labels.append("Right")

        report = metrics.classification_report(labels, preds, digits=3, output_dict=True)
        print(metrics.classification_report(labels, preds, digits=3))

        avg_f1_score = report["accuracy"]
        
        if ( avg_f1_score >= best_val_f1 ):
            best_val_f1 = avg_f1_score
            best_report = report
            self.saver.save(self.sess, os.path.join(self.store, "tf_val_ckp"), global_step=step )
        
        return best_val_f1, best_report

    def calculate_metrics(self,tp,tn,fn,fp):
        tpr = float(tp) / (float(tp) + float(fn))
        tnr = float(tn) / (float(tn) + float(fp))
        fpr = float(fp) / (float(fp) + float(tn))
        fnr = float(fn) / (float(fn) + float(tp))
        accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
        recall = tpr
        precision = float(tp) / (float(tp) + float(fp))
        f1_score = (2 * (precision * recall)) / (precision + recall)
        return [tpr,tnr,fpr,fnr,accuracy,recall,precision,f1_score]







