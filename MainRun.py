"""
Deep HypperSphere

author :zhengjian.002@163.com
 
"""


import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
tf.reset_default_graph()
import warnings
warnings.filterwarnings("ignore")

""" load data """
def Dataset():
        
    input_data=np.loadtxt("./InternetAds.txt")
    input_labels=np.loadtxt("./InternetAds_label.txt")
    
    train,test              =train_test_split(input_data,test_size=0.2)
    train_labels,test_labels=train_test_split(input_labels,test_size=0.2)
    
    num_dims = train.shape[1]
    time_step=1

	
    train_data = train.reshape((train.shape[0],time_step,train.shape[1]))
    test_data  = test.reshape((test.shape[0],time_step,test.shape[1]))

    return train_labels,test_labels,time_step,num_dims,train_data,test_data


train_label,test_label,time_step, num_dim, train_data,test_data=Dataset()


def variable_summaries(var):
    with tf.name_scope('summaries'):

        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar('mean', mean)
 
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)


def rmse(predictions, targets):
	return np.sqrt(np.mean((predictions - targets) ** 2))


class Config():
	def __init__(self, time_steps, num_dim):
		self.time_steps = time_steps
		self.num_dim = num_dim
		self.num_hidden = 50
		self.keep_prob = 0.98 
		self.lamda = 0.9
		self.gamma = 0.05
		self.tolerance = 1e-1
		self.num_epochs = 100
               
       
class DeepSphere():

	def weights(self, shape):
		return tf.compat.v1.get_variable(name="weight",shape=shape,dtype=tf.float32,initializer=tf.random_normal_initializer())

	def __init__(self, config):

		self.input = tf.compat.v1.placeholder(tf.float32,shape=[None,config.time_steps,config.num_dim])
		self.batch_size = tf.shape(self.input)[0]

		""" initialize centroid and radius neurons """
		self.centroid = tf.compat.v1.get_variable("centroid", [config.num_hidden], tf.float32, tf.random_normal_initializer())
		self.radius = tf.compat.v1.get_variable("radius", initializer=tf.constant(0.1))
        
        
		with tf.compat.v1.variable_scope("weights"):
			self.weights = self.weights(shape=[config.time_steps, 1])
			variable_summaries(self.weights)
			tf.compat.v1.summary.histogram('histogram', self.weights)

            
		rate=config.keep_prob
		with tf.compat.v1.variable_scope("encoder"):
			inputs = tf.nn.dropout(self.input, rate)
			inputs = tf.unstack(inputs,config.time_steps,1)
			""" run encode """
			self.enc_cell = tf.keras.layers.LSTMCell(config.num_hidden)

			enc_ouputs,enc_state = tf.nn.static_rnn(self.enc_cell,inputs,dtype=tf.float32)
			enc_ouputs = tf.stack(enc_ouputs,axis=0) 
			""" weighted sum """
			enc_outputs = tf.transpose(enc_ouputs, perm=[1, 2, 0]) 
			op = lambda x: tf.matmul(x,self.weights)
			z = tf.map_fn(op,enc_outputs)
			z = tf.squeeze(z)

		with tf.compat.v1.variable_scope("hypersphere_learning"):
			self.distance = tf.map_fn(tf.norm,
				(z - tf.reshape(tf.tile(self.centroid,[self.batch_size]),[self.batch_size, config.num_hidden])))
			distanceE2 = tf.square(self.distance)
			residue = tf.nn.relu(distanceE2 - tf.square(self.radius))
			penalty = tf.nn.relu(tf.exp(tf.square(self.radius) - distanceE2) - 1.0) + 1e-28
			penalty = tf.map_fn(lambda x: tf.divide(x,tf.reduce_sum(penalty)),penalty)
			self.penalty = penalty

			""" case-level label """
			self.label = (self.radius * (1.0 + config.tolerance) >= self.distance)

		with tf.compat.v1.variable_scope("decoder"):
			# inputs are zeros
			dec_inputs = [tf.zeros([self.batch_size,config.num_dim], dtype=tf.float32) for _ in range(config.time_steps)]
			self.dec_cell = tf.keras.layers.LSTMCell(config.num_hidden)	
            
			dec_output,dec_state = tf.nn.static_rnn(self.dec_cell,dec_inputs,initial_state=enc_state,dtype=tf.float32)			
			dec_output = tf.transpose(tf.stack(dec_output[::-1]),perm=[1,0,2]) 
			dec_output = tf.layers.dense(inputs=dec_output,units=config.num_dim,activation='sigmoid')


		with tf.compat.v1.variable_scope("loss"):
			""" recostruction error """
			self.rec_diff = self.input - dec_output 
			
			rec_error = tf.reduce_mean(tf.reduce_mean(tf.pow(self.rec_diff, 2), axis=1), axis=1) 
			penalized_rec_error = tf.reduce_mean(tf.multiply(rec_error, penalty))
			
			""" hypersphere learnong loss """
			hyper_loss = tf.square(self.radius) +config.gamma * tf.reduce_sum(residue) + tf.reduce_mean(distanceE2)
			self.loss = hyper_loss + config.lamda * penalized_rec_error


     
class MainDeep():
         
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    print("...................Run start.................\n")        
    config = Config(time_step,num_dim)
    model = DeepSphere(config)

    with tf.name_scope('train'):
        train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(model.loss)   
     
   

    with tf.compat.v1.Session () as sess:
        scalar=tf.summary.scalar('weights',model.weights)         
        merged = tf.compat.v1.summary.merge_all(scalar)
        sess.run(tf.compat.v1.global_variables_initializer())

        """ loop epoch """
        i=0
        for e in range(config.num_epochs):			
    #""" train """
			
            _,train_label_pred,_,R = sess.run([train_op, model.label, model.radius, model.distance,],feed_dict={model.input:train_data})	 
            train_acc = accuracy_score(train_label,train_label_pred)

            if e % 20 == 0:
                    print("\nepoch {}/{},train_acc = {:.6f}".format(e,config.num_epochs,train_acc))
                                    
		#""" test """
        _,test_label_pred, test_diff_pred = sess.run([model.loss, model.label, model.rec_diff],feed_dict={model.input:test_data})

    
    test_acc = accuracy_score(test_label,test_label_pred)
    F1_score = f1_score(test_label,test_label_pred,average='weighted')
    recall= recall_score(test_label,test_label_pred,average='weighted')
    
    mse = mean_squared_error(test_label,test_label_pred)
    
      
    print ("\n.........Testing results.........\n")
    
    print("..........Test accuracy............")
    print("test_acc = {:.4f}".format(test_acc))
    
    print("\n.........F1-score................")
    print("F1-score = {:.4f}".format(F1_score))
    
    print("\n.........recall................")
    print("recall = {:.4f}".format(recall))
    
    print("\n..........Average error..............")        
    print("Avaerage error = {:.4f}".format((mse)))
    
    print("\n..........Average Radius of Hypersphere............[Min, Max, Average]")
    print([np.amin(R),np.amax(R),np.average(R)])
    
    print("........End.......\n")
    
    
        
        
    
    
    
    

        

        