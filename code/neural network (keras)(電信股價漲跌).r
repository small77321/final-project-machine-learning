#�w��tensorflow�禡�w
install.packages('tensorflow')
#�w��keras�禡�w
install.packages('keras')

#�ޥ�tensorflow�禡�w
library(tensorflow)
#�ޥ�keras�禡�w
library(keras)

#Ū���V�m���
training_data <- read.csv(file.choose(), header = TRUE)
X <- subset(training_data, select = -Class)
Y <- training_data$Class
#�ഫ��matrix��ƫ��A
X <- data.matrix(X)

#�]�w�üƺؤl
use_session_with_seed(0)

#�]�w���g�������c
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 2, activation = "sigmoid", input_shape = c(3))  %>% #��J�Ѽ�: 1��, ��X�Ѽ�: 1��, �u�ʨ禡
  layer_dense(units = 1, activation = "sigmoid") #��J�Ѽ�: 1��, ��X�Ѽ�: 1��, �u�ʨ禡

#�]�w���g�����ǲߥؼ�
model %>% compile(
  loss='mean_squared_error', #�̤p����~�t
  optimizer='sgd' #��פU��
)

#�V�m���g����
history <- model %>% fit(
  X, #��J�Ѽ�
  Y, #��X�Ѽ�
  epochs = 30000, #�V�m�^�X��
  batch_size = 1 #�v���ץ��v��
)

#��ܯ��g�����v����
model$get_weights()

#�N���ո�ƥN�J�ҫ��i��w��,�è��o�w�����G
results <- model %>% predict(
  X
)

#�e�{���p���G
print(results)
