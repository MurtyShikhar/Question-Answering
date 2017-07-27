class Config:
	num_epochs = 10
	batch_size = 32
	train_embeddings=0
	max_gradient_norm=10.0
	hidden_state_size=150
    dropout=0.4
	embedding_size=300
	data_dir="data/squad"
	vocab_path="data/squad/vocab.dat"
	embed_path="data/squad/glove.trimmed.300.npz"
    dropout_val=0.4


	def get_paths(mode):
		question = "data/squad/%s.ids.question" %mode
	 	context = "data/squad/%s.ids.context" %mode
	 	answer = "data/squad/%s.span" %mode

	 	return question, context, answer 

	question_train, context_train, answer_train = get_paths("train")
 	question_dev ,context_dev ,answer_dev = get_paths("val")
