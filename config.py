class Config:
	num_epochs = 10
	batch_size = 32
	train_embeddings=False
	max_gradient_norm=10.0
	hidden_state_size=100
	embedding_size=100
	data_dir="data/squad"
	vocab_path="data/squad/vocab.dat"
	embed_path="data/squad/glove.trimmed.100.npz"

	def get_paths(mode):
		question = "data/squad/%s.ids.question" %mode
	 	context = "data/squad/%s.ids.context" %mode
	 	answer = "data/squad/%s.span" %mode

	 	return question, context, answer 

	question_train, context_train, answer_train = get_paths("train")
 	question_dev ,context_dev ,answer_dev = get_paths("val")
