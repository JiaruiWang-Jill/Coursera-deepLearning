# Neural network = encoding f(x)

def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist =  tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.maximum((basic_loss), 0.0)
	return loss



def who_is_it(image_path, database, model):
	encoding = img_to_encoding(image_path, model)
	min_dist = 100
	for (name, db_enc) in database.items():
		dist = np.linalg.norm(encoding - database[name])
		if dist < min_dist:
			min_dist = dist
			identity = name

	if min_dist > 0.7:
		print("not in database")
	else:
		print("it's " + str(identity) + ", the distance is, " + str(min_dist))
	return min_dist, identity


 