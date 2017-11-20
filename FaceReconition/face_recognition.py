from keras import backend as K
K.set_image_data_format('channels_first')
from inception_blocks import *
from keras.models import load_model
from fr_utils import load_weights_from_FaceNet, img_to_encoding


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0))

    return loss


def compute_sim(image_path_1, image_path_2, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path_1 -- path to an image1
    image_path_2 -- path to an image2
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path_1 and image_path_2.
    """

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding_1 = img_to_encoding(image_path_1, model)
    encoding_2 = img_to_encoding(image_path_2, model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding_1 - encoding_2)

    return dist


if __name__ == '__main__':
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weights_from_FaceNet(FRmodel)

    print(compute_sim('./images/arnaud.jpg', './images/benoit.jpg', FRmodel))

