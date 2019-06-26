from sklearn.metrics.pairwise import cosine_similarity
import heapq
from utils import *

def cosSimilarityUser(data):
    # Calculate the Cosine Similarity Matrix
    user_similarity = cosine_similarity(data)

    # Preview the Similarity Matrix
    print(user_similarity[:5, :5])
    print(np.shape(user_similarity))
    print("=" * 120)
    return user_similarity


def predictUser(ratings, similarity, num_items):
    # The Average Rating Values for Each User
    mean_user_rating = np.repeat(np.array([ratings.mean(axis=1)]), num_items, axis=0).T

    # The Difference Between Each Rating Value and The Average Value
    ratings_diff = ratings - mean_user_rating

    # Calculate the Predicted Score
    pred = mean_user_rating + \
           np.dot(similarity, ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred


def cosSimilarityItem(data):
    item_similarity = cosine_similarity(data.T)
    return item_similarity


def predictItem(ratings, similarity, num_users):
    # The Average Rating Values for Each Item
    mean_item_rating = np.repeat(np.array([ratings.mean(axis=0)]), num_users, axis=0)

    # The Difference Between Each Rating Value and The Average Value
    ratings_diff = ratings - mean_item_rating

    # Calculate the Predicted Score
    pred = mean_item_rating + \
           np.dot(ratings_diff, similarity) / np.abs(similarity).sum(axis=1)

    return pred


def recItemsForOneUser(pred_array, train_array, user, num_rec):
    # Change Training Arrary Into Sparse Matrix
    train_matrix = sp.csr_matrix(train_array)

    # Get the Item IDs in the Training Data For the Specified User
    train_items_for_user = train_matrix.getrow(user).nonzero()[1]

    # Create A Dictionary with Key-Value Pairs as ItemID-PredictedValue Pair
    pred_dict_for_user = dict(zip(np.arange(train_matrix.shape[1]), pred_array[user]))

    # Remove the Key-Value Pairs used in Training
    for iid in train_items_for_user:
        pred_dict_for_user.pop(iid)

    # Select the Top-N Items in The Sorted List
    rec_list_for_user = heapq.nlargest(num_rec, pred_dict_for_user.items(), key=lambda tup: tup[1])

    # Get the Item ID List From the Top-N Tuples
    rec_item_list = [tup[0] for tup in rec_list_for_user]
    return rec_item_list


def calMetrics(train_array, test_array, pred_array, at_K=10):
    # Get All the User IDs in Test Dataset
    test_matrix = sp.coo_matrix(test_array)
    test_users = test_matrix.row
    test_matrix = test_matrix.tocsr()

    # List to Store the Precision/Recall Value for Each User
    precision_u_at_K = []
    recall_u_at_K = []

    # Loop for Each User
    for u in test_users:
        # Get the Recommendation List for the User in Consideration
        rec_list_u = recItemsForOneUser(pred_array, train_array, u, at_K)

        # Generate a Item ID List For Testing
        item_list_u = test_matrix.getrow(u).nonzero()[1]

        # Calculate the Precision and Recall Value for this User
        precision_u, recall_u = Precision_and_Recall(rec_list_u, item_list_u)

        # Save the Precision/Recall Values
        precision_u_at_K.append(precision_u)
        recall_u_at_K.append(recall_u)

    # Calculate the Average Precision/Recall Values Over All Users
    print("Precision@"+str(at_K)+": "+str(np.mean(precision_u_at_K)))
    print("Recall@"+str(at_K)+": "+str(np.mean(recall_u_at_K)))
    print("=" * 120)

if __name__ == '__main__':
    # Load Data
    train, test, num_users, num_items, uid_min, iid_min = loadData(test_size=0.2)
    train_array, test_array = train.toarray(), test.toarray()

    # Similarity And Prediction Matrices (User)
    # similarity_user_array = cosSimilarityUser(train_array)
    # pred_user_array = predictUser(train_array, similarity_user_array, num_items)
    # Similarity And Prediction Matrices (Item)
    similarity_item_array = cosSimilarityItem(train_array)
    pred_item_array = predictItem(train_array, similarity_item_array, num_users)

    # Recommendation
    # rec_list = recItemsForOneUser(pred_user_array, train_array, 257, 10)
    rec_list = recItemsForOneUser(pred_item_array, train_array, 257, 10)
    print("The Recommendation List for User Is: " + str(rec_list+iid_min))
    print("=" * 120)

    # Metrics Calculation
    # calMetrics(train_array, test_array, pred_user_array, 5)
    calMetrics(train_array, test_array, pred_item_array, 5)
