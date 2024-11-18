import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from data_utils import ReadMyCsv1, ReadMyCsv3, StorFile
from feature_extraction import GenerateEmbeddingFeature, GenerateSampleFeature, GenerateBehaviorFeature
from model_utils import create_sae_model, create_attention_model
from train_and_evaluate import train_autoencoder, evaluate_classifier, TestOutput
import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Concatenate, Multiply
from joblib import dump
import numpy as np

def MyLabel(Sample):
    """Generate tag"""
    label = [1] * (len(Sample) // 2) + [0] * (len(Sample) // 2)
    return label

def main():
    print("Reading data files...")
    AllMiSequence, MiEmbedding, AllCircSequence, CircEmbedding, AllNodeBehavior = [], [], [], [], []
    ReadMyCsv1(AllMiSequence, 'database/miSequence_9905.csv')
    ReadMyCsv3(MiEmbedding, 'database/mi_fastext_embeddings.csv')
    ReadMyCsv1(AllCircSequence, 'database/circSequence_9905.csv')
    ReadMyCsv3(CircEmbedding, 'database/circ_fastext_embeddings.csv')
    ReadMyCsv1(AllNodeBehavior, 'database/AllNodeBehavior_grarep_9905.csv')

    PositiveSample_Train, PositiveSample_Validation, PositiveSample_Test = [], [], []
    ReadMyCsv1(PositiveSample_Train, 'database/PositiveSample_Train_9905.csv')
    ReadMyCsv1(PositiveSample_Validation, 'database/PositiveSample_Validation_9905.csv')
    ReadMyCsv1(PositiveSample_Test, 'database/PositiveSample_Test_9905.csv')

    NegativeSample_Train, NegativeSample_Validation, NegativeSample_Test = [], [], []
    ReadMyCsv1(NegativeSample_Train, 'database/NegativeSample_Train_9905.csv')
    ReadMyCsv1(NegativeSample_Validation, 'database/NegativeSample_Validation_9905.csv')
    ReadMyCsv1(NegativeSample_Test, 'database/NegativeSample_Test_9905.csv')

    print("Generating training, validation, and test pairs...")
    x_train_pair = PositiveSample_Train + NegativeSample_Train
    x_validation_pair = PositiveSample_Validation + NegativeSample_Validation
    x_test_pair = PositiveSample_Test + NegativeSample_Test

    print("Generating labels...")
    y_train_Pre = MyLabel(x_train_pair)
    y_validation_Pre = MyLabel(x_validation_pair)
    y_test_Pre = MyLabel(x_test_pair)

    print("Generating embedding features...")
    CircEmbeddingFeature = GenerateEmbeddingFeature(AllCircSequence, CircEmbedding, 64)
    MiEmbeddingFeature = GenerateEmbeddingFeature(AllMiSequence, MiEmbedding, 64)

    print("Generating sample features...")
    x_train_1_Attribute, x_train_2_Attribute = GenerateSampleFeature(x_train_pair, MiEmbeddingFeature,
                                                                     CircEmbeddingFeature)
    x_validation_1_Attribute, x_validation_2_Attribute = GenerateSampleFeature(x_validation_pair, MiEmbeddingFeature,
                                                                               CircEmbeddingFeature)
    x_test_1_Attribute, x_test_2_Attribute = GenerateSampleFeature(x_test_pair, MiEmbeddingFeature,
                                                                   CircEmbeddingFeature)

    print("Generating behavior features...")
    x_train_1_Behavior, x_train_2_Behavior = GenerateBehaviorFeature(x_train_pair, AllNodeBehavior)
    x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior)
    x_test_1_Behavior, x_test_2_Behavior = GenerateBehaviorFeature(x_test_pair, AllNodeBehavior)

    num_classes = 2
    y_train = keras.utils.to_categorical(y_train_Pre, num_classes)
    y_validation = keras.utils.to_categorical(y_validation_Pre, num_classes)
    y_test = keras.utils.to_categorical(y_test_Pre, num_classes)

    kf = KFold(n_splits=5)
    fold_index = 0

    for train_index, val_index in kf.split(x_train_pair):
        print(f"Starting fold {fold_index + 1}...")

        x_train_fold_pair = [x_train_pair[i] for i in train_index]
        x_val_fold_pair = [x_train_pair[i] for i in val_index]

        y_train_fold = [y_train_Pre[i] for i in train_index]
        y_val_fold = [y_train_Pre[i] for i in val_index]

        print("Generating sample features for current fold...")
        x_train_1_Attribute, x_train_2_Attribute = GenerateSampleFeature(x_train_fold_pair, MiEmbeddingFeature,
                                                                         CircEmbeddingFeature)
        x_validation_1_Attribute, x_validation_2_Attribute = GenerateSampleFeature(x_val_fold_pair, MiEmbeddingFeature,
                                                                                   CircEmbeddingFeature)

        print("Generating behavior features for current fold...")
        x_train_1_Behavior, x_train_2_Behavior = GenerateBehaviorFeature(x_train_fold_pair, AllNodeBehavior)
        x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_val_fold_pair, AllNodeBehavior)

        y_train_fold = keras.utils.to_categorical(y_train_fold, num_classes)
        y_val_fold = keras.utils.to_categorical(y_val_fold, num_classes)

        input_shape = (x_train_1_Attribute.shape[1], x_train_1_Attribute.shape[2], 1)
        autoencoder1, encoder1 = create_sae_model(input_shape)
        autoencoder2, encoder2 = create_sae_model(input_shape)

        print("Training the first stacked autoencoder...")
        train_autoencoder(autoencoder1, x_train_1_Attribute, x_validation_1_Attribute)

        print("Training the second stacked autoencoder...")
        train_autoencoder(autoencoder2, x_train_2_Attribute, x_validation_2_Attribute)

        x_train_1_Encoded = encoder1.predict(x_train_1_Attribute)
        x_train_2_Encoded = encoder2.predict(x_train_2_Attribute)
        x_validation_1_Encoded = encoder1.predict(x_validation_1_Attribute)
        x_validation_2_Encoded = encoder2.predict(x_validation_2_Attribute)
        x_test_1_Encoded = encoder1.predict(x_test_1_Attribute)
        x_test_2_Encoded = encoder2.predict(x_test_2_Attribute)


        behavior_shape = (x_train_1_Behavior.shape[1],)
        attention_model = create_attention_model(x_train_1_Encoded.shape[1:], x_train_2_Encoded.shape[1:],
                                                 behavior_shape)

        x_train_features = attention_model.predict(
            [x_train_1_Encoded, x_train_2_Encoded, x_train_1_Behavior, x_train_2_Behavior])
        x_validation_features = attention_model.predict(
            [x_validation_1_Encoded, x_validation_2_Encoded, x_validation_1_Behavior, x_validation_2_Behavior])
        x_test_features = attention_model.predict(
            [x_test_1_Encoded, x_test_2_Encoded, x_test_1_Behavior, x_test_2_Behavior])

        if len(y_train_fold.shape) > 1 and y_train_fold.shape[1] > 1:
            y_train_fold = np.argmax(y_train_fold, axis=1)
        if len(y_val_fold.shape) > 1 and y_val_fold.shape[1] > 1:
            y_val_fold = np.argmax(y_val_fold, axis=1)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(x_train_features)
        X_validation = scaler.transform(x_validation_features)
        X_test = scaler.transform(x_test_features)

        classifiers = {
            'MLP': MLPClassifier(max_iter=500),
        }

        for name, clf in classifiers.items():
            print(f"Training {name} classifier fold {fold_index + 1}...")
            clf.fit(X_train, y_train_fold)
            aupr, auc = TestOutput(clf, name, X_test, y_test, fold_index)
            auc_eval, aupr_eval = evaluate_classifier(clf, name, X_test, y_test)
            print(f"{name} classifier done fold {fold_index + 1}. AUC: {auc}, AUPR: {aupr}")
            dump(clf, f"{name}_model_fold_{fold_index}.joblib")

        fold_index += 1

if __name__ == '__main__':
    main()
