import numpy as np
from tqdm import tqdm

def GenerateEmbeddingFeature(SequenceList, EmbeddingList, PaddingLength):
    """Generate an embedded feature matrix"""
    SampleFeature = []
    for counter in tqdm(range(len(SequenceList)), desc="Generating Embedding Feature"):
        PairFeature = [SequenceList[counter][0]]
        FeatureMatrix = [[0] * (len(EmbeddingList[0]) - 1) for _ in range(PaddingLength)]
        try:
            for counter3 in range(PaddingLength):
                for counter4 in range(len(EmbeddingList)):
                    if SequenceList[counter][1][counter3] == EmbeddingList[counter4][0]:
                        FeatureMatrix[counter3] = EmbeddingList[counter4][1:]
                        break
        except:
            pass
        PairFeature.append(FeatureMatrix)
        SampleFeature.append(PairFeature)
    return SampleFeature

def GenerateSampleFeature(InteractionList, EmbeddingFeature1, EmbeddingFeature2):
    """Generated sample feature"""
    SampleFeature1, SampleFeature2 = [], []
    for counter in tqdm(range(len(InteractionList)), desc="Generating Sample Feature"):
        Pair1, Pair2 = InteractionList[counter]
        for counter1 in range(len(EmbeddingFeature1)):
            if EmbeddingFeature1[counter1][0] == Pair1:
                SampleFeature1.append(EmbeddingFeature1[counter1][1])
                break
        for counter2 in range(len(EmbeddingFeature2)):
            if EmbeddingFeature2[counter2][0] == Pair2:
                SampleFeature2.append(EmbeddingFeature2[counter2][1])
                break
    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')
    return SampleFeature1.reshape(SampleFeature1.shape[0], SampleFeature1.shape[1], SampleFeature1.shape[2], 1), SampleFeature2.reshape(SampleFeature2.shape[0], SampleFeature2.shape[1], SampleFeature2.shape[2], 1)

def GenerateBehaviorFeature(InteractionPair, NodeBehavior):
    """Generative behavior feature"""
    SampleFeature1, SampleFeature2 = [], []
    for i in tqdm(range(len(InteractionPair)), desc="Generating Behavior Feature"):
        Pair1, Pair2 = InteractionPair[i]
        for m in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[m][0]:
                SampleFeature1.append([float(x) for x in NodeBehavior[m][1:]])
                break
        for n in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[n][0]:
                SampleFeature2.append([float(x) for x in NodeBehavior[n][1:]])
                break
    return np.array(SampleFeature1).astype('float32'), np.array(SampleFeature2).astype('float32')
