import re


def get_occlusion_weights(pipeline, labels, sentence, label, proba):
    words = re.findall(r'\b\w+\b', sentence)

    occlusion_weights = []

    for i in range(len(words)):
        occluded_sentence = ' '.join([words[j] if j != i else '' for j in range(len(words))])

        occluded_proba = pipeline.predict_proba(occluded_sentence).flatten()[labels.index(label)]

        # convert numpy array to float
        occlusion_weight = float(proba - occluded_proba)

        occlusion_weights.append(occlusion_weight)

    return occlusion_weights
