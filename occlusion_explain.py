import re


def get_occlusion_weights(pipeline, class_names, sentence, class_name, proba):
    words = re.findall(r'\b\w+\b', sentence)

    occlusion_weights = []

    for word in words:
        occluded_sentence = ' '.join([w if w != word else '' for w in words])

        occluded_proba  = pipeline.predict_proba([occluded_sentence])[class_names.index(class_name)]
        # convert numpy array to float
        occlusion_weight = float(proba - occluded_proba)

        occlusion_weights.append(occlusion_weight)

    print(f'=================> {len(occlusion_weights)} occlusion_weights: {occlusion_weights}')

    return occlusion_weights
