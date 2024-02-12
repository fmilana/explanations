import re


def get_occlusion_weights(pipeline, class_names, sentence, class_name, proba):
    words = re.findall(r"\b\w+\b", sentence)

    occlusion_weights = []

    for word in words:
        occluded_sentence = " ".join([w if w != word else "" for w in words])

        occluded_probas = []

        # average 10 predictions with occluded word
        for _ in range(10):
            occluded_proba  = pipeline.predict_proba([occluded_sentence])[class_names.index(class_name)]
            occluded_probas.append(occluded_proba)
        
        average_occluded_proba = sum(occluded_probas) / len(occluded_probas)

        # convert numpy array to float
        occlusion_weight = float(proba - average_occluded_proba)

        occlusion_weights.append(occlusion_weight)

    print(f"=================> {len(occlusion_weights)} occlusion_weights: {occlusion_weights}")

    return occlusion_weights
