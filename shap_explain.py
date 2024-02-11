import shap
import numpy as np
from pathlib import Path

# https://coderzcolumn.com/tutorials/artificial-intelligence/explain-text-classification-models-using-shap-values-keras

html_path = Path("results/shap/shap.html")
png_path = Path("results/shap/shap.png")


def save_to_files(explanation):
    if Path(html_path):
        with open(html_path, "w", encoding="utf-8") as html_file:
            pass
        print(f"cleared {html_path}")

    with open(html_path, "a+", encoding="utf-8") as html_file:
        html_file.write(shap.plots.text(explanation, display=False))
    print(f"saved to {html_path}")

    # ----------------png plot giving index error-----------------
    # fig, axs = plt.subplots(1, len(categories), layout="constrained")

    # for i in range(len(categories)):
    #     print(f'Plotting {categories[i]}...')
        
    #     plt.sca(axs[i])
    #     axs[i].set_title(categories[i])
    #     # plt.figure(figsize=(20,20))
    #     shap.plots.bar(explanation[:, :, categories[i]].mean(axis=0),
    #                     max_display=len(sentence.split()),
    #                     order=shap.Explanation.argsort.flip,
    #                     # order=shap.Explanation.abs,
    #                     show=False)

    # fig.set_size_inches(6*len(categories), 10)
    # # plt.show()
    # plt.savefig(png_path)
    # print(f'saved to {png_path}')


def get_shap_weights(pipeline, class_names, sentence, class_name):
    explainer = shap.Explainer(pipeline.predict, masker=shap.maskers.Text(tokenizer=r"\b\w+\b"), output_names=class_names)

    explanation = explainer([sentence])

    squeezed_values = np.squeeze(explanation.values)
    print(f"type(explanation) : {type(explanation)}")
    print(f"SHAP Squeezed Values Shape : {squeezed_values.shape}")
    print(f"SHAP Base Values  : {explanation.base_values}")
    print(f"SHAP Data: {explanation.data[0]}")
    print(f"type SHAP Values : {type(squeezed_values)}")
    print(f"SHAP Values = {squeezed_values}")

    save_to_files(explanation)

    shap_array = np.squeeze(explanation.values)

    shap_weights = shap_array[:, class_names.index(class_name)].tolist()

    return shap_weights