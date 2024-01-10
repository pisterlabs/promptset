from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet
import pickle
import sys
import os
sys.path.append('scripts/analysis')
from analysis import file_utils  # nopep8
import coherence_utils  # nopep8


def create_all_coherence_models(models_folder: str, out_file: str, dictionary, texts):

    models_paths = [
        f.path
        for f in os.scandir(models_folder)
        if f.is_dir()
    ]

    for path in models_paths:
        model_name = "{}/model".format(path)

        if not os.path.isfile:
            continue

        lda = LdaMallet.load(model_name)
        result_data = coherence_utils.create_coherence_model(
            lda, dictionary, texts
        )

        coherence_utils.write_values_to_file(out_file, result_data)

        print("[coherence] created for {}".format(model_name))


if __name__ == "__main__":
    out_folder = "results/coherence"
    out_file = "{}/coherence_values.txt".format(out_folder)

    file_utils.recreate_folder(out_folder)
    coherence_utils.write_header(out_file)

    models_folder = "data/models"

    dictionary = pickle.load(open("data/dictionary.pickle", "rb"))
    texts = pickle.load(open("data/discursos_preproc.pickle", "rb"))

    print("[main] dict, texts fetched")

    create_all_coherence_models(
        models_folder, out_file, dictionary, texts
    )

    print("[coherence] done")
