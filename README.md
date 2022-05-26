# Semantic Search

Exploring semantic simility algorithms in NLP to develop solutions in text search. Rather than looking for text match, look for sentences or phrases with similar meaning. 

# Setup

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yaml
    ```

2. Install the CLIP repository 
    ```
    pip install git+https://github.com/openai/CLIP.git
    ```

3. Download pretrained weights (conceptual_weights.pt) and place them in the weights directory. You can find the [COCO](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing) and [Conceptual Captions](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view?usp=sharing) pretrained models on Google Drive.

4. Launch your environment with `conda activate understanding` or `source activate understanding`

5. Place text files to search in `data` folder

6. Run `python -i main.py` to open an interactive Python session. Use the method `search` to search for phrases and sentences that appear in the text. For example, `search("The quick brown fox jumps over the lazy dog")`.
