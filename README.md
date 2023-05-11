# Usage Instructions

1. Install Python 3.9 or higher. You can either use a virtual environment or your global environment. I have used global environment. For virtual environment, use `python3 -m venv venv` and activate it via command `./venv/bin/activate`.
2. Install the dependencies using `pip install -r requirements.txt`
3. Run the program using `python3 main.py`. This will start the fastAPI server on `localhost:8000`. You can browse to [http://localhost:8000](http://localhost:8000) for accessing the app.
4. The program will automatically identify if GPU is available and can use it for inference.
5. I have used the [Salesforce's BLIP Model](https://huggingface.co/Salesforce/blip-image-captioning-base) for inference. BLIP stands for ***Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation***. You can also try a different image captioning model by changing the appropriate lines initializing the model and tokenizer in `main.py`. You can read more about the model in their [paper](https://arxiv.org/abs/2201.12086).

> Note: The program will download the model and tokenizer if they are not already present. This might take a while for the first run. BLIP is **~990 MiB** in size. Have some patience while executing the program :) . Also, if you don't have a GPU the requests might take more time to process.